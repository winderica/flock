//! **THE DRIVER**: one call from a chain statement to the tower's
//! converged root — the production caller for the pipeline the e2e tests
//! used to hand-assemble.
//!
//! [`Tower::prove`] runs the whole tower over a BLAKE3 compression chain:
//! sequential LEAVES (each segment starts at the previous one's `h_end`),
//! adjacent pairs into FIRST-LEVEL nodes, then the SPINE — the BASE folds
//! the LAST two FLs and every level above prepends the next-earlier FL as
//! its fresh child, the chain LANE riding every level. From the second
//! spine fold on the circuit digest must not move (ONE steady shape —
//! asserted in the loop).
//!
//! [`Tower::discharge_root`] settles the ROOT-SIDE RESIDUE: the carried
//! accumulators against the native tables, the passenger against the
//! base's own tables, the statement against the root's publics. It is
//! the PROVER'S SELF-CHECK over its own in-memory objects — the
//! consumer's check is `verify_root` over [`Tower::root_bundle`]'s
//! artifacts, which re-runs these same legs over accumulators
//! REASSEMBLED from the bundle's publics, after the native circuit
//! verify.

use flock_core::{
    aggregate::Accumulator, circuit::Circuit, lincheck::LincheckCircuit, matrix_fold::MatrixClaim,
    pcs::jagged::JaggedParams,
};

use crate::{
    proof_io::TowerRootBundle,
    tower::{
        F128, TowerLeaf,
        chain::{build_chain_proof, native_chain},
        config::TowerConfig,
        fl_node::{FlNode, build_first_level_node_kids, combine_digests, inner_jagged_params},
        gates_blake3::{pack4, unpack4},
        node::{
            AppKind, ChainLane, NodeOut, SpineIn, build_node_outer_app_kind, digest_f128,
            entry_live, node_jagged_params,
        },
        online::census_kib,
        query::leaf_boolean_mats,
        span_count_word,
        verify::RootBundle,
    },
};

/// A tower's application statement, KIND-ERASED: the composed app block's
/// 8 packed F128 words — exactly what the root publishes, in the same
/// layout its [`AppKind`] produces — plus the span-counter count.
/// [`ChainStatement`] (here) and a digest workload's own statement type,
/// defined in that workload's own crate, are workload-specific VIEWS onto
/// the same 8 words; `to_app`/`from_app` convert between a view and this
/// wire form. [`Tower::discharge_root`]
/// and [`crate::tower::verify::verify_root`] compare `words` against the
/// root's own published app block word for word — no workload match, so a
/// third workload's leaf needs no change here.
#[derive(Copy, Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct AppStatement {
    pub words: [F128; 8],
    pub count: usize,
}

/// What a tower run attests: `h_end == H^{n_blocks}(h_start)` over the
/// BLAKE3 compression chain — endpoints AND count. The count rides the
/// app block's ninth word as the span counter `g^{n_blocks}` (the FL
/// bakes its shape-bound value; every node multiplies its children's),
/// so a standalone verifier pins the whole statement at every depth.
#[derive(Copy, Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ChainStatement {
    pub h_start: [u32; 16],
    pub h_end: [u32; 16],
    /// TOTAL compressions across the span: `n_leaves * blocks_per_leaf`.
    pub n_blocks: usize,
}

impl ChainStatement {
    /// This statement's app words, packed exactly as the root's chain FL
    /// publishes them: `[h_start x4 | h_end x4]`.
    pub fn to_app(&self) -> AppStatement {
        let word =
            |h: &[u32; 16], j: usize| pack4(h[4 * j..4 * j + 4].try_into().expect("4 words"));
        let mut words = [F128::ZERO; 8];
        for j in 0..4 {
            words[j] = word(&self.h_start, j);
            words[4 + j] = word(&self.h_end, j);
        }
        AppStatement {
            words,
            count: self.n_blocks,
        }
    }
    /// The inverse of [`Self::to_app`].
    pub fn from_app(a: &AppStatement) -> Self {
        let decode = |base: usize| -> [u32; 16] {
            let mut out = [0u32; 16];
            for j in 0..4 {
                out[4 * j..4 * j + 4].copy_from_slice(&unpack4(a.words[base + j]));
            }
            out
        };
        ChainStatement {
            h_start: decode(0),
            h_end: decode(4),
            n_blocks: a.count,
        }
    }
}

/// The digest tree in the TOWER'S FOLD ORDER, natively: leaves pair into
/// FLs left to right; the BASE folds the last two FLs; every level above
/// prepends the next-earlier FL. So the root is
/// `C(fl_0, C(fl_1, … C(fl_{k-2}, fl_{k-1})))` with `fl_i = C(l_{2i}, l_{2i+1})`.
/// Generic over any [`AppKind::Digest`] workload's per-leaf digests — a
/// digest workload built as a separate crate outside this one uses this
/// to compute its own tower's expected root digest natively.
pub fn native_digest_tree(leaf_digests: &[[u32; 8]]) -> [u32; 8] {
    assert!(
        leaf_digests.len() >= 4 && leaf_digests.len().is_multiple_of(2),
        "a rooted tower: an even number of leaves, at least four"
    );
    let fls: Vec<[u32; 8]> = leaf_digests
        .as_chunks::<2>()
        .0
        .iter()
        .map(|[a, b]| combine_digests(a, b))
        .collect();
    let k = fls.len();
    let mut cur = combine_digests(&fls[k - 2], &fls[k - 1]);
    for i in (0..k - 2).rev() {
        cur = combine_digests(&fls[i], &cur);
    }
    cur
}

/// Which root-side discharge refused — see [`Tower::discharge_root`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RootDischargeFailure {
    /// The root's app block does not carry the statement's span.
    Statement,
    /// The chain lane's boolean claims vs the chain BLAKE3 matrices.
    ChainLaneBoolean,
    /// The chain lane grew an element group it cannot have.
    ChainLaneElement,
    /// The chain lane's sigma claims vs the chain circuit's wiring.
    ChainLaneSigma,
    /// The chain lane's jagged claims vs the chain layout.
    ChainLaneJagged,
    /// The main fold's boolean claims vs the outer registry's matrices.
    Boolean,
    /// The main fold's element claims vs the outer element types.
    Element,
    /// The main fold's sigma claims vs the outer circuits' wiring.
    Sigma,
    /// The main fold's jagged claims vs the outer layouts.
    Jagged,
    /// The passenger: missing when owed, riding when not, or refusing the
    /// base's tables.
    Passenger,
}

/// A completed tower run: the ROOT (one recursable proof over the whole
/// chain) plus the material owners the root-side discharge needs — leaf 0
/// for the chain tables, FL 0 for the outer tables, the base for the
/// passenger's.
pub struct Tower {
    pub(super) cfg: TowerConfig,
    pub(super) statement: AppStatement,
    /// The app block's offset in every outer's public segment.
    pub(super) app_base: usize,
    /// Leaf 0 — the lane's material owner (one shape for every leaf),
    /// type-erased: whichever workload's [`TowerLeaf`] impl this tower ran.
    pub(super) chain: Box<dyn TowerLeaf>,
    /// FL 0 (the root's own fresh child) — the outer tables' owner (one
    /// shape for every FL).
    pub(super) fl: FlNode,
    /// The spine's BASE, kept whenever it is not the root itself: its
    /// circuit key survives at the root — as the k = 3 root's node-slot
    /// sigma key, or as the steady passenger's — so the discharge needs
    /// its tables.
    pub(super) base: Option<NodeOut>,
    pub(super) root: NodeOut,
    /// A STEADY root (two spine folds or more) owes the single orphan its
    /// transition made; a shallower root owes none.
    pub(super) expect_passenger: bool,
    /// The leaf size this tower ran at — the wire bundle's VK-selection
    /// tag. Compressions per leaf for a chain, or whatever unit a
    /// workload's own leaf counts.
    pub(super) blocks_per_leaf: usize,
    /// A second VK-selection tag a workload's own leaf may need (e.g. a
    /// Merkle-tree depth); `None` for a chain.
    pub(super) depth: Option<usize>,
    /// The run's cost ledger.
    pub(super) costs: TowerCosts,
}

/// Where a tower run's wall clock went: the per-stage ONLINE spans (the
/// per-statement work a deployment re-pays per proof) against the per-shape
/// SETUP (circuit emission + fill-plan compilation — statement-independent,
/// cacheable across runs, paid once per distinct shape here because the
/// driver does not yet cache FL/node shapes).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TowerCosts {
    pub leaves: usize,
    pub leaf_online_ms: f64,
    pub leaf_setup_ms: f64,
    pub fls: usize,
    pub fl_online_ms: f64,
    pub fl_setup_ms: f64,
    pub nodes: usize,
    pub node_online_ms: f64,
    pub node_setup_ms: f64,
    /// Wall clock of every FL / node builder call, end to end — everything
    /// the builders do, whichever bucket their own timers put it in.
    pub fl_wall_ms: f64,
    pub node_wall_ms: f64,
}

impl TowerCosts {
    pub fn online_ms(&self) -> f64 {
        self.leaf_online_ms + self.fl_online_ms + self.node_online_ms
    }
    pub fn setup_ms(&self) -> f64 {
        self.leaf_setup_ms + self.fl_setup_ms + self.node_setup_ms
    }
}

impl Tower {
    /// PROVE THE CHAIN: `n_leaves` sequential segments of
    /// `blocks_per_leaf` compressions each, folded to one root.
    ///
    /// `n_leaves` must be even and at least 4 (leaves pair into 2-ary FLs
    /// and the tower roots at a node; the k-ary FL door stays open but is
    /// not driven here). Leaves build sequentially — each segment's walk
    /// IS the chain compute — and every input drops once folded: a
    /// segment pair drops when its FL takes it (leaf 0 stays as the
    /// lane's material owner), so what stays resident is one leaf pair,
    /// the FL row (built forward, folded backward from the tail), and the
    /// current spine node.
    pub fn prove(
        cfg: TowerConfig,
        h_start: [u32; 16],
        blocks_per_leaf: usize,
        n_leaves: usize,
    ) -> Tower {
        // h_end is a pure function of (h_start, total blocks) — no need to
        // build a single leaf to know the statement up front.
        let h_end = native_chain(&h_start, n_leaves * blocks_per_leaf);
        let statement = ChainStatement {
            h_start,
            h_end,
            n_blocks: n_leaves * blocks_per_leaf,
        }
        .to_app();
        let mut h = h_start;
        Self::prove_leaves(
            cfg,
            n_leaves,
            statement,
            blocks_per_leaf,
            None,
            TowerCosts::default(),
            |_| {
                // Each segment starts at the last one's h_end — leaves
                // build sequentially, the walk IS the chain compute.
                let cp0 = build_chain_proof(cfg, h, blocks_per_leaf);
                let cp1 = build_chain_proof(cfg, cp0.h_end, blocks_per_leaf);
                h = cp1.h_end;
                (cp0, cp1)
            },
        )
    }

    /// **FOLD `n_leaves` LEAVES, TWO AT A TIME**: the shared tail
    /// [`Tower::prove`] (the BLAKE3 chain) and a workload's own driver,
    /// living in a separate crate entirely outside this one, both drive —
    /// `next_pair(i)` builds the tower's i-th
    /// (0-indexed) leaf pair of one concrete [`TowerLeaf`] type; this pairs
    /// each into a first-level node (leaf 0 of the run survives as the
    /// lane's material owner, type-erased into the `Tower`), then
    /// [`Self::fold_first_level`] settles the base and the spine. `pub`:
    /// the extension surface's driving entry point.
    pub fn prove_leaves<L: TowerLeaf + 'static>(
        cfg: TowerConfig,
        n_leaves: usize,
        statement: AppStatement,
        per_leaf: usize,
        depth: Option<usize>,
        mut costs: TowerCosts,
        mut next_pair: impl FnMut(usize) -> (L, L),
    ) -> Tower {
        assert!(
            n_leaves >= 4 && n_leaves.is_multiple_of(2),
            "the tower pairs leaves into 2-ary FLs and roots at a node: \
             n_leaves must be even and >= 4, got {n_leaves}"
        );
        let mut owner: Option<L> = None;
        let mut fls: Vec<FlNode> = Vec::with_capacity(n_leaves / 2);
        for i in 0..n_leaves / 2 {
            let (l0, l1) = next_pair(i);
            for l in [&l0, &l1] {
                costs.leaves += 1;
                costs.leaf_online_ms += l.online().total();
                costs.leaf_setup_ms += l.online().setup_ms;
            }
            let kids: [&dyn TowerLeaf; 2] = [&l0, &l1];
            let t_fl = std::time::Instant::now();
            fls.push(build_first_level_node_kids(cfg, &kids));
            costs.fl_wall_ms += t_fl.elapsed().as_secs_f64() * 1e3;
            // Leaf 0 survives as the lane's material owner (one shape for
            // every leaf); every other leaf drops here.
            owner.get_or_insert(l0);
        }
        Self::fold_first_level(
            cfg,
            fls,
            Box::new(owner.expect("leaf 0")),
            statement,
            per_leaf,
            depth,
            costs,
        )
    }

    /// The BASE and the SPINE over a built first level — shared by both
    /// workloads; the leaf owner supplies the lane's tables and the app
    /// composition rule.
    pub(super) fn fold_first_level(
        cfg: TowerConfig,
        fls: Vec<FlNode>,
        leaf: Box<dyn TowerLeaf>,
        statement: AppStatement,
        blocks_per_leaf: usize,
        depth: Option<usize>,
        mut costs: TowerCosts,
    ) -> Tower {
        let kind = leaf.app_kind();
        for fl in &fls {
            costs.fls += 1;
            costs.fl_online_ms += fl.t.total();
            costs.fl_setup_ms += fl.t.setup_ms;
        }
        let mut note_node = |n: &NodeOut| {
            costs.nodes += 1;
            costs.node_online_ms += n.online.total();
            costs.node_setup_ms += n.online.setup_ms;
        };
        let app_base = fls[0].stmt_base;
        let claims_base = fls[0].fold_pub_base;

        let inner = leaf.inner();
        let registry = &inner.built.shape.registry;
        let blake = leaf.r1cs();
        let blake_lc = blake.csc_lincheck_circuit();
        let chain_mats = [(&blake.a_0, &blake.b_0)];
        let chain_circs: Vec<&dyn LincheckCircuit> = vec![blake_lc];
        let chain_circuit = &inner.built.shape.circuit;
        let chain_jp = inner_jagged_params(inner);

        // ---- THE BASE: fresh-only over the LAST two FLs (the spine
        // grows by prepending, so the base covers the chain's tail) ----
        let k = fls.len();
        let mut node_wall_ms = 0.0f64;
        let t_node = std::time::Instant::now();
        let base = build_node_outer_app_kind(
            cfg,
            &[&fls[k - 2].lo, &fls[k - 1].lo],
            Some(app_base),
            kind,
            Some(ChainLane {
                registry,
                mats: &chain_mats,
                circs: &chain_circs,
                circuit: chain_circuit,
                params: &chain_jp,
                priors: &[&fls[k - 2].acc, &fls[k - 1].acc],
                claims_base,
            }),
            None,
        );
        node_wall_ms += t_node.elapsed().as_secs_f64() * 1e3;
        note_node(&base);

        // ---- THE SPINE: prepend the next-earlier FL, level by level ----
        let (base, root) = if k == 2 {
            (None, base)
        } else {
            let mut cur: Option<NodeOut> = None;
            for i in (0..k - 2).rev() {
                let prev = cur.as_ref().unwrap_or(&base);
                let prev_lane = prev.lane_acc.clone().expect("the lane rides every level");
                let t_node = std::time::Instant::now();
                let n = build_node_outer_app_kind(
                    cfg,
                    &[&fls[i].lo, &prev.lo],
                    Some(app_base),
                    kind,
                    Some(ChainLane {
                        registry,
                        mats: &chain_mats,
                        circs: &chain_circs,
                        circuit: chain_circuit,
                        params: &chain_jp,
                        priors: &[&fls[i].acc, &prev_lane],
                        claims_base,
                    }),
                    Some(SpineIn {
                        node_child: 1,
                        prior: &prev.block,
                        forge: false,
                    }),
                );
                node_wall_ms += t_node.elapsed().as_secs_f64() * 1e3;
                // ONE steady shape: from the second spine fold on, the
                // digest must not move (wall 3's convergence).
                if let Some(c) = &cur {
                    assert_eq!(
                        n.lo.shape.circuit.digest(),
                        c.lo.shape.circuit.digest(),
                        "the spine converged and must stay converged"
                    );
                }
                note_node(&n);
                cur = Some(n); // the folded spine node drops here
            }
            (
                Some(base),
                cur.expect("k > 2 built at least one spine node"),
            )
        };

        costs.node_wall_ms += node_wall_ms;
        let fl = fls.into_iter().next().expect("FL 0");
        Tower {
            cfg,
            statement,
            app_base,
            chain: leaf,
            fl,
            base,
            root,
            expect_passenger: k >= 4,
            blocks_per_leaf,
            depth,
            costs,
        }
    }

    /// The run's cost ledger — online vs per-shape setup, per stage.
    pub fn costs(&self) -> &TowerCosts {
        &self.costs
    }

    pub fn config(&self) -> TowerConfig {
        self.cfg
    }

    pub fn statement(&self) -> &AppStatement {
        &self.statement
    }

    /// The composition rule this tower's nodes used.
    pub fn app_kind(&self) -> AppKind {
        self.chain.app_kind()
    }

    /// The root proof's serialized size in KiB — the recursable artifact.
    pub fn root_proof_kib(&self) -> f64 {
        census_kib(&self.root.lo.proof)
    }

    /// **THE ROOT-SIDE RESIDUE.** Everything the tower defers lands here
    /// and discharges against the native tables:
    ///
    /// 1. the STATEMENT rides the root's app block (packed `h_start` then
    ///    `h_end`);
    /// 2. the CHAIN LANE — every leaf's claims — against the chain BLAKE3
    ///    matrices, the chain circuit's sigma table, and the chain layout;
    /// 3. the MAIN FOLD against the outer registry's matrices and element
    ///    types, and against the sigma/jagged tables of every circuit a
    ///    keyed slot can name — the FL's, the root's own, and the base's;
    /// 4. the PASSENGER — the spine's single orphan — against the base's
    ///    own tables, owed exactly when the root is steady.
    pub fn discharge_root(&self) -> Result<(), RootDischargeFailure> {
        self.discharge_with(
            &self.root.lo.public,
            self.root
                .lane_acc
                .as_ref()
                .expect("the lane rides every level"),
            &self.root.acc,
            &self.root.block.passenger,
            &self.statement,
            self.expect_passenger,
        )
    }

    /// The ROOT'S TRANSPORTABLE PART: what a consumer holds beside the
    /// statement — publics, proof, commitment. Everything else the
    /// standalone verifier derives from config via [`TowerVk`]; the
    /// wire-format milestone gives this a byte form.
    ///
    /// [`TowerVk`]: crate::tower::TowerVk
    pub fn root_bundle(&self) -> RootBundle<'_> {
        RootBundle {
            public: &self.root.lo.public,
            proof: &self.root.lo.proof,
            commitment: &self.root.lo.commitment,
        }
    }

    /// The root bundle ON THE WIRE: the proof-IO tower-root flavor
    /// (statement + publics + commitment + proof, tagged with config and
    /// leaf size for VK selection). `verify_root_bytes` is the consuming
    /// end.
    ///
    /// [`verify_root_bytes`]: crate::tower::verify_root_bytes
    pub fn root_bundle_bytes(&self) -> Vec<u8> {
        TowerRootBundle {
            config: self.cfg,
            blocks_per_leaf: self.blocks_per_leaf as u64,
            depth: self.depth.map_or(0, |d| d as u64),
            statement: self.statement,
            public: self.root.lo.public.clone(),
            commitment: self.root.lo.commitment.clone(),
            proof: self.root.lo.proof.clone(),
        }
        .to_bytes()
    }

    /// The discharge legs over CALLER-SUPPLIED artifacts: the prover's
    /// self-check hands in its own objects; the standalone verifier hands
    /// in what it REASSEMBLED from a bundle's public segment. `self`
    /// supplies only the native tables (leaf 0, FL 0, base, root) — one
    /// superset covers every depth, since discharge looks tables up per
    /// digest and unused entries are inert.
    pub(super) fn discharge_with(
        &self,
        public: &[F128],
        lane: &Accumulator,
        main: &Accumulator,
        passenger: &[([F128; 2], MatrixClaim)],
        statement: &AppStatement,
        expect_passenger: bool,
    ) -> Result<(), RootDischargeFailure> {
        use RootDischargeFailure::*;
        // (1) the statement: the root's app block, word for word — no
        // workload match needed, [`AppStatement`] is kind-erased.
        if public[self.app_base..self.app_base + 8] != statement.words[..] {
            return Err(Statement);
        }
        // THE SPAN COUNTER: app word 8 is g^{n_blocks}, composed
        // multiplicatively child-to-parent, and word 9 is the counter
        // BASE g^{blocks_per_leaf} — the root end of the grounding chain
        // (each parent copy-constrains its children's base word; the
        // root's is also pinned natively by check_public). Together the
        // count is proof-bound at EVERY depth.
        if public[self.app_base + 8] != span_count_word(statement.count as u128)
            || public[self.app_base + 9] != span_count_word(self.blocks_per_leaf as u128)
        {
            return Err(Statement);
        }
        // (2) the lane vs the leaf tables (leaf 0's shape).
        let blake = self.chain.r1cs();
        let chain_mats = [(&blake.a_0, &blake.b_0)];
        let chain_circuit = &self.chain.inner().built.shape.circuit;
        if !lane.discharge(&chain_mats) {
            return Err(ChainLaneBoolean);
        }
        if !lane.per_element.is_empty() {
            return Err(ChainLaneElement);
        }
        if !lane.discharge_sigma(&[chain_circuit]) {
            return Err(ChainLaneSigma);
        }
        let chain_jp = inner_jagged_params(self.chain.inner());
        if !lane.discharge_jagged(&[(chain_circuit.digest(), &chain_jp)]) {
            return Err(ChainLaneJagged);
        }
        // (3) the main fold vs the outer tables.
        let fl_lo = &self.fl.lo;
        if !main.discharge(&leaf_boolean_mats(fl_lo)) {
            return Err(Boolean);
        }
        let el_mats: Vec<_> = fl_lo
            .shape
            .registry
            .element_types()
            .iter()
            .map(|t| {
                let e = t.element_type().expect("element table");
                (e.a_0(), e.b_0())
            })
            .collect();
        if !main.discharge_element(&el_mats) {
            return Err(Element);
        }
        let fl_jp = node_jagged_params(fl_lo);
        let root_jp = node_jagged_params(&self.root.lo);
        let base_jp = self.base.as_ref().map(|b| node_jagged_params(&b.lo));
        let mut circuits: Vec<&Circuit> = vec![&fl_lo.shape.circuit, &self.root.lo.shape.circuit];
        let mut layouts: Vec<([u8; 32], &JaggedParams)> = vec![
            (fl_lo.shape.circuit.digest(), &fl_jp),
            (self.root.lo.shape.circuit.digest(), &root_jp),
        ];
        if let (Some(b), Some(jp)) = (&self.base, &base_jp) {
            circuits.push(&b.lo.shape.circuit);
            layouts.push((b.lo.shape.circuit.digest(), jp));
        }
        if !main.discharge_sigma(&circuits) {
            return Err(Sigma);
        }
        if !main.discharge_jagged(&layouts) {
            return Err(Jagged);
        }
        // (4) the passenger: (sigma-shaped, jagged-shaped), keyed by the
        // base — the only orphan a spine ever makes.
        let live = passenger.iter().any(|(_, c)| entry_live(c));
        if live != expect_passenger {
            return Err(Passenger);
        }
        if live {
            let base = self.base.as_ref().expect("a steady root keeps its base");
            let base_d = base.lo.shape.circuit.digest();
            if passenger.len() != 2
                || passenger[0].0 != digest_f128(&base_d)
                || passenger[1].0 != digest_f128(&base_d)
                || !entry_live(&passenger[0].1)
                || !entry_live(&passenger[1].1)
            {
                return Err(Passenger);
            }
            let pacc = Accumulator {
                registry_digest: main.registry_digest,
                per_type: Vec::new(),
                per_element: Vec::new(),
                sigma: vec![(base_d, passenger[0].1.clone())],
                jagged: vec![(base_d, passenger[1].1.clone())],
            };
            let jp = base_jp.as_ref().expect("computed with the base");
            if !pacc.discharge_sigma(&[&base.lo.shape.circuit])
                || !pacc.discharge_jagged(&[(base_d, jp)])
            {
                return Err(Passenger);
            }
        }
        Ok(())
    }
}
