//! **The Tornado leaf**: the withdrawal-batch circuit that plugs into
//! `flock-prover`'s recursion tower as a [`TowerLeaf`], in place of the
//! BLAKE3 hash chain `chain.rs` implements inside that crate.
//!
//! Every hash is one row of the fused [`crate::step`] table — SHA-256 with
//! the level swap in the same block — so the leaf is ONE boolean type,
//! which is what the tower's first-level node fold expects of a leaf.
//! `IV`, `DOMAIN_NF` and the constant-zero bit enter as fixed public inputs
//! (part of the circuit digest, not prover-chosen); the `k` fan-out, the
//! Merkle chain and every other copy constraint are the circuit's own
//! wiring argument.
//!
//! ## The statement: a digest, not a list
//!
//! The tower's application block is a FIXED 8-word tail every node
//! publishes, so a leaf cannot publish its `N` `(nh, root)` pairs up the
//! tree. Instead the leaf hashes them in-circuit — a Merkle–Damgård chain
//! of `N` more step rows, `d_{i+1} = SHA256_c(d_i, nh_i ‖ root_i)` from
//! `d_0 = IV` — and publishes only `D = d_N`. The first-level node then
//! combines its two children's digests with ONE BLAKE3 compression (its
//! FS-chain slot, [`AppKind::Digest`]), and so does every internal node:
//! the root certifies a tree of digests over every withdrawal's public
//! outputs, plus the count via the span counter. A consumer holding the
//! withdrawal list recomputes the tree natively ([`crate::note::native_leaf_digest`] /
//! [`flock_prover::tower::combine_digests`]).
//!
//! Leaf public segment (declaration order): `[iv0, iv1, dom0, dom1, zero, D0, D1]`.

use std::{
    any::Any,
    sync::{Arc, Mutex, OnceLock},
    time::Instant,
};

use flock_core::{
    circuit::builder::{
        BuiltCircuit, CircuitShape, CircuitWitness, GateType, ShapeBuilder, SlotId, SlotWitness,
        Wire,
    },
    lincheck::LincheckCircuit,
    pcs::{Commitment, PcsParams},
    r1cs::BlockR1cs,
    schedule::TableType,
    union::UnionInstance,
    verifier::{
        FlockVerifyError, verify_ligerito_union_circuit, verify_ligerito_union_circuit_deferred,
    },
};
use flock_field::F128;
use flock_merkle::HashKind;
use flock_prover::{
    prover::{UnionSlotProverInput, prove_fast_ligerito_union_circuit},
    tower::{
        AppKind, DOMAIN, FsChallenger, MixedInner, MixedProof, Online, TowerConfig, TowerLeaf,
        pcs_batch_for, steady_reps,
    },
};

use crate::{
    note::{
        DOMAIN_NF, H256, PublicInputs, SHA256_IV, Withdrawal, digest_words, native_leaf_digest,
    },
    step::{self, StepRow},
};

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

/// One fused step row. Inputs in schema order: `H_in` (2), the bit word,
/// `node` (2), `sib` (2). Outputs: `H_out` (2).
pub(crate) struct StepGate {
    pub(crate) nu: usize,
}

impl GateType for StepGate {
    type Row = StepRow;
    type Hint = ();

    fn table(&self) -> TableType {
        TableType::from_block_r1cs(&step::build_block_r1cs(self.nu))
            .with_io_schema(step::io_schema())
    }

    fn eval(&self, inputs: &[F128], _hint: &(), outputs: &mut Vec<F128>) -> Self::Row {
        let row = StepRow {
            h_in: crate::note::words_digest([inputs[0], inputs[1]]),
            bit: inputs[2].lo & 1 == 1,
            node: crate::note::words_digest([inputs[3], inputs[4]]),
            sib: crate::note::words_digest([inputs[5], inputs[6]]),
        };
        outputs.extend_from_slice(&digest_words(&row.output()));
        row
    }

    fn witness(&self, _rows: &[Self::Row], _nu: usize) -> SlotWitness {
        SlotWitness::DeferredToRows
    }
}

// ---------------------------------------------------------------------------
// The shape
// ---------------------------------------------------------------------------

/// Where the leaf publishes its digest.
pub(crate) const LEAF_PUB_DIGEST: usize = 5;
/// The leaf's public segment length.
pub(crate) const LEAF_PUB_LEN: usize = 7;

/// A built Tornado leaf shape: statement-independent, emitted once per
/// `(depth, n_withdrawals)` and cached.
#[derive(Clone)]
pub struct TornadoShape {
    pub(crate) shape: CircuitShape,
    pub(crate) step: SlotId,
    pub depth: usize,
    pub n_withdrawals: usize,
    pub nu: usize,
}

/// The fewest step rows a leaf may have. Flock ships Ligerito security
/// configs only for `m ≥ 22`, and a leaf's committed size is set by its
/// LIVE row count (`UnionInstance::dense_m`): each row is `2^15` bits, so
/// `m = 22` needs more than `2^(22-7-8) = 64` rows. A one-withdrawal leaf at
/// depth 20 is 23 rows, so small batches are topped up with padding rows.
pub const MIN_LEAF_ROWS: usize = 65;

impl TornadoShape {
    /// Step rows per withdrawal: commitment, nullifier, `D` levels, and the
    /// digest-chain link.
    pub fn rows_per_withdrawal(depth: usize) -> usize {
        depth + 3
    }

    /// Padding rows a batch of `n` needs to reach [`MIN_LEAF_ROWS`].
    pub fn padding_rows(depth: usize, n: usize) -> usize {
        MIN_LEAF_ROWS.saturating_sub(n * Self::rows_per_withdrawal(depth))
    }

    /// Every step row the shape emits, padding included.
    pub fn total_rows(depth: usize, n: usize) -> usize {
        n * Self::rows_per_withdrawal(depth) + Self::padding_rows(depth, n)
    }

    /// SHA-256 compressions of application work per withdrawal (the digest
    /// link excluded).
    pub fn compressions_per_withdrawal(depth: usize) -> usize {
        depth + 2
    }
}

pub(crate) fn tornado_shape_cached(depth: usize, n: usize) -> Arc<TornadoShape> {
    type Cache = Mutex<Vec<((usize, usize), Arc<TornadoShape>)>>;
    static CACHE: OnceLock<Cache> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
    let mut g = cache.lock().unwrap();
    if let Some((_, s)) = g.iter().find(|(k, _)| *k == (depth, n)) {
        return s.clone();
    }
    let s = Arc::new(build_tornado_shape(depth, n));
    g.push(((depth, n), s.clone()));
    s
}

/// Emit the circuit for `n` independent depth-`depth` withdrawals.
pub fn build_tornado_shape(depth: usize, n: usize) -> TornadoShape {
    assert!(depth >= 1 && n >= 1);
    let n_rows = TornadoShape::total_rows(depth, n);
    let nu = n_rows.next_power_of_two().trailing_zeros() as usize;

    let mut sb = ShapeBuilder::new(nu);
    let step = sb.slot(StepGate { nu });

    // Fixed publics: IV pair, DOMAIN pair, the constant-zero bit.
    let iv = digest_words(&SHA256_IV);
    let iv_w = [sb.fixed_public_input(iv[0]), sb.fixed_public_input(iv[1])];
    let dom = digest_words(&DOMAIN_NF);
    let dom_w = [sb.fixed_public_input(dom[0]), sb.fixed_public_input(dom[1])];
    let zero = sb.fixed_public_input(F128::ZERO);

    // Every withdrawal's (nh, root) output wires, for the digest chain.
    let mut nh_root: Vec<[Wire; 4]> = Vec::with_capacity(n);
    for _ in 0..n {
        let island = sb.begin_island();
        let k = [sb.input(), sb.input()];
        let r = [sb.input(), sb.input()];
        // cm = SHA256_c(IV, k ‖ r)
        let cm = sb.gate(step, &[iv_w[0], iv_w[1], zero, k[0], k[1], r[0], r[1]]);
        // nh = SHA256_c(IV, k ‖ DOMAIN_NF) — the same k wires.
        let nh = sb.gate(
            step,
            &[iv_w[0], iv_w[1], zero, k[0], k[1], dom_w[0], dom_w[1]],
        );
        // The authentication path.
        let mut node = cm;
        for _ in 0..depth {
            let bit = sb.input();
            let sib = [sb.input(), sb.input()];
            node = sb.gate(
                step,
                &[iv_w[0], iv_w[1], bit, node[0], node[1], sib[0], sib[1]],
            );
        }
        sb.end_island(island);
        nh_root.push([nh[0], nh[1], node[0], node[1]]);
    }

    // The digest chain: d_0 = IV, d_{i+1} = SHA256_c(d_i, nh_i ‖ root_i).
    let mut d = [iv_w[0], iv_w[1]];
    for w in &nh_root {
        d = {
            let o = sb.gate(step, &[d[0], d[1], zero, w[0], w[1], w[2], w[3]]);
            [o[0], o[1]]
        };
    }
    sb.publish(d[0]);
    sb.publish(d[1]);

    // Padding to MIN_LEAF_ROWS: a chain of fixed-input compressions
    // (`SHA256_c(IV, 0 ‖ 0)`, then each feeding the next). Every input is a
    // fixed public or a gate output, so the rows carry no prover choice and
    // touch nothing the statement reads.
    let mut pad = [zero, zero];
    for _ in 0..TornadoShape::padding_rows(depth, n) {
        let o = sb.gate(step, &[iv_w[0], iv_w[1], zero, pad[0], pad[1], zero, zero]);
        pad = [o[0], o[1]];
    }

    let shape = sb.finish().expect("the tornado leaf builds");
    assert_eq!(shape.counts.len(), 1, "one boolean type");
    TornadoShape {
        shape,
        step,
        depth,
        n_withdrawals: n,
        nu,
    }
}

/// The statement inputs in declaration order, for [`CircuitShape::run`]: the
/// fixed publics (IV pair, DOMAIN pair, zero), then per withdrawal `k`, `r`,
/// and per level the bit word and the sibling pair.
pub(crate) fn tornado_vals(ws: &[Withdrawal]) -> Vec<F128> {
    let iv = digest_words(&SHA256_IV);
    let dom = digest_words(&DOMAIN_NF);
    let mut v = vec![iv[0], iv[1], dom[0], dom[1], F128::ZERO];
    for w in ws {
        v.extend_from_slice(&digest_words(&w.note.nullifier));
        v.extend_from_slice(&digest_words(&w.note.secret));
        for (sib, &b) in w.siblings.iter().zip(w.bits.iter()) {
            v.push(F128::new(u64::from(b), 0));
            v.extend_from_slice(&digest_words(sib));
        }
    }
    v
}

// ---------------------------------------------------------------------------
// The leaf proof
// ---------------------------------------------------------------------------

/// A proved Tornado leaf: the same [`MixedInner`] a chain leaf carries, so
/// the first-level node consumes it through the same walker.
pub struct TornadoLeaf {
    pub(crate) inner: MixedInner,
    pub(crate) shape: Arc<TornadoShape>,
    /// The withdrawals' public outputs, in leaf order.
    pub public: Vec<PublicInputs>,
    /// `D` — the digest published in the leaf's app words.
    pub digest: H256,
    /// What the leaf cost, split SETUP vs ONLINE — see [`TowerLeaf::online`].
    pub(crate) t: Online,
    #[allow(dead_code)]
    pub(crate) onlines: Vec<Online>,
}

impl TornadoLeaf {
    pub fn n_withdrawals(&self) -> usize {
        self.shape.n_withdrawals
    }
    pub fn depth(&self) -> usize {
        self.shape.depth
    }
}

impl TowerLeaf for TornadoLeaf {
    fn inner(&self) -> &MixedInner {
        &self.inner
    }
    fn r1cs(&self) -> Arc<BlockR1cs> {
        tornado_step_r1cs(self.inner.nu)
    }
    fn count(&self) -> usize {
        self.n_withdrawals()
    }
    fn app_kind(&self) -> AppKind {
        AppKind::Digest
    }
    fn app_words(&self) -> [F128; 8] {
        let dw = digest_words(&self.digest);
        [
            dw[0],
            dw[1],
            F128::ZERO,
            F128::ZERO,
            F128::ZERO,
            F128::ZERO,
            F128::ZERO,
            F128::ZERO,
        ]
    }
    fn app_pub_range(&self) -> (usize, usize) {
        (LEAF_PUB_DIGEST, 2)
    }
    fn online(&self) -> Online {
        self.t
    }
}

/// The step-table block R1CS per `nu`, cached process-wide — the leaf's own,
/// the FL's fold matrices, the lane's discharge tables all read one block
/// (and its lazily built CSC lincheck circuit) through here.
pub(crate) fn tornado_step_r1cs(nu: usize) -> Arc<BlockR1cs> {
    type Cache = Mutex<Vec<(usize, Arc<BlockR1cs>)>>;
    static CACHE: OnceLock<Cache> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
    let mut g = cache.lock().unwrap();
    if let Some((_, r)) = g.iter().find(|(k, _)| *k == nu) {
        return r.clone();
    }
    let r = Arc::new(step::build_block_r1cs(nu));
    g.push((nu, r.clone()));
    r
}

/// What a leaf PROVER hands over: the proof and the public segment it
/// proves — and nothing else. The private trace (notes, siblings, path bits,
/// every intermediate hash) never leaves the prover. Serializable: this is
/// the wire object a user sends an aggregator.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct LeafProof {
    pub public: Vec<F128>,
    pub proof: MixedProof,
    pub commitment: Commitment,
}

/// Why a [`LeafProof`] was refused.
#[derive(Debug)]
pub enum AdmitError {
    /// The public segment is not the claimed withdrawals' statement.
    Statement,
    /// The commitment's PCS parameters are not this shape's — a prover may
    /// not pick its own security parameters.
    Params,
    /// The proof itself refused (transcript replay, openings, wiring).
    Proof(FlockVerifyError),
    /// The deferred constraint-matrix claims refused against the step table.
    Matrix,
    /// The deferred copy-constraint (sigma) claims refused.
    Sigma,
}

/// The leaf profile's PCS parameters for a shape — derived, never taken from
/// a prover.
fn leaf_pcs_params(cfg: TowerConfig, ts: &TornadoShape) -> PcsParams {
    let union = UnionInstance::new(&ts.shape.registry, ts.shape.counts.clone());
    assert!(!union.has_element(), "a tornado leaf is boolean-only");
    let pf = cfg.leaf_profile();
    let batch = pcs_batch_for(&union, pf);
    PcsParams {
        m: union.dense_m(),
        log_inv_rate: 1,
        profile: pf,
        log_batch_size: batch,
        num_lanes: union.commit_lanes(batch),
        merkle_hash: HashKind::Blake3,
    }
}

/// The public segment a batch's statement determines:
/// `[iv0, iv1, dom0, dom1, zero, D0, D1]`.
pub fn leaf_public(public: &[PublicInputs]) -> Vec<F128> {
    let iv = digest_words(&SHA256_IV);
    let dom = digest_words(&DOMAIN_NF);
    let d = digest_words(&native_leaf_digest(public));
    let v = vec![iv[0], iv[1], dom[0], dom[1], F128::ZERO, d[0], d[1]];
    debug_assert_eq!(v.len(), LEAF_PUB_LEN);
    debug_assert_eq!(v[LEAF_PUB_DIGEST], d[0]);
    v
}

/// **PROVE** a batch of withdrawals as one union-circuit leaf — the prover's
/// whole job, and nothing a verifier needs to do. Returns the wire object and
/// the online cost record of the last of `1 + steady_reps()` iterations.
pub fn prove_tornado_leaf(
    cfg: TowerConfig,
    depth: usize,
    ws: &[Withdrawal],
) -> (LeafProof, Online, Vec<Online>) {
    let n = ws.len();
    let t_shape = Instant::now();
    let ts = tornado_shape_cached(depth, n);
    let shape_ms = t_shape.elapsed().as_secs_f64() * 1e3;
    let nu = ts.nu;
    let t_setup = Instant::now();
    let r1cs = tornado_step_r1cs(nu);
    let lc = r1cs.csc_lincheck_circuit();
    let pcs_params = leaf_pcs_params(cfg, &ts);
    let setup_ms = shape_ms + t_setup.elapsed().as_secs_f64() * 1e3;

    let vals = tornado_vals(ws);
    let no_hints: Vec<&(dyn Any + Sync)> = Vec::new();

    let reps = 1 + steady_reps();
    let mut onlines: Vec<Online> = Vec::with_capacity(reps);
    let mut fin = None;
    for _ in 0..reps {
        let t0 = Instant::now();
        let witness = ts.shape.run(&vals, &no_hints);
        let walk_ms = t0.elapsed().as_secs_f64() * 1e3;
        let union = UnionInstance::new(&ts.shape.registry, ts.shape.counts.clone());
        let t1 = Instant::now();
        let wit = step::generate_witness_batch_major_partial(witness.rows::<StepGate>(ts.step), nu);
        let witgen_ms = t1.elapsed().as_secs_f64() * 1e3;
        let t2 = Instant::now();
        let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
        let (p, c, _) = prove_fast_ligerito_union_circuit(
            &union,
            &ts.shape.circuit,
            &witness.public,
            &pcs_params,
            vec![UnionSlotProverInput::new(wit, lc)],
            Vec::new(),
            &mut ch,
        );
        let prove_ms = t2.elapsed().as_secs_f64() * 1e3;
        let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
        onlines.push(Online {
            setup_ms,
            walk_ms,
            witgen_ms,
            prove_ms,
            wall_ms,
            ..Online::default()
        });
        fin = Some(LeafProof {
            public: witness.public,
            proof: MixedProof::Rs(p),
            commitment: c,
        });
    }
    let lp = fin.expect("one online iteration");
    (lp, *onlines.last().expect("one online iteration"), onlines)
}

/// **ADMIT** a leaf proof — the verifier side. Checks that the public segment
/// IS the claimed withdrawals' statement and that the PCS parameters are the
/// shape's, runs the deferred verify (which re-derives the matrix and wiring
/// assertions the recursion will fold — never taken from the prover), then
/// discharges those assertions natively so a bad proof is refused HERE
/// rather than poisoning a whole tower at its root.
///
/// The returned leaf carries a public-only witness: it is exactly what an
/// aggregator that never saw the private trace can hold, and it is all a
/// first-level node reads from its children.
pub fn admit_tornado_leaf(
    cfg: TowerConfig,
    depth: usize,
    lp: LeafProof,
    claimed: &[PublicInputs],
) -> Result<TornadoLeaf, AdmitError> {
    let n = claimed.len();
    let ts = tornado_shape_cached(depth, n);
    if lp.public != leaf_public(claimed) {
        return Err(AdmitError::Statement);
    }
    let pcs = leaf_pcs_params(cfg, &ts);
    if lp.commitment.params != pcs {
        return Err(AdmitError::Params);
    }
    let r1cs = tornado_step_r1cs(ts.nu);
    let lcs: Vec<&dyn LincheckCircuit> = vec![r1cs.csc_lincheck_circuit()];
    let built = BuiltCircuit {
        shape: ts.shape.clone(),
        witness: CircuitWitness::public_only(lp.public),
    };
    let union = UnionInstance::new(&built.shape.registry, built.shape.counts.clone());
    let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
    let p = match &lp.proof {
        MixedProof::Rs(p) => p,
        MixedProof::Ag(_) => {
            return Err(AdmitError::Proof(FlockVerifyError::ClassMismatch));
        }
    };
    let (_claims, work, sigma) = verify_ligerito_union_circuit_deferred(
        &union,
        &built.shape.circuit,
        &built.witness.public,
        &lcs,
        &lp.commitment,
        p,
        &pcs,
        &mut ch,
    )
    .map_err(AdmitError::Proof)?;
    let boolean = work.boolean.as_ref().ok_or(AdmitError::Matrix)?;
    if boolean.check(&union, &lcs).is_err() || work.element.is_some() {
        return Err(AdmitError::Matrix);
    }
    if !sigma.check(&built.shape.circuit) {
        return Err(AdmitError::Sigma);
    }
    let digest = native_leaf_digest(claimed);
    Ok(TornadoLeaf {
        inner: MixedInner {
            nu: ts.nu,
            built,
            proof: lp.proof,
            commitment: lp.commitment,
            pcs,
            work,
            sigma,
        },
        shape: ts,
        public: claimed.to_vec(),
        digest,
        t: Online::default(),
        onlines: Vec::new(),
    })
}

/// Prove a batch of withdrawals as ONE union-circuit leaf under the tower's
/// leaf profile, and admit it — the single-party path ([`prove_tornado_leaf`]
/// then [`admit_tornado_leaf`]), with the leaf's cost record set to the
/// PROVE, not the self-check.
pub fn build_tornado_leaf(cfg: TowerConfig, depth: usize, ws: &[Withdrawal]) -> TornadoLeaf {
    let (lp, t, onlines) = prove_tornado_leaf(cfg, depth, ws);
    let public: Vec<PublicInputs> = ws.iter().map(Withdrawal::public).collect();
    let mut leaf =
        admit_tornado_leaf(cfg, depth, lp, &public).expect("an honest tornado leaf is admitted");
    leaf.t = t;
    leaf.onlines = onlines;
    leaf
}

/// Verify a leaf natively against the claimed withdrawal list.
pub fn verify_tornado_leaf(
    leaf: &TornadoLeaf,
    public: &[PublicInputs],
) -> Result<(), FlockVerifyError> {
    let inner = &leaf.inner;
    let union = UnionInstance::new(
        &inner.built.shape.registry,
        inner.built.shape.counts.clone(),
    );
    let r1cs = tornado_step_r1cs(inner.nu);
    let lcs: Vec<&dyn LincheckCircuit> = vec![r1cs.csc_lincheck_circuit()];

    // The public segment IS the statement.
    if leaf_public(public) != inner.built.witness.public {
        return Err(FlockVerifyError::CircuitMismatch);
    }

    let mut ch = FsChallenger::with_chained_blake3(DOMAIN);
    match &inner.proof {
        MixedProof::Rs(p) => verify_ligerito_union_circuit(
            &union,
            &inner.built.shape.circuit,
            &inner.built.witness.public,
            &lcs,
            &inner.commitment,
            p,
            &inner.pcs,
            &mut ch,
        )
        .map(|_| ()),
        MixedProof::Ag(_) => unreachable!("the tornado leaf proves RS"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{sample_withdrawal, test_config};

    /// The circuit's published digest equals the natively recomputed one for
    /// every batch shape, and every step row's H_out is the muxed hash.
    #[test]
    fn tornado_circuit_publishes_the_native_digest() {
        let depth = 4;
        for n in [1usize, 3] {
            let ts = build_tornado_shape(depth, n);
            let ws: Vec<Withdrawal> = (0..n)
                .map(|i| sample_withdrawal(i as u32 + 1, depth))
                .collect();
            let witness = ts.shape.run(&tornado_vals(&ws), &[]);
            let public: Vec<PublicInputs> = ws.iter().map(Withdrawal::public).collect();
            let d = digest_words(&native_leaf_digest(&public));
            assert_eq!(witness.public.len(), LEAF_PUB_LEN);
            assert_eq!(&witness.public[LEAF_PUB_DIGEST..], &d, "n={n}: digest");
            let rows = witness.rows::<StepGate>(ts.step);
            assert_eq!(rows.len(), TornadoShape::total_rows(depth, n));
            // The withdrawals' rows come first (D+2 each), the digest chain
            // last; row 0 of each withdrawal is the commitment.
            let per = TornadoShape::compressions_per_withdrawal(depth);
            for (i, w) in ws.iter().enumerate() {
                let r = &rows[i * per];
                assert_eq!(r.h_in, SHA256_IV);
                assert!(!r.bit);
                assert_eq!(r.output(), w.note.commitment(), "w{i} cm");
            }
        }
    }

    /// Prove/verify round-trip; a tampered list is rejected.
    #[test]
    fn tornado_leaf_proves_and_verifies() {
        let cfg = test_config();
        let depth = 8;
        let n = 8;
        let ws: Vec<Withdrawal> = (0..n)
            .map(|i| sample_withdrawal(i as u32 + 1, depth))
            .collect();
        let leaf = build_tornado_leaf(cfg, depth, &ws);
        let public: Vec<PublicInputs> = ws.iter().map(Withdrawal::public).collect();
        verify_tornado_leaf(&leaf, &public).expect("honest tornado leaf verifies");

        let mut bad = public.clone();
        bad[3].root[0] ^= 1;
        assert!(verify_tornado_leaf(&leaf, &bad).is_err(), "tampered root");
        let mut bad = public.clone();
        bad.swap(0, 5);
        assert!(
            verify_tornado_leaf(&leaf, &bad).is_err(),
            "permuted members"
        );
    }

    /// **The first-level node over two TORNADO leaves**: both verified
    /// deferred in one outer, their step-table claims folded 2→1, and the
    /// app block carrying `BLAKE3(D_0 ‖ D_1)` with the count.
    #[test]
    #[ignore] // Heavier — run with `-- --ignored`.
    fn tornado_first_level_node() {
        use flock_prover::tower::{
            TowerLeaf, build_first_level_node_kids, census_kib, combine_digests,
        };
        let cfg = test_config();
        let (depth, n) = (8usize, 8usize);
        let mk = |seed: u32| -> Vec<Withdrawal> {
            (0..n)
                .map(|i| sample_withdrawal(seed + i as u32, depth))
                .collect()
        };
        let (w0, w1) = (mk(1), mk(1000));
        let l0 = build_tornado_leaf(cfg, depth, &w0);
        let l1 = build_tornado_leaf(cfg, depth, &w1);
        let kids: [&dyn TowerLeaf; 2] = [&l0, &l1];
        let fl = build_first_level_node_kids(cfg, &kids);
        let want = combine_digests(&l0.digest, &l1.digest);
        assert_eq!(fl.digest(), want);
        let dw = digest_words(&want);
        assert_eq!(fl.lo.public[fl.stmt_base], dw[0]);
        assert_eq!(fl.lo.public[fl.stmt_base + 1], dw[1]);
        println!(
            "\nTORNADO FIRST-LEVEL NODE: 2 x {n} withdrawals x depth {depth}\n  \
             leaf: nu {} | prove {:.0} ms each\n  \
             FL: dense_m {} | publics {} | prove {:.0} ms | verify {:.0} ms | proof {:.1} KiB",
            l0.inner.nu,
            l0.t.prove_ms,
            fl.lo.pcs.m,
            fl.lo.public.len(),
            fl.t.prove_ms,
            fl.t.verify_ms,
            census_kib(&fl.lo.proof),
        );
    }
}
