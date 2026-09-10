//! Post-quantum Tornado Cash withdrawal prover / verifier over the **step
//! circuit** — the decomposition Flock is actually designed for.
//!
//! Same statement as [`crate::pipeline`], different decomposition. Instead of
//! compiling the whole withdrawal (commitment, nullifier hash, and `D` Merkle
//! levels) into one giant R1CS block, this proves `2^n_log` copies of the
//! *small* one-compression [step circuit](crate::step) and links them with the
//! [slot-wiring argument](crate::wiring).
//!
//! In the language of the Flock paper (§1, §4.6), the step circuit is the
//! repeated relation `F` and [`crate::wiring`] is an instantiation of the
//! **glue circuit `G`** — discharged, as the paper prescribes, by a small
//! auxiliary protocol whose only output is one extra `ẑ` evaluation claim
//! folded into the PCS opening, *not* by extra R1CS rows.
//!
//! Consequences:
//!
//! * The per-block matrices are `2^15 × 2^15` regardless of tree depth, so
//!   circuit build time and memory no longer grow with `D`.
//! * Lincheck walks `A_0` once, and `A_0` shrinks from `D + 2` compressions of
//!   nonzeros to exactly one — a 34× reduction at depth 32.
//! * `m` drops (24 → 22 at depth 32), so a quarter as much data is committed.
//!
//! Instance layout (`n_comp = D + 2` instances; any leftover instances are
//! filled with the valid all-zero-input compression and carry no constraints):
//!
//! ```text
//!   0        cm = SHA256_c(k ‖ r)
//!   1        nh = SHA256_c(k ‖ DOMAIN_NF)
//!   2+d      node_{d+1} = SHA256_c(mux(node_d, sib_d, b_d))   d = 0..D−1
//! ```
//!
//! and the public wiring constraints (see [`crate::wiring`]):
//!
//! ```text
//!   node(0) ⊕ node(1)          = 0          same nullifier k feeds both
//!   t(0)                        = 0          cm's mux is the identity
//!   t(1)                        = 0          nh's mux is the identity
//!   sib(1)                      = DOMAIN_NF  domain separation
//!   node(2) ⊕ H_out(0)         = 0          node_0 = cm
//!   node(2+d) ⊕ H_out(1+d)     = 0          Merkle chaining
//!   H_out(1+D)                  = root       public output
//!   H_out(1)                    = nh         public output
//! ```
//!
//! Pinning `t = 0` on the two note hashes is what stops the prover from
//! swapping the mux and spending `SHA256_c(k ‖ r)` a second time under the
//! nullifier `r`.

use flock_prover::{
    challenger::{Challenger, FsChallenger},
    field::F128,
    lincheck::{build_eq_table, pack_z_lincheck_from_packed},
    pcs::{
        self, Commitment, DirectEqInd, LOG_PACKING, PackedDirectClaim, PackedDirectClaimRef,
        PcsParams, ligerito::LigeritoProfile,
    },
    proof::ZClaim,
    prover::ProveCore,
};
use serde::{Deserialize, Serialize};

use crate::{
    circuit::{PublicInputs, Witness},
    note::DOMAIN_NF,
    step::{SLOT_HOUT, SLOT_NODE, SLOT_SIB, SLOT_T, StepCircuit, StepInput},
    wiring::{self, Constraint, WiringProof},
};

const LOG_INV_RATE: usize = 1;
const LOG_BATCH_SIZE: usize = 6;
const DOMAIN: &[u8] = b"pq-tornado-withdraw-stepwise-v1";

/// Smallest `m` with an embedded Ligerito security config.
const MIN_M: usize = 22;

/// A withdrawal proof: one witness commitment, one R1CS proof over the batched
/// step circuit, one wiring sumcheck, one batched PCS opening.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StepwiseProof {
    pub commitment: Commitment,
    pub zerocheck: flock_prover::zerocheck::ZerocheckProof,
    pub lincheck: flock_prover::lincheck::LincheckProof,
    pub wiring: WiringProof,
    pub pcs_open: pcs::BatchOpeningProofLigerito,
}

#[derive(Debug)]
pub enum VerifyError {
    R1cs(flock_prover::verifier::VerifyError),
    Wiring(wiring::WiringError),
    Pcs(pcs::VerifyError),
}

/// The withdrawal circuit: a step-circuit R1CS sized to hold one depth-`depth`
/// withdrawal.
pub struct StepwiseCircuit {
    pub depth: usize,
    pub step: StepCircuit,
}

impl StepwiseCircuit {
    /// SHA-256 compressions per withdrawal: commitment, nullifier, `D` levels.
    pub fn n_compressions(depth: usize) -> usize {
        depth + 2
    }

    /// Build the circuit for a depth-`depth` tree, choosing the smallest legal
    /// instance count. `n_log` is floored at `MIN_M − K_LOG` because Flock only
    /// ships Ligerito security configs for `m ∈ [22, 35]`.
    pub fn build(depth: usize) -> Self {
        assert!(depth >= 1);
        let n_log = usize::max(
            Self::n_compressions(depth)
                .next_power_of_two()
                .trailing_zeros() as usize,
            MIN_M - crate::step::K_LOG,
        );
        assert!(
            crate::step::K_LOG + n_log <= 35,
            "tree too deep for Flock's embedded Ligerito configs (m > 35)"
        );
        let step = StepCircuit::build(n_log);
        flock_prover::scratch::prewarm_prover(step.r1cs.m);
        Self { depth, step }
    }

    pub fn m(&self) -> usize {
        self.step.r1cs.m
    }

    /// Per-instance private inputs. Every hash has the same shape
    /// `H_out = SHA256_c(IV, mux(node, sib, bit))`.
    pub fn step_inputs(&self, w: &Witness) -> Vec<StepInput> {
        assert_eq!(w.siblings.len(), self.depth);
        assert_eq!(w.bits.len(), self.depth);
        let mut out = Vec::with_capacity(Self::n_compressions(self.depth));
        // 0: cm = SHA256_c(k ‖ r)
        out.push(StepInput {
            node: w.note.nullifier,
            sib: w.note.secret,
            bit: false,
        });
        // 1: nh = SHA256_c(k ‖ DOMAIN_NF)
        out.push(StepInput {
            node: w.note.nullifier,
            sib: DOMAIN_NF,
            bit: false,
        });
        // 2..: the Merkle path
        let mut node = w.note.commitment();
        for d in 0..self.depth {
            let inp = StepInput {
                node,
                sib: w.siblings[d],
                bit: w.bits[d],
            };
            node = inp.output();
            out.push(inp);
        }
        out
    }

    /// The public inputs a witness attests to.
    pub fn public_inputs(&self, w: &Witness) -> PublicInputs {
        PublicInputs {
            root: crate::note::recompute_root(&w.note.commitment(), &w.siblings, &w.bits),
            nullifier_hash: w.note.nullifier_hash(),
        }
    }

    /// The `D + 6` public wiring constraints for the claimed public inputs.
    pub fn constraints(&self, public: &PublicInputs) -> Vec<Constraint> {
        let slot = |i: usize, s: usize| StepCircuit::slot_index(i, s);
        let (cm, nf) = (0usize, 1usize);
        let lvl = |d: usize| 2 + d;

        let mut cons = Vec::with_capacity(self.depth + 6);
        // The nullifier `k` drives both note hashes.
        cons.push(Constraint::zero(vec![
            slot(cm, SLOT_NODE),
            slot(nf, SLOT_NODE),
        ]));
        // Both note hashes must use the identity mux (`left = node`).
        cons.push(Constraint::zero(vec![slot(cm, SLOT_T)]));
        cons.push(Constraint::zero(vec![slot(nf, SLOT_T)]));
        // Domain separation for the nullifier hash.
        cons.push(Constraint::eq_const(slot(nf, SLOT_SIB), DOMAIN_NF));
        // Merkle chaining: node_0 = cm, node_{d+1} = H_out(level d).
        cons.push(Constraint::zero(vec![
            slot(lvl(0), SLOT_NODE),
            slot(cm, SLOT_HOUT),
        ]));
        for d in 1..self.depth {
            cons.push(Constraint::zero(vec![
                slot(lvl(d), SLOT_NODE),
                slot(lvl(d - 1), SLOT_HOUT),
            ]));
        }
        // Public outputs.
        cons.push(Constraint::eq_const(
            slot(lvl(self.depth - 1), SLOT_HOUT),
            public.root,
        ));
        cons.push(Constraint::eq_const(
            slot(nf, SLOT_HOUT),
            public.nullifier_hash,
        ));
        cons
    }
}

fn pcs_params(m: usize) -> PcsParams {
    PcsParams {
        m,
        log_inv_rate: LOG_INV_RATE,
        log_batch_size: LOG_BATCH_SIZE,
        profile: LigeritoProfile::Fast,
    }
}

/// Bind the statement: circuit shape, withdrawal context, and the public
/// inputs. `BlockR1cs::statement_digest` already covers `m` / `k_log`, but the
/// tree depth determines the *wiring constraint list*, so it is bound
/// explicitly rather than left implicit in the verifier's arguments.
fn new_challenger(depth: usize, context: &[u8], public: &PublicInputs) -> FsChallenger {
    let mut ch = FsChallenger::new(DOMAIN);
    ch.observe_label(b"pq-tornado-shape");
    ch.observe_bytes(&(depth as u64).to_le_bytes());
    ch.observe_label(b"pq-tornado-context");
    ch.observe_bytes(context);
    ch.observe_label(b"pq-tornado-public");
    let mut buf = Vec::with_capacity(64);
    for w in public.root.iter().chain(public.nullifier_hash.iter()) {
        buf.extend_from_slice(&w.to_le_bytes());
    }
    ch.observe_bytes(&buf);
    ch
}

fn x_outer_full(point: &flock_prover::lincheck::QuirkyPoint) -> Vec<F128> {
    let mut v = point.x_inner_rest.clone();
    v.extend_from_slice(&point.x_outer);
    v
}

/// Produce a withdrawal proof, bound to the public `context`. Returns the proof
/// and the public inputs it attests to.
pub fn prove(
    circuit: &StepwiseCircuit,
    witness: &Witness,
    context: &[u8],
) -> (StepwiseProof, PublicInputs) {
    let public = circuit.public_inputs(witness);
    let inputs = circuit.step_inputs(witness);
    let proof = prove_raw(circuit, &inputs, &public, context);
    (proof, public)
}

/// Prove directly from per-instance step inputs and a chosen public statement.
///
/// [`prove`] is this with `inputs` / `public` derived honestly from a witness.
/// Exposed so soundness tests can drive the prover with adversarial instance
/// assignments (a spliced Merkle link, a swapped mux, …) and check that the
/// wiring layer rejects them.
pub fn prove_raw(
    circuit: &StepwiseCircuit,
    inputs: &[StepInput],
    public: &PublicInputs,
    context: &[u8],
) -> StepwiseProof {
    let constraints = circuit.constraints(public);
    let r1cs = &circuit.step.r1cs;
    let m = r1cs.m;
    let params = pcs_params(m);

    let trace = std::env::var_os("PQT_TRACE").is_some();
    let mut t = std::time::Instant::now();
    let mut tick = |label: &str| {
        if trace {
            eprintln!(
                "  [stepwise::prove] {label:<28} {:7.2} ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
        t = std::time::Instant::now();
    };

    let (prefaulted, z_packed) = pcs::prefault_codeword_during(&params, || {
        let z_full = circuit.step.generate_witness(inputs);
        pcs::pack_witness(&z_full, m)
    });
    tick("witness gen + pack");

    let lig_config =
        pcs::ligerito::prover_config_for(m - LOG_PACKING, LOG_BATCH_SIZE, params.profile)
            .expect("ligerito prover config");

    let a_packed = r1cs.apply_a_packed(&z_packed);
    let b_packed = r1cs.apply_b_packed(&z_packed);
    let z_lincheck = pack_z_lincheck_from_packed(&z_packed, m, r1cs.k_log);
    let lincheck_circuit = r1cs.csc_lincheck_circuit();
    tick("A·z, B·z, lincheck pack");

    let mut ch = new_challenger(circuit.depth, context, public);
    let core: ProveCore = flock_prover::prover::prove_fast_core_with_codeword(
        r1cs,
        &params,
        z_packed,
        a_packed,
        b_packed,
        z_lincheck,
        lincheck_circuit,
        prefaulted,
        &mut ch,
    );
    tick("commit+zerocheck+lincheck");

    // The glue: one sumcheck over the 256-bit slot folds.
    let (wiring_proof, wiring_claim) = wiring::prove(&core.z_packed, m, &constraints, &mut ch);
    let wiring_pd = PackedDirectClaim {
        eq_ind: DirectEqInd::Dense(build_eq_table(&wiring_claim.point)),
        point: wiring_claim.point,
        value: wiring_claim.value,
    };
    tick("wiring sumcheck");

    let padding = r1cs.padding_spec();
    let ab_x = x_outer_full(&core.ab.point);
    let c_x = x_outer_full(&core.c.point);

    let ProveCore {
        zc_proof,
        lc_proof,
        commitment,
        prover_data,
        z_packed,
        s_hat_v_ab,
        s_hat_v_c,
        ..
    } = core;

    let pcs_open = pcs::open_batch_mixed_ligerito_with_precomputed_s_hat_v(
        z_packed,
        &prover_data,
        &commitment,
        &[ab_x.as_slice(), c_x.as_slice()],
        &[s_hat_v_ab.as_deref(), Some(s_hat_v_c.as_slice())],
        &[wiring_pd],
        &padding,
        &lig_config,
        &mut ch,
    );
    tick("pcs open");

    StepwiseProof {
        commitment,
        zerocheck: zc_proof,
        lincheck: lc_proof,
        wiring: wiring_proof,
        pcs_open,
    }
}

/// Verify a withdrawal proof against the claimed public inputs and context.
pub fn verify(
    circuit: &StepwiseCircuit,
    proof: &StepwiseProof,
    public: &PublicInputs,
    context: &[u8],
) -> Result<(), VerifyError> {
    let r1cs = &circuit.step.r1cs;
    let m = r1cs.m;
    let params = pcs_params(m);
    let constraints = circuit.constraints(public);
    let lincheck_circuit = r1cs.csc_lincheck_circuit();

    let mut ch = new_challenger(circuit.depth, context, public);
    let (ab, c) = flock_prover::verifier::verify_core(
        r1cs,
        &proof.zerocheck,
        &proof.lincheck,
        &proof.commitment,
        lincheck_circuit,
        &mut ch,
    )
    .map_err(VerifyError::R1cs)?;

    let wiring_claim =
        wiring::verify(&proof.wiring, m, &constraints, &mut ch).map_err(VerifyError::Wiring)?;

    let lig_v_config =
        pcs::ligerito::verifier_config_for(m - LOG_PACKING, LOG_BATCH_SIZE, params.profile)
            .expect("ligerito verifier config");

    let ab_x = x_outer_full(&ab.point);
    let c_x = x_outer_full(&c.point);
    let claims: [ZClaim; 2] = [ab, c];
    let values = [claims[0].value, claims[1].value];
    let z_skips = [claims[0].point.z_skip, claims[1].point.z_skip];

    pcs::verify_opening_batch_ligerito_mixed(
        &proof.commitment,
        &values,
        &z_skips,
        &[ab_x.as_slice(), c_x.as_slice()],
        &[PackedDirectClaimRef {
            point: &wiring_claim.point,
            value: wiring_claim.value,
        }],
        &proof.pcs_open,
        &lig_v_config,
        &mut ch,
    )
    .map_err(VerifyError::Pcs)?;

    Ok(())
}
