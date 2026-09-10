//! Prover + verifier pipeline for the post-quantum Tornado Cash withdrawal
//! circuit, over Flock's Ligerito PCS.
//!
//! The base R1CS proof (zerocheck + lincheck) certifies that the committed
//! witness satisfies the whole withdrawal circuit. On top of it we open **two
//! packed-direct PCS claims** — the revealed Merkle root and nullifier hash —
//! against the same witness commitment, exactly the mechanism Flock's
//! `merkle_path` protocol uses to expose a public root. The verifier recomputes
//! the expected opened value from the *public* root / nullifier hash, so a
//! prover cannot substitute a different committed value.
//!
//! The public withdrawal *context* (an opaque byte string — e.g. recipient ‖
//! relayer ‖ fee) is absorbed into the Fiat–Shamir transcript, making a proof
//! non-malleably bound to its intended withdrawal.

use flock_prover::{
    challenger::{Challenger, FsChallenger},
    field::F128,
    lincheck::pack_z_lincheck_from_packed,
    pcs::{
        self, Commitment, DirectEqInd, LOG_PACKING, PackedDirectClaim, PackedDirectClaimRef,
        PcsParams, ligerito::LigeritoProfile, ring_switch::build_eq_sparse,
    },
    proof::ZClaim,
    prover::{ProveCore, prove_fast_core_with_codeword},
};
use serde::{Deserialize, Serialize};

use crate::{
    circuit::{PublicInputs, TornadoCircuit, Witness},
    note::H256,
};

const LOG_INV_RATE: usize = 1;
// Must equal `initial_k` of Flock's embedded Ligerito security configs
// (`configs/ligerito/m*_fast.toml`), which are registered for m ∈ [22, 35].
const LOG_BATCH_SIZE: usize = 6;
const DOMAIN: &[u8] = b"pq-tornado-withdraw-v1";

/// A complete withdrawal proof (self-contained: carries its own witness
/// commitment). Serializable for the CLI.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WithdrawProof {
    pub commitment: Commitment,
    pub zerocheck: flock_prover::zerocheck::ZerocheckProof,
    pub lincheck: flock_prover::lincheck::LincheckProof,
    pub pcs_open: pcs::BatchOpeningProofLigerito,
}

#[derive(Debug)]
pub enum VerifyError {
    /// Base R1CS (zerocheck/lincheck + [ab,c] opening) replay failed.
    R1cs(flock_prover::verifier::VerifyError),
    /// The batched PCS opening (including the public-output claims) failed.
    Pcs(pcs::VerifyError),
}

fn pcs_params(m: usize) -> PcsParams {
    PcsParams {
        m,
        log_inv_rate: LOG_INV_RATE,
        log_batch_size: LOG_BATCH_SIZE,
        profile: LigeritoProfile::Fast,
    }
}

fn new_challenger(context: &[u8]) -> FsChallenger {
    let mut ch = FsChallenger::new(DOMAIN);
    ch.observe_label(b"pq-tornado-context");
    ch.observe_bytes(context);
    ch
}

/// `eq(τ,0)·w0 + eq(τ,1)·w1 = (1+τ)·w0 + τ·w1` — the 1-coord fold of a 256-bit
/// (2-word) region to a single field element.
fn fold2(tau: F128, w0: F128, w1: F128) -> F128 {
    (F128::ONE + tau) * w0 + tau * w1
}

/// The two 128-bit words of a 256-bit value, in the witness's bit layout
/// (word `w` bit `b` at logical bit `32·w + b`; F128 `lo` = bits 0..64).
fn h256_to_f128s(v: &H256) -> (F128, F128) {
    let w = |a: u32, b: u32| (a as u64) | ((b as u64) << 32);
    (
        F128 {
            lo: w(v[0], v[1]),
            hi: w(v[2], v[3]),
        },
        F128 {
            lo: w(v[4], v[5]),
            hi: w(v[6], v[7]),
        },
    )
}

/// Build the packed-direct claim point for a 256-bit region whose first
/// (128-bit-aligned) wire is `wire_base`, living in block 0, folded by `tau`.
fn region_point(m: usize, k_log: usize, wire_base: usize, tau: F128) -> Vec<F128> {
    let l = m - LOG_PACKING; // total packed-MLE coords
    let word_in_block = wire_base / 128; // even (256-bit aligned)
    debug_assert_eq!(word_in_block % 2, 0);
    let word_bits = k_log - LOG_PACKING; // coords indexing the word within a block
    let q = word_in_block >> 1;
    let mut point = vec![F128::ZERO; l];
    point[0] = tau; // fold coordinate over {word, word+1}
    for j in 1..word_bits {
        if (q >> (j - 1)) & 1 == 1 {
            point[j] = F128::ONE;
        }
    }
    // Remaining coords (block index) stay ZERO → block 0. Every zero coord
    // halves the live support of `build_eq_sparse`.
    point
}

/// Value of a packed-direct region claim as read from the packed witness.
fn region_value_witness(zp: &[F128], wire_base: usize, tau: F128) -> F128 {
    let word = wire_base / 128;
    fold2(tau, zp[word], zp[word + 1])
}

/// Inline of flock's crate-private `quirky_x_outer_full`.
fn x_outer_full(point: &flock_prover::lincheck::QuirkyPoint) -> Vec<F128> {
    let mut v = point.x_inner_rest.clone();
    v.extend_from_slice(&point.x_outer);
    v
}

/// Produce a withdrawal proof for `witness` against `circuit`, bound to the
/// public `context`. Returns the proof and the public inputs it attests to.
pub fn prove(
    circuit: &TornadoCircuit,
    witness: &Witness,
    context: &[u8],
) -> (WithdrawProof, PublicInputs) {
    let r1cs = &circuit.r1cs;
    let pi = circuit.public_inputs(witness);
    let m = r1cs.m;
    let params = pcs_params(m);

    let trace = std::env::var_os("PQT_TRACE").is_some();
    let mut t = std::time::Instant::now();
    let mut tick = |label: &str| {
        if trace {
            eprintln!(
                "  [pipeline::prove] {label:<28} {:7.2} ms",
                t.elapsed().as_secs_f64() * 1e3
            );
        }
        t = std::time::Instant::now();
    };

    // Witness: one block tiled across the 2^N_BLOCKS_LOG outer copies (Flock's
    // lincheck needs n_outer ≥ 8). Page-faulting the PCS codeword is overlapped
    // with witness generation.
    let (prefaulted, z_packed) = pcs::prefault_codeword_during(&params, || {
        let block = circuit.generate_block_witness(witness);
        let mut z_full = Vec::with_capacity(block.len() * r1cs.n_outer());
        for _ in 0..r1cs.n_outer() {
            z_full.extend_from_slice(&block);
        }
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

    let mut ch = new_challenger(context);
    let core: ProveCore = prove_fast_core_with_codeword(
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

    // Public-output fold coordinate (transcript-driven).
    let tau = ch.sample_f128();

    let pd_claims: Vec<PackedDirectClaim> = [circuit.root_hout_base(), circuit.nf_hout_base()]
        .map(|base| {
            let point = region_point(m, r1cs.k_log, base, tau);
            PackedDirectClaim {
                eq_ind: DirectEqInd::Sparse(build_eq_sparse(&point)),
                value: region_value_witness(&core.z_packed, base, tau),
                point,
            }
        })
        .into();

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

    let pre_ab: Option<&[F128]> = s_hat_v_ab.as_deref();
    let pre_c: Option<&[F128]> = Some(s_hat_v_c.as_slice());
    let pcs_open = pcs::open_batch_mixed_ligerito_with_precomputed_s_hat_v(
        z_packed,
        &prover_data,
        &commitment,
        &[ab_x.as_slice(), c_x.as_slice()],
        &[pre_ab, pre_c],
        &pd_claims,
        &padding,
        &lig_config,
        &mut ch,
    );
    tick("pcs open");

    (
        WithdrawProof {
            commitment,
            zerocheck: zc_proof,
            lincheck: lc_proof,
            pcs_open,
        },
        pi,
    )
}

/// Verify a withdrawal proof against its public inputs and context.
pub fn verify(
    circuit: &TornadoCircuit,
    proof: &WithdrawProof,
    public: &PublicInputs,
    context: &[u8],
) -> Result<(), VerifyError> {
    let r1cs = &circuit.r1cs;
    let m = r1cs.m;
    let params = pcs_params(m);
    let lincheck_circuit = r1cs.csc_lincheck_circuit();

    let mut ch = new_challenger(context);
    let (ab, c) = flock_prover::verifier::verify_core(
        r1cs,
        &proof.zerocheck,
        &proof.lincheck,
        &proof.commitment,
        lincheck_circuit,
        &mut ch,
    )
    .map_err(VerifyError::R1cs)?;

    let tau = ch.sample_f128();

    let pd_points: [(Vec<F128>, F128); 2] = [
        (circuit.root_hout_base(), public.root),
        (circuit.nf_hout_base(), public.nullifier_hash),
    ]
    .map(|(base, v)| {
        let (w0, w1) = h256_to_f128s(&v);
        (region_point(m, r1cs.k_log, base, tau), fold2(tau, w0, w1))
    });
    let pd_refs: Vec<PackedDirectClaimRef<'_>> = pd_points
        .iter()
        .map(|(point, value)| PackedDirectClaimRef {
            point,
            value: *value,
        })
        .collect();

    let lig_v_config =
        pcs::ligerito::verifier_config_for(m - LOG_PACKING, LOG_BATCH_SIZE, params.profile)
            .expect("ligerito verifier config");

    let ab_x = x_outer_full(&ab.point);
    let c_x = x_outer_full(&c.point);
    let claims: [ZClaim; 2] = [ab.clone(), c.clone()];
    let values = [claims[0].value, claims[1].value];
    let z_skips = [claims[0].point.z_skip, claims[1].point.z_skip];

    pcs::verify_opening_batch_ligerito_mixed(
        &proof.commitment,
        &values,
        &z_skips,
        &[ab_x.as_slice(), c_x.as_slice()],
        &pd_refs,
        &proof.pcs_open,
        &lig_v_config,
        &mut ch,
    )
    .map_err(VerifyError::Pcs)?;

    Ok(())
}
