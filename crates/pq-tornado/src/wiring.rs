//! **Slot-wiring argument** — public copy-constraints between 256-bit-aligned
//! regions of the committed witness, proved by one small sumcheck that reduces
//! to a *single* extra PCS opening.
//!
//! This is what replaces in-circuit wires when the same small circuit is
//! instantiated many times: instead of paying `O(#instances)` R1CS rows to
//! carry a value from one hash to the next, the R1CS proves only the
//! *per-instance* relation and this layer proves the *cross-instance* relations
//! against the same commitment.
//!
//! ## Statement
//!
//! View the `2^m`-bit committed witness as `2^ν` consecutive **slots** of 256
//! bits each (`ν = m − 8`); each slot is exactly two 128-bit packed words. A
//! constraint is a public set of slot indices plus a public 256-bit right-hand
//! side:
//!
//! ```text
//!   ⊕_{s ∈ S_c} slot(s) = rhs_c            (c = 0 .. C−1)
//! ```
//!
//! That single form expresses everything the batched Tornado statement needs:
//!
//! * `{node(i), H_out(j)}, rhs = 0` — chain hash `j`'s output into hash `i`;
//! * `{node(i), node(j)}, rhs = 0` — fan-out (both hashes read the same `k`);
//! * `{t(i)}, rhs = 0` — pin instance `i`'s mux to the identity;
//! * `{sib(i)}, rhs = DOMAIN_NF` — pin a slot to a public constant;
//! * `{H_out(i)}, rhs = root` — reveal a slot as a public output.
//!
//! ## Protocol
//!
//! Sample `τ` and fold each slot to one field element,
//! `G(x) = (1+τ)·ẑ_p(2x) + τ·ẑ_p(2x+1)` — i.e. `G = ẑ_p(τ, ·)`, the packed MLE
//! with its lowest coordinate bound to `τ`. Two distinct 256-bit values agree
//! under this fold with probability `2^-128`.
//!
//! Sample `γ` and set `W(x) = Σ_c γ^c·[x ∈ S_c]` and `T = Σ_c γ^c·fold_τ(rhs_c)`.
//! All constraints hold iff `Σ_x W(x)·G(x) = T` (up to `(C+1)/2^128`). That is a
//! degree-2 product sumcheck over `ν` variables whose output claim
//! `G(r) = ẑ_p(τ, r) = v` is discharged as a
//! [`PackedDirectClaim`](flock_prover::pcs::PackedDirectClaim) in the *existing*
//! batched opening — so the whole linking layer costs `ν` round messages
//! (`32·ν` bytes) and one extra γ-combined basis term.
//!
//! `W` is public and sparse (`Σ_c |S_c|` taps), so the verifier evaluates its
//! MLE at the sumcheck's random point in `O(ν·Σ_c |S_c|)` field ops without ever
//! materializing a table.
//!
//! Only `v` — one evaluation of `ẑ_p` at a transcript-random point — is
//! revealed, so the individual linked values (the private Merkle nodes, the
//! nullifier `k`) stay hidden, exactly as in Flock's own chain/merkle-path
//! shift arguments.

use flock_prover::{challenger::Challenger, field::F128};
use serde::{Deserialize, Serialize};

use crate::note::H256;

/// Slot width in bits. A slot is two packed 128-bit words.
pub const SLOT_BITS: usize = 256;
/// `log2(SLOT_BITS)`.
pub const LOG_SLOT_BITS: usize = 8;

/// One public copy-constraint: `⊕_{s ∈ slots} slot(s) = rhs`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Constraint {
    /// Global slot indices (`instance · slots_per_block + local_slot`).
    pub slots: Vec<usize>,
    /// Public 256-bit right-hand side (all-zero for a pure copy).
    pub rhs: H256,
}

impl Constraint {
    /// `⊕ slots = 0`.
    pub fn zero(slots: Vec<usize>) -> Self {
        Self {
            slots,
            rhs: [0u32; 8],
        }
    }
    /// `slot = value` — pin a slot to a public constant / reveal it.
    pub fn eq_const(slot: usize, value: H256) -> Self {
        Self {
            slots: vec![slot],
            rhs: value,
        }
    }
}

/// The wiring sumcheck proof: `ν` degree-2 round messages `(q(1), q(∞))` plus
/// the folded value. `q(0)` is recovered from the running claim.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WiringProof {
    pub rounds: Vec<(F128, F128)>,
    pub value: F128,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WiringError {
    MalformedProof,
    SumcheckFinal,
}

/// The single packed-MLE claim the wiring argument reduces to.
#[derive(Clone, Debug)]
pub struct WiringClaim {
    /// Evaluation point of `ẑ_p`, length `m − 7`: `[τ, r_0, .., r_{ν−1}]`.
    pub point: Vec<F128>,
    pub value: F128,
}

/// The two 128-bit packed words of a 256-bit value (word `w` bit `b` lives at
/// logical bit `32w + b`; `F128::lo` covers bits 0..64).
fn h256_words(v: &H256) -> (F128, F128) {
    let j = |a: u32, b: u32| (a as u64) | ((b as u64) << 32);
    (
        F128 {
            lo: j(v[0], v[1]),
            hi: j(v[2], v[3]),
        },
        F128 {
            lo: j(v[4], v[5]),
            hi: j(v[6], v[7]),
        },
    )
}

#[inline]
fn fold_slot(tau: F128, w0: F128, w1: F128) -> F128 {
    (F128::ONE + tau) * w0 + tau * w1
}

/// `eq(point, x)` for an integer `x` (LSB-first coordinates).
fn eq_at_index(point: &[F128], x: usize) -> F128 {
    let mut acc = F128::ONE;
    for (j, &p) in point.iter().enumerate() {
        acc *= if (x >> j) & 1 == 1 { p } else { F128::ONE + p };
    }
    acc
}

/// Sample `(τ, γ)` — shared by prover and verifier, in the same transcript
/// order.
fn sample_challenges<Ch: Challenger>(ch: &mut Ch) -> (F128, F128) {
    ch.observe_label(b"pq-tornado-wiring-v1");
    let tau = ch.sample_f128();
    let gamma = ch.sample_f128();
    (tau, gamma)
}

/// Prove the wiring constraints against the packed witness `zp` (length
/// `2^{m−7}`).
pub fn prove<Ch: Challenger>(
    zp: &[F128],
    m: usize,
    constraints: &[Constraint],
    ch: &mut Ch,
) -> (WiringProof, WiringClaim) {
    let nu = m - LOG_SLOT_BITS;
    let n_slots = 1usize << nu;
    assert_eq!(zp.len(), 1usize << (m - 7), "packed witness length");

    let (tau, gamma) = sample_challenges(ch);

    // G(x) = ẑ_p(τ, x): fold each 256-bit slot to one field element.
    let one_plus_tau = F128::ONE + tau;
    let mut g: Vec<F128> = {
        use rayon::prelude::*;
        zp.par_chunks_exact(2)
            .map(|w| one_plus_tau * w[0] + tau * w[1])
            .collect()
    };
    debug_assert_eq!(g.len(), n_slots);

    // W(x) = Σ_c γ^c·[x ∈ S_c] — sparse, but materialized for the sumcheck.
    let mut wt = vec![F128::ZERO; n_slots];
    let mut gp = F128::ONE;
    for c in constraints {
        for &s in &c.slots {
            assert!(s < n_slots, "constraint slot {s} out of range");
            wt[s] += gp;
        }
        gp *= gamma;
    }

    // Degree-2 product sumcheck, binding the highest remaining variable first.
    // Parallel while the tables are large; the tail rounds are too small for
    // rayon to pay for itself.
    const PAR_MIN: usize = 1 << 12;
    let mut rounds = Vec::with_capacity(nu);
    let mut r_pts = Vec::with_capacity(nu);
    for _ in 0..nu {
        let half = g.len() / 2;
        let (wlo_s, whi_s) = wt.split_at(half);
        let (glo_s, ghi_s) = g.split_at(half);

        // q(1) = Σ W_hi·g_hi ; leading coeff = Σ ΔW·Δg
        let (e1, einf) = if half >= PAR_MIN {
            use rayon::prelude::*;
            (0..half)
                .into_par_iter()
                .map(|i| {
                    let (wl, wh) = (wlo_s[i], whi_s[i]);
                    let (gl, gh) = (glo_s[i], ghi_s[i]);
                    (wh * gh, (wh + wl) * (gh + gl))
                })
                .reduce(
                    || (F128::ZERO, F128::ZERO),
                    |(a0, a1), (b0, b1)| (a0 + b0, a1 + b1),
                )
        } else {
            let mut e1 = F128::ZERO;
            let mut einf = F128::ZERO;
            for i in 0..half {
                let (wl, wh) = (wlo_s[i], whi_s[i]);
                let (gl, gh) = (glo_s[i], ghi_s[i]);
                e1 += wh * gh;
                einf += (wh + wl) * (gh + gl);
            }
            (e1, einf)
        };

        ch.observe_f128(e1);
        ch.observe_f128(einf);
        let r = ch.sample_f128();

        if half >= PAR_MIN {
            use rayon::prelude::*;
            let (wlo_m, whi_m) = wt.split_at_mut(half);
            let (glo_m, ghi_m) = g.split_at_mut(half);
            wlo_m
                .par_iter_mut()
                .zip(whi_m.par_iter())
                .for_each(|(lo, &hi)| *lo = *lo + r * (hi + *lo));
            glo_m
                .par_iter_mut()
                .zip(ghi_m.par_iter())
                .for_each(|(lo, &hi)| *lo = *lo + r * (hi + *lo));
        } else {
            for i in 0..half {
                wt[i] = wt[i] + r * (wt[i + half] + wt[i]);
                g[i] = g[i] + r * (g[i + half] + g[i]);
            }
        }
        wt.truncate(half);
        g.truncate(half);
        rounds.push((e1, einf));
        r_pts.push(r);
    }

    let value = g[0];
    let mut point = Vec::with_capacity(nu + 1);
    point.push(tau);
    point.extend(final_point(&r_pts));
    (WiringProof { rounds, value }, WiringClaim { point, value })
}

/// Round challenges → LSB-first evaluation point (round `k` bound variable
/// `ν−1−k`).
fn final_point(r_pts: &[F128]) -> Vec<F128> {
    let nu = r_pts.len();
    let mut full = vec![F128::ZERO; nu];
    for (k, &r) in r_pts.iter().enumerate() {
        full[nu - 1 - k] = r;
    }
    full
}

/// Verify the wiring proof and return the packed-MLE claim the PCS must open.
pub fn verify<Ch: Challenger>(
    proof: &WiringProof,
    m: usize,
    constraints: &[Constraint],
    ch: &mut Ch,
) -> Result<WiringClaim, WiringError> {
    let nu = m - LOG_SLOT_BITS;
    if proof.rounds.len() != nu {
        return Err(WiringError::MalformedProof);
    }
    let n_slots = 1usize << nu;
    if constraints
        .iter()
        .any(|c| c.slots.iter().any(|&s| s >= n_slots))
    {
        return Err(WiringError::MalformedProof);
    }

    let (tau, gamma) = sample_challenges(ch);

    // Initial claim T = Σ_c γ^c·fold_τ(rhs_c).
    let mut claim = F128::ZERO;
    let mut gp = F128::ONE;
    for c in constraints {
        let (w0, w1) = h256_words(&c.rhs);
        claim += gp * fold_slot(tau, w0, w1);
        gp *= gamma;
    }

    // Replay the sumcheck.
    let mut r_pts = Vec::with_capacity(nu);
    for &(e1, einf) in &proof.rounds {
        ch.observe_f128(e1);
        ch.observe_f128(einf);
        let r = ch.sample_f128();
        // q(X) = einf·X² + c1·X + q(0), q(0) = claim + q(1) in char 2.
        let e0 = claim + e1;
        let c1 = e0 + e1 + einf;
        claim = einf * r * r + c1 * r + e0;
        r_pts.push(r);
    }
    let full = final_point(&r_pts);

    // W(full) from the sparse tap list.
    let mut w_final = F128::ZERO;
    let mut gp = F128::ONE;
    for c in constraints {
        for &s in &c.slots {
            w_final += gp * eq_at_index(&full, s);
        }
        gp *= gamma;
    }

    if claim != w_final * proof.value {
        return Err(WiringError::SumcheckFinal);
    }

    let mut point = Vec::with_capacity(nu + 1);
    point.push(tau);
    point.extend(full);
    Ok(WiringClaim {
        point,
        value: proof.value,
    })
}

#[cfg(test)]
mod tests {
    use flock_prover::{challenger::FsChallenger, pcs};

    use super::*;

    fn packed_from_bits(bits: &[bool], m: usize) -> Vec<F128> {
        pcs::pack_witness(bits, m)
    }

    fn write_slot(bits: &mut [bool], slot: usize, v: &H256) {
        for i in 0..256 {
            bits[slot * 256 + i] = (v[i / 32] >> (i % 32)) & 1 == 1;
        }
    }

    fn run(m: usize, bits: &[bool], cons: &[Constraint]) -> Result<(), WiringError> {
        let zp = packed_from_bits(bits, m);
        let mut ch = FsChallenger::new(b"wiring-test");
        let (proof, claim_p) = prove(&zp, m, cons, &mut ch);
        let mut ch = FsChallenger::new(b"wiring-test");
        let claim_v = verify(&proof, m, cons, &mut ch)?;
        assert_eq!(claim_p.point, claim_v.point);
        assert_eq!(claim_p.value, claim_v.value);
        Ok(())
    }

    #[test]
    fn accepts_satisfied_constraints_and_rejects_violations() {
        let m = 12; // 4096 bits = 16 slots
        let a: H256 = [1, 2, 3, 4, 5, 6, 7, 8];
        let konst: H256 = [0xdead, 0xbeef, 9, 9, 9, 9, 9, 9];
        let mut bits = vec![false; 1 << m];
        write_slot(&mut bits, 2, &a);
        write_slot(&mut bits, 5, &a); // copy of slot 2
        write_slot(&mut bits, 7, &konst);
        // slot 9 stays zero

        let cons = vec![
            Constraint::zero(vec![2, 5]),
            Constraint::eq_const(7, konst),
            Constraint::zero(vec![9]),
        ];
        run(m, &bits, &cons).expect("honest wiring must verify");

        // Break the copy.
        let mut bad = bits.clone();
        bad[5 * 256 + 3] ^= true;
        assert!(run(m, &bad, &cons).is_err());

        // Break the constant.
        let mut bad = bits.clone();
        bad[7 * 256 + 100] ^= true;
        assert!(run(m, &bad, &cons).is_err());

        // Break the zero pin.
        let mut bad = bits.clone();
        bad[9 * 256 + 255] ^= true;
        assert!(run(m, &bad, &cons).is_err());
    }

    #[test]
    fn claim_point_matches_packed_mle_evaluation() {
        // The sumcheck output must be exactly ẑ_p at the returned point.
        let m = 11;
        let mut bits = vec![false; 1 << m];
        for (i, b) in bits.iter_mut().enumerate() {
            *b = (i * 2654435761) % 7 < 3;
        }
        let zp = packed_from_bits(&bits, m);
        let cons = vec![Constraint::zero(vec![0, 1])];
        let mut ch = FsChallenger::new(b"wiring-eval");
        let (_proof, claim) = prove(&zp, m, &cons, &mut ch);

        // Direct MLE evaluation of the packed witness at `claim.point`.
        let mut acc = F128::ZERO;
        for (i, &w) in zp.iter().enumerate() {
            acc += eq_at_index(&claim.point, i) * w;
        }
        assert_eq!(acc, claim.value);
    }
}
