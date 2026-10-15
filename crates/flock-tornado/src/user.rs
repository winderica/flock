//! **The user side**: one withdrawal, one proof, produced on the user's own
//! machine. The private witness — the note, the authentication path — never
//! leaves it; what goes to an aggregator is a [`UserProof`]: a leaf proof
//! over exactly one withdrawal plus the withdrawal's public outputs.
//!
//! A one-withdrawal circuit is smaller than the smallest shipped security
//! configuration, so its shape is padded to [`crate::leaf::MIN_LEAF_ROWS`]
//! rows; the padding is fixed-input compressions and changes nothing the
//! statement reads.

use flock_prover::tower::{Online, TowerConfig};

use crate::{
    leaf::{AdmitError, LeafProof, admit_tornado_leaf, prove_tornado_leaf},
    note::{PublicInputs, Withdrawal},
};

/// What a user sends an aggregator: the proof of one withdrawal and the
/// withdrawal's public outputs (`root`, `nullifier_hash`) — the same two
/// values a Tornado withdrawal reveals anyway.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct UserProof {
    pub public: PublicInputs,
    pub leaf: LeafProof,
}

/// Prove one withdrawal. Returns the wire object and the prover's cost
/// record (`Online::total()` is the prove wall clock).
pub fn prove_withdrawal(cfg: TowerConfig, depth: usize, w: &Withdrawal) -> (UserProof, Online) {
    let (leaf, t, _) = prove_tornado_leaf(cfg, depth, std::slice::from_ref(w));
    (
        UserProof {
            public: w.public(),
            leaf,
        },
        t,
    )
}

/// Verify a user proof natively — what a user can run to check their own
/// proof before sending it, and exactly what an aggregator runs on intake.
pub fn verify_withdrawal(cfg: TowerConfig, depth: usize, p: &UserProof) -> Result<(), AdmitError> {
    admit_tornado_leaf(cfg, depth, p.leaf.clone(), std::slice::from_ref(&p.public)).map(|_| ())
}
