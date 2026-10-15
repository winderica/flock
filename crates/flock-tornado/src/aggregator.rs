//! **The aggregator side**: collect [`UserProof`]s, verify each on intake,
//! and fold them all into ONE recursive root with `flock-prover`'s tower.
//!
//! The aggregator never sees a private witness. Each admitted proof becomes a
//! one-withdrawal leaf with a public-only witness ([`admit_tornado_leaf`]);
//! the tower's first-level nodes read nothing else from their children.
//!
//! Intake verification is full (proof + deferred matrix and wiring
//! assertions discharged natively), so a bad proof is rejected at the door
//! instead of failing the whole tower at its root.

use std::time::Instant;

use flock_prover::tower::{Tower, TowerConfig, TowerCosts, native_digest_tree};

use crate::{
    leaf::{AdmitError, TornadoLeaf, admit_tornado_leaf},
    note::native_leaf_digest,
    tower::TornadoStatement,
    user::UserProof,
};

/// An aggregator for one `(cfg, depth)`.
pub struct Aggregator {
    cfg: TowerConfig,
    depth: usize,
    leaves: Vec<TornadoLeaf>,
    /// Milliseconds spent admitting each proof, in admission order.
    pub admit_ms: Vec<f64>,
}

impl Aggregator {
    pub fn new(cfg: TowerConfig, depth: usize) -> Self {
        Self {
            cfg,
            depth,
            leaves: Vec::new(),
            admit_ms: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.leaves.len()
    }

    pub fn is_empty(&self) -> bool {
        self.leaves.is_empty()
    }

    /// Verify a user proof and queue it for aggregation. A refused proof is
    /// not queued.
    pub fn admit(&mut self, p: UserProof) -> Result<(), AdmitError> {
        let t = Instant::now();
        let leaf = admit_tornado_leaf(
            self.cfg,
            self.depth,
            p.leaf,
            std::slice::from_ref(&p.public),
        );
        self.admit_ms.push(t.elapsed().as_secs_f64() * 1e3);
        self.leaves.push(leaf?);
        Ok(())
    }

    /// The statement the root will attest, computable from the admitted
    /// public outputs alone.
    pub fn statement(&self) -> TornadoStatement {
        let digests: Vec<[u32; 8]> = self
            .leaves
            .iter()
            .map(|l| native_leaf_digest(&l.public))
            .collect();
        TornadoStatement {
            digest: native_digest_tree(&digests),
            n_withdrawals: self.leaves.len(),
        }
    }

    /// Fold every admitted proof into one root. The tower pairs leaves into
    /// two-child first-level nodes and roots at a node, so the count must be
    /// even and at least four.
    pub fn aggregate(self) -> Tower {
        let n = self.leaves.len();
        assert!(
            n >= 4 && n.is_multiple_of(2),
            "the tower needs an even number of proofs, at least four; have {n}"
        );
        let statement = self.statement().to_app();
        let mut slots: Vec<Option<TornadoLeaf>> = self.leaves.into_iter().map(Some).collect();
        Tower::prove_leaves(
            self.cfg,
            n,
            statement,
            1,
            Some(self.depth),
            TowerCosts::default(),
            |i| {
                let l0 = slots[2 * i].take().expect("each leaf folds once");
                let l1 = slots[2 * i + 1].take().expect("each leaf folds once");
                (l0, l1)
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        leaf::AdmitError,
        testing::{sample_withdrawal, test_config},
        user::{prove_withdrawal, verify_withdrawal},
    };

    /// A user proof round-trips through bincode and verifies; a proof
    /// presented with someone else's public outputs is refused; a proof with
    /// a tampered public segment is refused.
    #[test]
    fn user_proof_verifies_and_rejects_tampering() {
        let cfg = test_config();
        let depth = 20;
        let w = sample_withdrawal(7, depth);
        let (p, _) = prove_withdrawal(cfg, depth, &w);

        let bytes = bincode::serialize(&p).expect("serialize");
        let p: UserProof = bincode::deserialize(&bytes).expect("deserialize");
        verify_withdrawal(cfg, depth, &p).expect("an honest user proof verifies");

        // Claiming another withdrawal's outputs with this proof.
        let mut stolen = p.clone();
        stolen.public = sample_withdrawal(8, depth).public();
        assert!(matches!(
            verify_withdrawal(cfg, depth, &stolen),
            Err(AdmitError::Statement)
        ));

        // A doctored public segment consistent with a doctored claim: the
        // statement check passes, the proof must not.
        let mut forged = p.clone();
        forged.public.root[0] ^= 1;
        forged.leaf.public = crate::leaf::leaf_public(std::slice::from_ref(&forged.public));
        assert!(matches!(
            verify_withdrawal(cfg, depth, &forged),
            Err(AdmitError::Proof(_))
        ));
    }

    /// Four users, one aggregator, one root that discharges and attests the
    /// digest over exactly the four users' outputs.
    #[test]
    #[ignore] // Heavier — run with `-- --ignored`.
    fn four_users_aggregate_to_one_root() {
        let cfg = test_config();
        let depth = 20;
        let mut agg = Aggregator::new(cfg, depth);
        let mut digests = Vec::new();
        for i in 0..4u32 {
            let w = sample_withdrawal(100 + i, depth);
            digests.push(native_leaf_digest(std::slice::from_ref(&w.public())));
            let (p, _) = prove_withdrawal(cfg, depth, &w);
            agg.admit(p).expect("honest proof admitted");
        }
        let want = agg.statement();
        assert_eq!(want.digest, native_digest_tree(&digests));
        let tower = agg.aggregate();
        tower.discharge_root().expect("the root discharges");
        assert_eq!(TornadoStatement::from_app(tower.statement()), want);
    }
}
