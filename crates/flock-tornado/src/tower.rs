//! Driving `flock-prover`'s recursion tower over Tornado leaves: the
//! kind-erased statement's Tornado-specific VIEW ([`TornadoStatement`]),
//! and the two entry points a chain tower's [`Tower::prove`] /
//! [`TowerVk::generate`] have no Tornado-specific counterpart for —
//! [`prove_tornado`] and [`generate_tornado_vk`] — built entirely on
//! `flock-prover`'s leaf extension surface (`flock_prover::tower`).

use flock_field::F128;
use flock_prover::tower::{
    AppStatement, Tower, TowerConfig, TowerCosts, TowerVk, native_digest_tree,
};

use crate::{
    leaf::build_tornado_leaf,
    note::{H256, digest_words, native_leaf_digest, words_digest},
    testing,
};

/// What a TORNADO tower run attests: `digest` is the root of the digest
/// tree over every withdrawal's `(nullifier_hash, root)` — each leaf's
/// SHA-256 chain, combined level by level with one BLAKE3 compression in
/// the tower's fold order ([`native_digest_tree`]) — and `n_withdrawals`
/// is the count, bound by the span counter exactly as the chain's is.
#[derive(Copy, Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TornadoStatement {
    pub digest: H256,
    pub n_withdrawals: usize,
}

impl TornadoStatement {
    /// This statement's app words, packed exactly as the root's digest FL
    /// publishes them: `[D0, D1, 0 x6]`.
    pub fn to_app(&self) -> AppStatement {
        let dw = digest_words(&self.digest);
        AppStatement {
            words: [
                dw[0],
                dw[1],
                F128::ZERO,
                F128::ZERO,
                F128::ZERO,
                F128::ZERO,
                F128::ZERO,
                F128::ZERO,
            ],
            count: self.n_withdrawals,
        }
    }
    /// The inverse of [`Self::to_app`].
    pub fn from_app(a: &AppStatement) -> Self {
        TornadoStatement {
            digest: words_digest([a.words[0], a.words[1]]),
            n_withdrawals: a.count,
        }
    }
}

/// **PROVE A TORNADO WITHDRAWAL SET**: `leaves.len()` batches of
/// `withdrawals_per_leaf` depth-`depth` withdrawals each, folded to one
/// root whose statement is the digest tree over every withdrawal's
/// public outputs plus the count.
///
/// Same geometry rule as `Tower::prove`: an even number of leaves, at
/// least four, one batch size for all.
pub fn prove_tornado(
    cfg: TowerConfig,
    depth: usize,
    leaves: &[Vec<crate::note::Withdrawal>],
) -> Tower {
    let n_leaves = leaves.len();
    let per_leaf = leaves[0].len();
    assert!(
        leaves.iter().all(|l| l.len() == per_leaf),
        "one batch size per tower"
    );
    // The leaves' digests are pure functions of the withdrawal lists (no
    // proof needed to know them), so the statement is computable up
    // front — the same trick the chain tower's h_end uses.
    let digests: Vec<H256> = leaves
        .iter()
        .map(|l| {
            let p: Vec<_> = l.iter().map(crate::note::Withdrawal::public).collect();
            native_leaf_digest(&p)
        })
        .collect();
    let statement = TornadoStatement {
        digest: native_digest_tree(&digests),
        n_withdrawals: n_leaves * per_leaf,
    }
    .to_app();
    Tower::prove_leaves(
        cfg,
        n_leaves,
        statement,
        per_leaf,
        Some(depth),
        TowerCosts::default(),
        |i| {
            let l0 = build_tornado_leaf(cfg, depth, &leaves[2 * i]);
            let l1 = build_tornado_leaf(cfg, depth, &leaves[2 * i + 1]);
            (l0, l1)
        },
    )
}

/// The Tornado VK for one `(cfg, depth, withdrawals_per_leaf)`: a
/// six-leaf reference tower over synthetic withdrawals. Shapes are
/// statement-independent, so its circuits, tables and layouts are those
/// of any production tower at the same geometry.
pub fn generate_tornado_vk(cfg: TowerConfig, depth: usize, per_leaf: usize) -> TowerVk {
    let leaves: Vec<Vec<crate::note::Withdrawal>> = (0..6u32)
        .map(|i| testing::sample_withdrawal_set(0xF00D + 100 * i, depth, per_leaf))
        .collect();
    TowerVk::from_tower(prove_tornado(cfg, depth, &leaves))
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use flock_prover::tower::{
        ChainStatement, RootDischargeFailure, TowerVerifyError, verify_root, verify_root_bytes,
    };

    use super::*;
    use crate::testing::test_config;

    /// **THE TORNADO TOWER, END TO END**: six leaves of withdrawals folded
    /// to one root (three FLs, a base, a steady spine node), the root-side
    /// residue discharged, the root verified STANDALONE from its wire bytes
    /// against a VK generated from geometry alone — and the lies a
    /// consumer could tell rejected: a wrong digest, a wrong count, and
    /// (kind-erased now: `AppStatement` carries only words + count) a
    /// same-shape statement from the WRONG workload, which fails on the
    /// same statement leg rather than a separate `VkMismatch`.
    #[test]
    #[ignore] // Heavier — run with `-- --ignored`.
    fn tornado_tower_end_to_end() {
        let cfg = test_config();
        let (depth, per_leaf, n_leaves) = (8usize, 8usize, 6usize);
        let leaves: Vec<Vec<crate::note::Withdrawal>> = (0..n_leaves as u32)
            .map(|i| testing::sample_withdrawal_set(0xA11CE + 1000 * i, depth, per_leaf))
            .collect();

        let t0 = Instant::now();
        let tower = prove_tornado(cfg, depth, &leaves);
        let prove_s = t0.elapsed().as_secs_f64();
        tower
            .discharge_root()
            .expect("the root-side residue discharges");

        // The statement is the native digest tree over the leaves' lists.
        let st = TornadoStatement::from_app(tower.statement());
        let leaf_digests: Vec<[u32; 8]> = leaves
            .iter()
            .map(|l| {
                let p: Vec<_> = l.iter().map(crate::note::Withdrawal::public).collect();
                native_leaf_digest(&p)
            })
            .collect();
        assert_eq!(st.digest, native_digest_tree(&leaf_digests));
        assert_eq!(st.n_withdrawals, n_leaves * per_leaf);

        // Standalone, over the wire, against a geometry-derived VK.
        let t1 = Instant::now();
        let vk = generate_tornado_vk(cfg, depth, per_leaf);
        let vk_s = t1.elapsed().as_secs_f64();
        let bytes = tower.root_bundle_bytes();
        let t2 = Instant::now();
        let got = verify_root_bytes(&vk, &bytes).expect("the wire bundle verifies standalone");
        let verify_s = t2.elapsed().as_secs_f64();
        assert_eq!(
            TornadoStatement::from_app(&got),
            st,
            "the statement rode the wire"
        );

        // A wrong digest dies on the statement leg.
        let mut wrong = st;
        wrong.digest[0] ^= 1;
        assert!(matches!(
            verify_root(&vk, &wrong.to_app(), &tower.root_bundle()),
            Err(TowerVerifyError::Discharge(RootDischargeFailure::Statement))
        ));
        // A wrong count (a whole extra leaf pair, so geometry passes) dies
        // on the span counter.
        let mut lied = st;
        lied.n_withdrawals += 2 * per_leaf;
        assert!(matches!(
            verify_root(&vk, &lied.to_app(), &tower.root_bundle()),
            Err(TowerVerifyError::Discharge(RootDischargeFailure::Statement))
        ));
        // A same-shape CHAIN statement's app words against this tornado
        // root: the words don't match what the root published, so it dies
        // on the statement leg — there is no separate "wrong workload"
        // check any more, and none is needed (`AppStatement` is kind-erased).
        let chain_st = ChainStatement {
            h_start: [0; 16],
            h_end: [0; 16],
            n_blocks: n_leaves * per_leaf,
        }
        .to_app();
        assert!(matches!(
            verify_root(&vk, &chain_st, &tower.root_bundle()),
            Err(TowerVerifyError::Discharge(RootDischargeFailure::Statement))
        ));

        println!(
            "\nTORNADO TOWER {cfg:?}: {n_leaves} leaves x {per_leaf} withdrawals x depth {depth} \
             = {} withdrawals ({} SHA-256 compressions)\n  \
             ONE root proof: {:.1} KiB | prove {prove_s:.1}s | bundle {:.1} KiB\n  \
             VERIFIED STANDALONE: vk generate {vk_s:.1}s (one-time) | verify {verify_s:.3}s",
            st.n_withdrawals,
            st.n_withdrawals * crate::leaf::TornadoShape::compressions_per_withdrawal(depth),
            tower.root_proof_kib(),
            bytes.len() as f64 / 1024.0,
        );
    }
}
