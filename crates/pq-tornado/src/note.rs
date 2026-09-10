//! Post-quantum Tornado Cash note scheme.
//!
//! A **note** is a pair of 256-bit secrets `(nullifier k, secret r)`. Its
//! on-tree **commitment** is `cm = SHA256_c(k ‖ r)` and its public
//! double-spend tag is the **nullifier hash** `nh = SHA256_c(k ‖ DOMAIN_NF)`,
//! where `SHA256_c(x ‖ y) = SHA-256 compression of the 512-bit block (x‖y)
//! under the fixed IV` (a fixed-length Merkle–Damgård hash — collision
//! resistant for fixed-length inputs).
//!
//! ## Why this is post-quantum
//!
//! * The commitment / nullifier / Merkle hashing is **SHA-256**, whose only
//!   quantum speedup is Grover (a quadratic loss on preimage, none on the
//!   collision resistance that soundness relies on) — unlike Tornado's original
//!   Pedersen hash, whose collision resistance is the discrete-log problem and
//!   thus fully broken by Shor.
//! * There are **no signatures and no discrete-log / pairing assumptions** in
//!   the note scheme; withdrawal authority is knowledge of the hash preimage
//!   `(k, r)`, a purely hash-based (PQ) notion.
//! * The accompanying proof system (Flock/Ligerito) is a transparent
//!   hash/code-based SNARK — no trusted setup, no quantum-broken pairings.

use crate::sha256::{SHA256_IV, compress};

/// A 256-bit value as 8 little-endian-within-word `u32`s. Word `w` bit `b`
/// lives at logical bit `32·w + b` (matching the circuit's bit layout).
pub type H256 = [u32; 8];

/// Domain-separation constant mixed into the nullifier hash so it can never
/// coincide with a commitment (`cm` uses the random `secret` in the same slot).
pub const DOMAIN_NF: H256 = [
    0x544f_524e,
    0x4144_4f5f,
    0x4e55_4c4c,
    0x4946_4945,
    0x525f_5630,
    0x0000_0001,
    0x0000_0000,
    0x0000_0000,
];

/// `SHA256_c(left ‖ right)` — one SHA-256 compression of the 512-bit block
/// formed by concatenating two 256-bit values, under the fixed IV.
pub fn hash_pair(left: &H256, right: &H256) -> H256 {
    let mut m = [0u32; 16];
    m[..8].copy_from_slice(left);
    m[8..].copy_from_slice(right);
    compress(&SHA256_IV, &m)
}

/// Commitment `cm = SHA256_c(k ‖ r)`.
pub fn commitment(nullifier: &H256, secret: &H256) -> H256 {
    hash_pair(nullifier, secret)
}

/// Nullifier hash `nh = SHA256_c(k ‖ DOMAIN_NF)`.
pub fn nullifier_hash(nullifier: &H256) -> H256 {
    hash_pair(nullifier, &DOMAIN_NF)
}

/// A note (deposit secret material). `serde` so the CLI can persist it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Note {
    pub nullifier: H256,
    pub secret: H256,
}

impl Note {
    pub fn commitment(&self) -> H256 {
        commitment(&self.nullifier, &self.secret)
    }
    pub fn nullifier_hash(&self) -> H256 {
        nullifier_hash(&self.nullifier)
    }
}

/// A fixed-depth binary Merkle tree over note commitments, hashing with
/// [`hash_pair`]. Empty leaves are the zero value; internal zero-subtree hashes
/// are precomputed.
#[derive(Clone, Debug)]
pub struct MerkleTree {
    pub depth: usize,
    /// `layers[0]` = leaves (length `2^depth`), `layers[depth]` = `[root]`.
    pub layers: Vec<Vec<H256>>,
}

impl MerkleTree {
    /// Build a tree of the given depth from the provided leaves (commitments),
    /// zero-padding to `2^depth` leaves.
    pub fn new(depth: usize, leaves: &[H256]) -> Self {
        let n = 1usize << depth;
        assert!(leaves.len() <= n, "too many leaves for depth {depth}");
        let mut layer: Vec<H256> = leaves.to_vec();
        layer.resize(n, [0u32; 8]);
        let mut layers = vec![layer];
        for _ in 0..depth {
            let prev = layers.last().unwrap();
            let next: Vec<H256> = prev
                .chunks(2)
                .map(|pair| hash_pair(&pair[0], &pair[1]))
                .collect();
            layers.push(next);
        }
        Self { depth, layers }
    }

    pub fn root(&self) -> H256 {
        self.layers[self.depth][0]
    }

    /// Authentication path for leaf `index`: `(siblings, bits)` where `bits[d]`
    /// is the path bit at level `d` (0 = our node is the left child at that
    /// level, 1 = right child).
    pub fn path(&self, index: usize) -> (Vec<H256>, Vec<bool>) {
        assert!(index < (1usize << self.depth));
        let mut siblings = Vec::with_capacity(self.depth);
        let mut bits = Vec::with_capacity(self.depth);
        let mut idx = index;
        for d in 0..self.depth {
            let sib = idx ^ 1;
            siblings.push(self.layers[d][sib]);
            bits.push(idx & 1 == 1);
            idx >>= 1;
        }
        (siblings, bits)
    }
}

/// Recompute a Merkle root from a leaf and its authentication path — the exact
/// computation the circuit enforces in zero knowledge.
pub fn recompute_root(leaf: &H256, siblings: &[H256], bits: &[bool]) -> H256 {
    let mut node = *leaf;
    for (sib, &bit) in siblings.iter().zip(bits.iter()) {
        node = if bit {
            hash_pair(sib, &node)
        } else {
            hash_pair(&node, sib)
        };
    }
    node
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_recomputes_root() {
        let leaves: Vec<H256> = (0..8u32)
            .map(|i| commitment(&[i; 8], &[i.wrapping_mul(7) + 1; 8]))
            .collect();
        let tree = MerkleTree::new(4, &leaves);
        for idx in 0..8usize {
            let (sibs, bits) = tree.path(idx);
            let got = recompute_root(&leaves[idx], &sibs, &bits);
            assert_eq!(got, tree.root(), "leaf {idx}");
        }
    }
}
