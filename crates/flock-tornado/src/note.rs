//! The note scheme: deposit notes, withdrawals, and the SHA-256 hashes that
//! bind them — the leaf-level (native) statement a withdrawal attests to,
//! independent of the circuit that proves it (see [`crate::leaf`]).
//!
//! One *withdrawal* proves, over a private note `(nullifier k, secret r)`, a
//! private authentication path, and a public `(root, nullifier_hash)`:
//!
//! 1. `cm = SHA256_c(IV, k ‖ r)` — the commitment (the tree leaf).
//! 2. `nh = SHA256_c(IV, k ‖ DOMAIN_NF)` — the double-spend tag. It reuses the
//!    same `k`, so `nh` is bound to the spent note.
//! 3. `node_0 = cm`; per level a private bit selects the left/right ordering
//!    of `(node_d, sib_d)` and `node_{d+1} = SHA256_c(IV, left ‖ right)`.

use flock_field::F128;
use flock_prover::r1cs_hashes::{merkle_r1cs::SLOT_WORDS, sha2};

// ---------------------------------------------------------------------------
// Word packing (word `w` bit `b` at logical bit 32w + b — the committed layout)
// ---------------------------------------------------------------------------

pub(crate) fn pack4(w: [u32; 4]) -> F128 {
    F128::new(
        w[0] as u64 | ((w[1] as u64) << 32),
        w[2] as u64 | ((w[3] as u64) << 32),
    )
}

pub(crate) fn unpack4(v: F128) -> [u32; 4] {
    [
        v.lo as u32,
        (v.lo >> 32) as u32,
        v.hi as u32,
        (v.hi >> 32) as u32,
    ]
}

/// A 256-bit value as the two committed words that hold it.
pub fn digest_words(d: &H256) -> [F128; 2] {
    [
        pack4([d[0], d[1], d[2], d[3]]),
        pack4([d[4], d[5], d[6], d[7]]),
    ]
}

/// The inverse of [`digest_words`].
pub fn words_digest(w: [F128; 2]) -> H256 {
    let (x, y) = (unpack4(w[0]), unpack4(w[1]));
    [x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3]]
}

// ---------------------------------------------------------------------------
// The note scheme
// ---------------------------------------------------------------------------

/// A 256-bit value as 8 `u32`s.
pub type H256 = [u32; SLOT_WORDS];

/// SHA-256 IV (FIPS 180-4 §5.3.3) — the fixed chaining value every hash uses.
pub const SHA256_IV: H256 = [
    0x6a09_e667,
    0xbb67_ae85,
    0x3c6e_f372,
    0xa54f_f53a,
    0x510e_527f,
    0x9b05_688c,
    0x1f83_d9ab,
    0x5be0_cd19,
];

/// Domain separation for the nullifier hash, so it can never coincide with a
/// commitment (which carries the random `secret` in the same slot).
pub const DOMAIN_NF: H256 = [
    0x544f_524e,
    0x4144_4f5f,
    0x4e55_4c4c,
    0x4946_4945,
    0x525f_5630,
    0x0000_0001,
    0,
    0,
];

/// `SHA256_c(h_in, left ‖ right)` — one compression.
pub fn hash_pair_from(h_in: &H256, left: &H256, right: &H256) -> H256 {
    let mut m = [0u32; 16];
    m[..8].copy_from_slice(left);
    m[8..].copy_from_slice(right);
    sha2::sha256_compress(h_in, &m)
}

/// `SHA256_c(IV, left ‖ right)`.
pub fn hash_pair(left: &H256, right: &H256) -> H256 {
    hash_pair_from(&SHA256_IV, left, right)
}

/// A deposit note.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Note {
    pub nullifier: H256,
    pub secret: H256,
}

impl Note {
    pub fn commitment(&self) -> H256 {
        hash_pair(&self.nullifier, &self.secret)
    }
    pub fn nullifier_hash(&self) -> H256 {
        hash_pair(&self.nullifier, &DOMAIN_NF)
    }
}

/// One withdrawal's private witness.
#[derive(Clone, Debug)]
pub struct Withdrawal {
    pub note: Note,
    pub siblings: Vec<H256>,
    /// `bits[d] = false` puts the running node LEFT at level `d`.
    pub bits: Vec<bool>,
}

impl Withdrawal {
    /// The public statement this witness attests to.
    pub fn public(&self) -> PublicInputs {
        let mut node = self.note.commitment();
        for (sib, &bit) in self.siblings.iter().zip(self.bits.iter()) {
            node = if bit {
                hash_pair(sib, &node)
            } else {
                hash_pair(&node, sib)
            };
        }
        PublicInputs {
            root: node,
            nullifier_hash: self.note.nullifier_hash(),
        }
    }
}

/// What a withdrawal reveals.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PublicInputs {
    pub root: H256,
    pub nullifier_hash: H256,
}

/// The leaf's digest over its withdrawals' public outputs, natively:
/// `d_0 = IV`, `d_{i+1} = SHA256_c(d_i, nh_i ‖ root_i)`.
pub fn native_leaf_digest(public: &[PublicInputs]) -> H256 {
    let mut d = SHA256_IV;
    for pi in public {
        d = hash_pair_from(&d, &pi.nullifier_hash, &pi.root);
    }
    d
}
