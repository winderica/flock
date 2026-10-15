//! **A post-quantum Tornado Cash withdrawal batch as a Flock workload.**
//!
//! This crate is a Tornado-shaped LEAF for [`flock_prover`]'s recursion
//! tower, built entirely on that crate's leaf extension surface
//! (`flock_prover::tower`: [`flock_prover::tower::TowerLeaf`],
//! [`flock_prover::tower::MixedInner`], [`flock_prover::tower::Online`],
//! [`flock_prover::tower::DOMAIN`], [`flock_prover::tower::TowerConfig`],
//! [`flock_prover::tower::Tower::prove_leaves`],
//! [`flock_prover::tower::TowerVk::from_tower`], and friends) — the same
//! surface `flock-prover`'s own BLAKE3 hash-chain leaf (`chain.rs`, inside
//! that crate) is built on. `flock-prover` itself knows nothing about
//! Tornado, withdrawals, or this crate.
//!
//! - [`note`] — the note scheme: deposit notes, withdrawals, and the
//!   SHA-256 hashes that bind them (the native, circuit-independent
//!   statement).
//! - [`step`] — the fused SHA-256+Merkle-swap R1CS table every hash in the
//!   leaf circuit is one row of.
//! - [`leaf`] — the leaf circuit itself
//!   ([`leaf::build_tornado_shape`]/[`leaf::build_tornado_leaf`]) and its
//!   `impl TowerLeaf for TornadoLeaf`.
//! - [`tower`] — driving `flock-prover`'s tower over Tornado leaves:
//!   [`tower::prove_tornado`], [`tower::generate_tornado_vk`], and the
//!   Tornado-specific view of the tower's kind-erased statement
//!   ([`tower::TornadoStatement`]).
//! - [`testing`] — deterministic synthetic withdrawals, shared by
//!   [`tower::generate_tornado_vk`]'s reference tower and this crate's own
//!   tests.

pub mod aggregator;
pub mod leaf;
pub mod note;
pub mod step;
pub mod testing;
pub mod tower;
pub mod user;

pub use aggregator::Aggregator;
pub use leaf::{
    AdmitError, LeafProof, TornadoLeaf, TornadoShape, admit_tornado_leaf, build_tornado_leaf,
    build_tornado_shape, prove_tornado_leaf, verify_tornado_leaf,
};
pub use note::{
    DOMAIN_NF, H256, Note, PublicInputs, SHA256_IV, Withdrawal, digest_words, hash_pair,
    hash_pair_from, native_leaf_digest, words_digest,
};
pub use tower::{TornadoStatement, generate_tornado_vk, prove_tornado};
pub use user::{UserProof, prove_withdrawal, verify_withdrawal};
