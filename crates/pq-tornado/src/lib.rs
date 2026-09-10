//! # pq-tornado
//!
//! A **post-quantum Tornado Cash** variant built on the [Flock] proving system.
//!
//! Classic Tornado Cash relies on two quantum-broken pieces: the Pedersen hash
//! (collision resistance = discrete log) for commitments, and a pairing-based
//! Groth16 zk-SNARK (with trusted setup). This crate replaces both:
//!
//! * **Hashing** → SHA-256 (only a quadratic Grover speedup, no structural
//!   break), which is exactly the primitive Flock's R1CS-over-GF(2) engine is
//!   fast at. See [`sha256`] and [`note`].
//! * **Proof system** → Flock / Ligerito, a transparent hash-and-code-based
//!   SNARK: no trusted setup, no pairings, plausibly post-quantum.
//!
//! The withdrawal statement is commitment opening, nullifier-hash derivation,
//! and a private Merkle authentication path. It is compiled two ways:
//!
//! * **stepwise** (preferred) — [`step`] holds *one* SHA-256 compression with
//!   an in-circuit multiplexer, block-diagonally repeated `2^n_log` times;
//!   [`wiring`] proves the cross-instance copy-constraints with a single small
//!   sumcheck against the same commitment; [`stepwise`] runs the pipeline. In
//!   the Flock paper's language the step circuit is the repeated relation `F`
//!   and [`wiring`] is an instantiation of the **glue circuit `G`**. Circuit
//!   size is independent of tree depth.
//! * **monolithic** — [`circuit`] wires all `D + 2` compressions into one R1CS
//!   block, proved by [`pipeline`]. Simpler, but `k_log` grows with the depth
//!   and most of the committed witness is padding.
//!
//! See `README.md` for the measured comparison (≈15× lower prove cost and ≈22×
//! lower verify cost at depth 32).
//!
//! ## Privacy caveat
//!
//! Soundness (a withdrawal proof can only be produced by someone who knows a
//! deposited note's preimage) is unconditional given SHA-256 collision
//! resistance and Flock's soundness. Full statistical **zero-knowledge** — i.e.
//! that the proof leaks nothing about *which* note is spent — additionally
//! requires a hiding/masked PCS layer that upstream Flock does not yet provide;
//! this is documented as a known limitation, not silently assumed.
//!
//! [Flock]: https://github.com/succinctlabs/flock

pub mod circuit;
pub mod note;
pub mod pipeline;
pub mod sha256;
pub mod step;
pub mod stepwise;
pub mod wiring;

pub use note::{H256, MerkleTree, Note};
