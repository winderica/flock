//! **THE LEAF TRAIT**: what the tower's first-level node, internal node,
//! driver and standalone verifier need from a workload's leaf proof —
//! without naming the workload. A leaf is any single-boolean-type
//! union-circuit proof whose own public segment carries an 8-word (or
//! narrower, zero-padded) application statement that a first-level node's
//! [`AppKind`] knows how to compose two of.
//!
//! A workload plugs into the tower by implementing [`TowerLeaf`] on its own
//! leaf proof type: `impl TowerLeaf for ChainProof` lives in `chain.rs`; a
//! second workload crate's own leaf type implements it the same way,
//! entirely outside this crate. Everything upward of the leaf —
//! [`crate::tower::fl_node::build_first_level_node_kids`], the internal
//! node, [`crate::tower::driver::Tower`], and
//! [`crate::tower::verify::verify_root`] — talks to leaves only through
//! this trait (and the generic [`crate::tower::driver::AppStatement`] the
//! app words settle into), so a third workload plugs in without touching
//! any of them.

use std::sync::Arc;

use flock_core::r1cs::BlockR1cs;

use crate::tower::{F128, MixedInner, node::AppKind, online::Online};

/// A single-boolean-type leaf workload the tower's first-level node folds
/// two of. `pub`, and [`MixedInner`]/[`Online`] travel with it, so a
/// workload crate outside this one can implement it on its own leaf proof
/// type and drive the tower exactly as `chain.rs` does.
pub trait TowerLeaf {
    /// The leaf's union-circuit proof and its deferred verification data —
    /// what the generic child walker replays.
    fn inner(&self) -> &MixedInner;
    /// The leaf's ONE boolean table, at its own `nu` — cached process-wide
    /// by the implementor (every leaf, the FL's fold matrices, and the
    /// lane's discharge tables all read the same cache).
    fn r1cs(&self) -> Arc<BlockR1cs>;
    /// The span counter's unit: compressions for a BLAKE3 chain leaf, or
    /// whatever unit a workload's own leaf counts.
    fn count(&self) -> usize;
    /// How this leaf's app block composes upward — chain adjacency or a
    /// digest-tree combine (see [`AppKind`]).
    fn app_kind(&self) -> AppKind;
    /// The leaf's own application statement, natively, as the 8 packed
    /// F128 words a node's app block eventually publishes: a chain leaf's
    /// `[h_start x4 | h_end x4]`, a digest leaf's `[D0, D1, 0 x6]`.
    fn app_words(&self) -> [F128; 8];
    /// `(offset, len)` of the RAW app material inside the leaf's own
    /// public segment — chain: offset 3, len 8 (`h_start` then `h_end`);
    /// digest: offset 5, len 2 (`D0`, `D1`) — the first-level node wires
    /// its composition from here (`AppKind::emit_compose`).
    fn app_pub_range(&self) -> (usize, usize);
    /// This leaf's online cost record — folded into the driver's
    /// [`crate::tower::driver::TowerCosts`] ledger.
    fn online(&self) -> Online;
}
