//! Deterministic synthetic withdrawals — [`generate_tornado_vk`]'s
//! reference tower and this crate's own tests share one generator so the
//! shapes they materialize agree. `pub` (not `#[cfg(test)]`): a VK is
//! generated at runtime, in release builds, from these same samples.
//!
//! [`generate_tornado_vk`]: crate::tower::generate_tornado_vk

use std::array::from_fn;

use crate::note::{H256, Note, Withdrawal};

pub fn sample_note(seed: u32) -> Note {
    Note {
        nullifier: from_fn(|i| seed.wrapping_mul(0x9E37_79B1).wrapping_add(i as u32)),
        secret: from_fn(|i| {
            seed.wrapping_mul(0x85EB_CA77)
                .wrapping_add(0x1000 + i as u32)
        }),
    }
}

pub fn sample_withdrawal(seed: u32, depth: usize) -> Withdrawal {
    let note = sample_note(seed);
    let siblings: Vec<H256> = (0..depth)
        .map(|d| sample_note(seed ^ (0xBEEF + d as u32)).nullifier)
        .collect();
    let bits: Vec<bool> = (0..depth).map(|d| (seed >> (d % 32)) & 1 == 1).collect();
    Withdrawal {
        note,
        siblings,
        bits,
    }
}

/// `n` withdrawals seeded from `seed`.
pub fn sample_withdrawal_set(seed: u32, depth: usize, n: usize) -> Vec<Withdrawal> {
    (0..n)
        .map(|i| sample_withdrawal(seed + i as u32, depth))
        .collect()
}

/// Bench/test knob: which production tower this crate's `#[ignore]`d tests
/// exercise. `TOWER_CONFIG=chain100` selects the 100-bit tower;
/// `TOWER_CONFIG=chain128`, or unset, the 128-bit production tower. Any
/// other value is a typo and panics instead of silently selecting the
/// default. Mirrors `flock-prover`'s own (crate-private) test knob of the
/// same name.
#[cfg(test)]
pub fn test_config() -> flock_prover::tower::TowerConfig {
    use std::env::{VarError, var};

    use flock_prover::tower::TowerConfig;
    match var("TOWER_CONFIG").as_deref() {
        Ok("chain100") => TowerConfig::Chain100,
        Ok("chain128") | Err(VarError::NotPresent) => TowerConfig::Chain128,
        other => panic!("TOWER_CONFIG must be `chain100` or `chain128`, got {other:?}"),
    }
}
