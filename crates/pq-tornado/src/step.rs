//! The **step circuit**: one SHA-256 compression with an in-circuit
//! left/right multiplexer, as a *small* R1CS-over-GF(2) block that is
//! block-diagonally repeated `2^n_log` times.
//!
//! This is the "small circuit, many instances" counterpart to
//! [`crate::circuit`]'s monolithic block. Every Tornado hash — the commitment
//! `SHA256_c(k ‖ r)`, the nullifier hash `SHA256_c(k ‖ DOMAIN_NF)`, and each
//! Merkle node `SHA256_c(left ‖ right)` — has the *same* shape
//!
//! ```text
//!   t     = bit · (sib ⊕ node)          (256 AND rows)
//!   left  = node ⊕ t                    (linear)
//!   right = sib  ⊕ t                    (linear)
//!   H_out = SHA256_c(IV, left ‖ right)
//! ```
//!
//! so one 2^15-wire block serves all of them:
//!
//! | `bit` | `node` | `sib` | computes |
//! |---|---|---|---|
//! | 0 | `k` | `r` | `cm = SHA256_c(k ‖ r)` |
//! | 0 | `k` | `DOMAIN_NF` | `nh = SHA256_c(k ‖ DOMAIN_NF)` |
//! | `b_d` | `node_d` | `sib_d` | `node_{d+1}` (Merkle level `d`) |
//!
//! The input chaining value is *pinned to the SHA-256 IV by constant supports*
//! (there are no free `H_in` wires the prover could choose), which is what
//! makes each instance the honest fixed-IV compression.
//!
//! Nothing in this block links instances together — that is the job of
//! [`crate::wiring`], which proves the public copy-constraints between
//! 256-bit-aligned *slots* of the committed witness.
//!
//! ## Slot layout (256-bit aligned, so the wiring layer can address them)
//!
//! ```text
//!   slot 0  bits    0.. 256   flags: bit 0 = constant-1 wire, bit 1 = mux bit
//!   slot 1  bits  256.. 512   node   (free)
//!   slot 2  bits  512.. 768   sib    (free)
//!   slot 3  bits  768..1024   t      (mux AND output)
//!   slot 4  bits 1024..1280   H_in   (pinned to the IV)
//!   slot 5  bits 1280..1536   H_out
//!   slot 6,7             ..2048   M = left ‖ right
//!   ...     bits 2048..32512   compression internals
//! ```

use flock_prover::r1cs::{BlockR1cs, SparseBinaryMatrix, WitnessLayout};

use crate::{
    note::H256,
    sha256::{
        COMP_STRIDE, CompLayout, Rows, SHA256_IV, Sup, build_compression, fill_compression_witness,
    },
};

/// `log2` of the block size in wires.
pub const K_LOG: usize = 15;
/// Univariate-skip dimension (matches Flock's hash encoders).
pub const K_SKIP: usize = 6;
/// Wires per block.
pub const K: usize = 1 << K_LOG;

/// Width of an addressable slot, in wires.
pub const SLOT_BITS: usize = 256;
/// Slots per block.
pub const SLOTS_PER_BLOCK: usize = K / SLOT_BITS; // 128

// ---- Wire / slot assignment ------------------------------------------------

/// Constant-1 wire (slot 0, bit 0).
pub const CONST_WIRE: usize = 0;
/// Private mux selector (slot 0, bit 1).
pub const BIT_WIRE: usize = 1;

/// Slot holding `node` — the chained 256-bit input.
pub const SLOT_NODE: usize = 1;
/// Slot holding `sib` — the free 256-bit input.
pub const SLOT_SIB: usize = 2;
/// Slot holding `t = bit·(sib ⊕ node)`. `t = 0` ⟺ the mux is the identity.
pub const SLOT_T: usize = 3;

const NODE_BASE: usize = SLOT_NODE * SLOT_BITS;
const SIB_BASE: usize = SLOT_SIB * SLOT_BITS;
const T_BASE: usize = SLOT_T * SLOT_BITS;

/// First wire of the SHA-256 compression sub-block.
const COMP_BASE: usize = 4 * SLOT_BITS; // 1024

/// Slot holding the compression output `H_out`.
pub const SLOT_HOUT: usize = (COMP_BASE + 256) / SLOT_BITS; // 5

/// Real (non-padding) wires per block.
pub const USEFUL_BITS: usize = COMP_BASE + COMP_STRIDE; // 32512

const _: () = assert!(USEFUL_BITS <= K);

/// One instance's private inputs.
#[derive(Clone, Copy, Debug, Default)]
pub struct StepInput {
    pub node: H256,
    pub sib: H256,
    pub bit: bool,
}

impl StepInput {
    /// The compression's output `H_out`.
    pub fn output(&self) -> H256 {
        let (l, r) = self.mux();
        crate::note::hash_pair(&l, &r)
    }
    fn mux(&self) -> (H256, H256) {
        if self.bit {
            (self.sib, self.node)
        } else {
            (self.node, self.sib)
        }
    }
}

/// The block-diagonal R1CS over `2^n_log` copies of the step block.
pub struct StepCircuit {
    pub n_log: usize,
    pub r1cs: BlockR1cs,
}

fn const_support_h256(v: &H256) -> Vec<Sup> {
    (0..256)
        .map(|i| {
            if (v[i / 32] >> (i % 32)) & 1 == 1 {
                vec![CONST_WIRE]
            } else {
                Sup::new()
            }
        })
        .collect()
}

impl StepCircuit {
    /// Build the R1CS for `2^n_log` independent step instances.
    pub fn build(n_log: usize) -> Self {
        assert!(n_log >= 3, "flock's lincheck needs n_outer >= 8");

        let mut a: Vec<Sup> = vec![Sup::new(); K];
        let mut b: Vec<Sup> = vec![Sup::new(); K];

        // constant-1 wire: z·z = z, pinned to 1 by `const_pin`.
        a[CONST_WIRE] = vec![CONST_WIRE];
        b[CONST_WIRE] = vec![CONST_WIRE];

        // Free inputs (`z · 1 = z` tautology rows).
        for w in std::iter::once(BIT_WIRE)
            .chain(NODE_BASE..NODE_BASE + 256)
            .chain(SIB_BASE..SIB_BASE + 256)
        {
            a[w] = vec![w];
            b[w] = vec![CONST_WIRE];
        }

        // Mux AND rows: t_i = bit · (sib_i ⊕ node_i).
        for i in 0..256 {
            let t = T_BASE + i;
            a[t] = vec![BIT_WIRE];
            let (x, y) = (NODE_BASE + i, SIB_BASE + i);
            b[t] = vec![x.min(y), x.max(y)];
        }

        // The compression: H_in pinned to the IV, message = left ‖ right.
        let iv_support: [Sup; 256] = const_support_h256(&SHA256_IV).try_into().unwrap();
        let m_support: [Sup; 512] = (0..512)
            .map(|j| {
                let (src, i) = if j < 256 {
                    (NODE_BASE, j)
                } else {
                    (SIB_BASE, j - 256)
                };
                let (x, y) = (src + i, T_BASE + i);
                vec![x.min(y), x.max(y)]
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let mut rows = Rows {
            a: &mut a,
            b: &mut b,
            const_wire: CONST_WIRE,
        };
        build_compression(
            &mut rows,
            CompLayout { base: COMP_BASE },
            &iv_support,
            &m_support,
        );

        let mk = |rows: Vec<Sup>| SparseBinaryMatrix {
            num_rows: K,
            num_cols: K,
            rows,
        };
        let r1cs = BlockR1cs {
            m: K_LOG + n_log,
            k_log: K_LOG,
            k_skip: K_SKIP,
            useful_bits: USEFUL_BITS,
            a_0: mk(a),
            b_0: mk(b),
            c_0: mk((0..K).map(|i| vec![i]).collect()),
            layout: WitnessLayout::RowMajor,
            const_pin: Some(CONST_WIRE),
            digest_cache: std::sync::OnceLock::new(),
            csc_cache: std::sync::OnceLock::new(),
        };
        // Warm the CSC transpose (setup cost, kept off the prove path).
        r1cs.csc_lincheck_circuit();

        Self { n_log, r1cs }
    }

    /// Number of instances.
    pub fn n_instances(&self) -> usize {
        1 << self.n_log
    }

    /// Global 256-bit slot index of `slot` inside instance `i` (row-major).
    #[inline]
    pub fn slot_index(i: usize, slot: usize) -> usize {
        i * SLOTS_PER_BLOCK + slot
    }

    /// Boolean witness for all `2^n_log` instances (length `2^m`). Unspecified
    /// instances are filled with the (valid) all-zero-input compression.
    pub fn generate_witness(&self, inputs: &[StepInput]) -> Vec<bool> {
        let n = self.n_instances();
        assert!(inputs.len() <= n, "too many instances");
        let mut z = vec![false; n << K_LOG];
        let pad = StepInput::default();
        // Instances are independent — fill them in parallel.
        use rayon::prelude::*;
        z.par_chunks_mut(K).enumerate().for_each(|(i, blk)| {
            fill_step(blk, inputs.get(i).unwrap_or(&pad));
        });
        z
    }
}

/// Fill one instance's block witness.
pub fn fill_step(z: &mut [bool], inp: &StepInput) {
    debug_assert_eq!(z.len(), K);
    z[CONST_WIRE] = true;
    z[BIT_WIRE] = inp.bit;
    for i in 0..256 {
        let (w, b) = (i / 32, i % 32);
        let node_bit = (inp.node[w] >> b) & 1 == 1;
        let sib_bit = (inp.sib[w] >> b) & 1 == 1;
        z[NODE_BASE + i] = node_bit;
        z[SIB_BASE + i] = sib_bit;
        z[T_BASE + i] = inp.bit && (node_bit ^ sib_bit);
    }
    let (l, r) = inp.mux();
    let mut m = [0u32; 16];
    m[..8].copy_from_slice(&l);
    m[8..].copy_from_slice(&r);
    fill_compression_witness(z, CompLayout { base: COMP_BASE }, &SHA256_IV, &m);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_block_satisfies_r1cs() {
        let c = StepCircuit::build(3);
        let inputs: Vec<StepInput> = (0..8u32)
            .map(|i| StepInput {
                node: [i; 8],
                sib: [i.wrapping_mul(31) + 7; 8],
                bit: i % 2 == 1,
            })
            .collect();
        let z = c.generate_witness(&inputs);
        assert!(c.r1cs.satisfies(&z));

        // H_out slot really holds the muxed compression output.
        for (i, inp) in inputs.iter().enumerate() {
            let want = inp.output();
            let base = i * K + SLOT_HOUT * SLOT_BITS;
            for j in 0..256 {
                assert_eq!(z[base + j], (want[j / 32] >> (j % 32)) & 1 == 1);
            }
        }
    }
}
