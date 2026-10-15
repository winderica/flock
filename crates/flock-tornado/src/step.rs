//! **The SHA-256 step table**: one compression with the Merkle-level
//! conditional swap FUSED into the same block, so a Merkle-path workload is
//! ONE boolean table type.
//!
//! ```text
//!   t_j     = (1 + bit) · (node_j ⊕ sib_j)      j ∈ 0..256
//!   left_j  = sib_j  ⊕ t_j
//!   right_j = node_j ⊕ t_j
//!   H_out   = SHA256_c(H_in, left ‖ right)
//! ```
//!
//! `bit = 0` puts `node` LEFT. With `bit = 0` and free `H_in` the row is a
//! plain compression `SHA256_c(H_in, node ‖ sib)`, which is how the same
//! table also serves the note hashes ([`crate::note`]) and the leaf's
//! digest chain ([`crate::leaf`]).
//!
//! The SHA-256 half IS [`flock_prover::r1cs_hashes::sha2`]'s block — same
//! rows, same layout, same witness driver — with two changes:
//!
//! 1. the 512 `M` rows stop being free inputs and become the swap's linear
//!    outputs (`A` support `{node_j, t_j}` / `{sib_j, t_j}`, `B` the constant),
//!    so `A·z` at those rows is still the `M` bit the driver already writes;
//! 2. four word-aligned regions are appended past `sha2::USEFUL_BITS`: the
//!    bit word, `node`, `sib`, and `t` (the one AND per bit).
//!
//! Why fuse rather than keep two tables: the recursion tower's first-level
//! node folds exactly ONE boolean type per leaf (`num_boolean() == 1`), and
//! its accumulator lane is sized for that. One table keeps the tower's fold
//! groups, envelope and tape geometry untouched.
//!
//! ## Layout (`K_LOG = 15`)
//!
//! ```text
//!   [0, 25470)         sha2's block verbatim (H_in @ 0, H_out @ 256, M @ 512, …)
//!   [25472, 25600)     bit word   (the relation reads bit 0; the rest ride free)
//!   [25600, 25856)     node
//!   [25856, 26112)     sib
//!   [26112, 26368)     t
//! ```

use flock_core::{
    r1cs::{BlockR1cs, SparseBinaryMatrix},
    schedule::IoWord,
    union::SlotWitnessDest,
};
use flock_field::F128;
use flock_prover::r1cs_hashes::{
    common::{
        BM_V, BmRow, build_block_r1cs_with_matrices, drive_witness_batch_major_partial,
        drive_witness_batch_major_partial_into, or_bit_row, or_u32_row,
    },
    sha2::{self, K, K_LOG, K_SKIP, M_BASE, SLOT_BITS, WORD_BITS, Z_CONST_POS},
};

/// The appended regions, word-aligned past sha2's useful bits.
pub const BIT_BASE: usize = sha2::USEFUL_BITS.next_multiple_of(128); // 25,472
pub const NODE_BASE: usize = BIT_BASE + 128; // 25,600
pub const SIB_BASE: usize = NODE_BASE + SLOT_BITS; // 25,856
pub const T_BASE: usize = SIB_BASE + SLOT_BITS; // 26,112
pub const USEFUL_BITS: usize = T_BASE + SLOT_BITS; // 26,368

const _: () = assert!(USEFUL_BITS <= K);
const _: () = assert!(BIT_BASE.is_multiple_of(128) && NODE_BASE.is_multiple_of(128));

/// One step row's inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepRow {
    pub h_in: [u32; 8],
    pub bit: bool,
    pub node: [u32; 8],
    pub sib: [u32; 8],
}

impl StepRow {
    /// `(left, right)` after the swap.
    pub fn mux(&self) -> ([u32; 8], [u32; 8]) {
        if self.bit {
            (self.sib, self.node)
        } else {
            (self.node, self.sib)
        }
    }

    /// The compression the sha2 half proves.
    pub fn compression(&self) -> sha2::Compression {
        let (l, r) = self.mux();
        let mut m = [0u32; 16];
        m[..8].copy_from_slice(&l);
        m[8..].copy_from_slice(&r);
        (self.h_in, m)
    }

    /// `H_out`.
    pub fn output(&self) -> [u32; 8] {
        let (h, m) = self.compression();
        sha2::sha256_compress(&h, &m)
    }
}

// ---------------------------------------------------------------------------
// IO schema
// ---------------------------------------------------------------------------

pub const IO_H_IN0: usize = 0;
pub const IO_BIT: usize = 2;
pub const IO_NODE0: usize = 3;
pub const IO_SIB0: usize = 5;
pub const IO_H_OUT0: usize = 7;

/// Inputs: `H_in` (2 words), the bit word, `node` (2), `sib` (2). Outputs:
/// `H_out` (2). Everything a caller supplies is here; `M` is internal (the
/// swap's output) and `t` is internal (the AND).
pub fn io_schema() -> Vec<IoWord> {
    let w = |bit: usize| {
        debug_assert_eq!(bit % 128, 0);
        bit / 128
    };
    vec![
        IoWord::input(w(sha2::H_BASE)),
        IoWord::input(w(sha2::H_BASE) + 1),
        IoWord::input(w(BIT_BASE)),
        IoWord::input(w(NODE_BASE)),
        IoWord::input(w(NODE_BASE) + 1),
        IoWord::input(w(SIB_BASE)),
        IoWord::input(w(SIB_BASE) + 1),
        IoWord::output(w(sha2::H_OUT_BASE)),
        IoWord::output(w(sha2::H_OUT_BASE) + 1),
    ]
}

// ---------------------------------------------------------------------------
// Matrices
// ---------------------------------------------------------------------------

/// `(A_0, B_0)`: sha2's block with the `M` rows rewired to the swap and the
/// swap's own rows appended. `C_0 = I`.
pub fn build_matrices() -> (SparseBinaryMatrix, SparseBinaryMatrix) {
    let (a_sha, b_sha) = sha2::build_matrices();
    let mut a: Vec<Vec<usize>> = (*a_sha.rows).clone();
    let mut b: Vec<Vec<usize>> = (*b_sha.rows).clone();
    assert_eq!(a.len(), K);
    let gc = Z_CONST_POS;

    let free = |a: &mut Vec<Vec<usize>>, b: &mut Vec<Vec<usize>>, r: usize| {
        a[r] = vec![r];
        b[r] = vec![gc];
    };
    for j in 0..128 {
        free(&mut a, &mut b, BIT_BASE + j);
    }
    for j in 0..SLOT_BITS {
        free(&mut a, &mut b, NODE_BASE + j);
        free(&mut a, &mut b, SIB_BASE + j);
    }
    for j in 0..SLOT_BITS {
        // t_j = (1 + bit) · (node_j ⊕ sib_j) — the only AND.
        a[T_BASE + j] = vec![gc, BIT_BASE];
        b[T_BASE + j] = sorted(vec![NODE_BASE + j, SIB_BASE + j]);
        // left  = M[0..8]  : sib ⊕ t
        // right = M[8..16] : node ⊕ t
        let (mi, mb) = (j / WORD_BITS, j % WORD_BITS);
        let left = sha2::m_bit(mi, mb);
        let right = sha2::m_bit(8 + mi, mb);
        debug_assert_eq!(left, M_BASE + j);
        a[left] = sorted(vec![SIB_BASE + j, T_BASE + j]);
        b[left] = vec![gc];
        a[right] = sorted(vec![NODE_BASE + j, T_BASE + j]);
        b[right] = vec![gc];
    }
    let m = |rows| SparseBinaryMatrix::new(K, K, rows);
    (m(a), m(b))
}

fn sorted(mut v: Vec<usize>) -> Vec<usize> {
    v.sort_unstable();
    v
}

pub fn build_block_r1cs(n_log: usize) -> BlockR1cs {
    let (a_0, b_0) = build_matrices();
    build_block_r1cs_with_matrices(
        n_log,
        K_LOG,
        K_SKIP,
        USEFUL_BITS,
        a_0,
        b_0,
        Some(Z_CONST_POS),
    )
}

// ---------------------------------------------------------------------------
// Witness
// ---------------------------------------------------------------------------

/// Eight rows in lockstep: the sha2 driver fills the compression half, then
/// the swap's regions are appended. `A·z` at the `M` rows is what sha2 wrote
/// (the swap's outputs ARE the message), so only the new rows need `a`/`b`.
fn build_group(inputs: [&StepRow; BM_V], rz: &mut [BmRow], ra: &mut [BmRow], rb: &mut [BmRow]) {
    let comps: [sha2::Compression; BM_V] = std::array::from_fn(|j| inputs[j].compression());
    sha2::build_group_batch_major(std::array::from_fn(|j| &comps[j]), rz, ra, rb);

    let bits: [u32; BM_V] = std::array::from_fn(|j| u32::from(inputs[j].bit));
    let nbit: [u32; BM_V] = std::array::from_fn(|j| 1 ^ bits[j]);
    // The bit word: 128 free rows (z, a = z, b = 1); only column 0 carries
    // the bit, the rest are zero.
    or_u32_row(rz, BIT_BASE, &bits);
    or_u32_row(ra, BIT_BASE, &bits);
    for j in 0..128 {
        or_bit_row(rb, BIT_BASE + j);
    }
    for w in 0..8 {
        let node: [u32; BM_V] = std::array::from_fn(|j| inputs[j].node[w]);
        let sib: [u32; BM_V] = std::array::from_fn(|j| inputs[j].sib[w]);
        let x: [u32; BM_V] = std::array::from_fn(|j| node[j] ^ sib[j]);
        // t = (1 + bit) · (node ⊕ sib): a = 1+bit (replicated), b = x.
        let t: [u32; BM_V] = std::array::from_fn(|j| x[j] & nbit[j].wrapping_neg());
        let nb32: [u32; BM_V] = std::array::from_fn(|j| nbit[j].wrapping_neg());
        for (buf, v) in [(&mut *rz, &node), (&mut *ra, &node)] {
            or_u32_row(buf, NODE_BASE + WORD_BITS * w, v);
        }
        for (buf, v) in [(&mut *rz, &sib), (&mut *ra, &sib)] {
            or_u32_row(buf, SIB_BASE + WORD_BITS * w, v);
        }
        or_u32_row(rz, T_BASE + WORD_BITS * w, &t);
        or_u32_row(ra, T_BASE + WORD_BITS * w, &nb32);
        or_u32_row(rb, T_BASE + WORD_BITS * w, &x);
    }
    // b = 1 on the free node/sib rows.
    for j in 0..SLOT_BITS {
        or_bit_row(rb, NODE_BASE + j);
        or_bit_row(rb, SIB_BASE + j);
    }
}

pub fn generate_witness_batch_major_partial(
    rows: &[StepRow],
    n_log: usize,
) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
    drive_witness_batch_major_partial(rows, n_log, K_LOG, USEFUL_BITS, build_group)
}

pub fn generate_witness_batch_major_partial_into(
    rows: &[StepRow],
    n_log: usize,
    dst: SlotWitnessDest<'_>,
) -> Vec<u8> {
    drive_witness_batch_major_partial_into(rows, n_log, K_LOG, USEFUL_BITS, dst, build_group)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_row(seed: u32) -> StepRow {
        let g = |k: u32| -> [u32; 8] {
            std::array::from_fn(|i| {
                seed.wrapping_mul(0x9E37_79B1 ^ k)
                    .wrapping_add(i as u32 * 7)
            })
        };
        StepRow {
            h_in: g(1),
            bit: seed % 2 == 1,
            node: g(2),
            sib: g(3),
        }
    }

    /// Batch-major → row-major: chunk-column `c` of instance `o`.
    fn to_row_major(z_b: &[F128], n_log: usize) -> Vec<F128> {
        let chunks = K / 128;
        let mut out = vec![F128::ZERO; z_b.len()];
        for o in 0..1usize << n_log {
            for c in 0..chunks {
                out[o * chunks + c] = z_b[(c << n_log) + o];
            }
        }
        out
    }

    fn word_bits(z: &[F128], base: usize) -> [u32; 8] {
        std::array::from_fn(|w| {
            let mut v = 0u32;
            for b in 0..32 {
                let g = base + 32 * w + b;
                let e = z[g >> 7];
                let s = g & 127;
                let bit = if s < 64 {
                    (e.lo >> s) & 1
                } else {
                    (e.hi >> (s - 64)) & 1
                };
                v |= (bit as u32) << b;
            }
            v
        })
    }

    /// The fused block is satisfied by the driver's witness (as the R1CS
    /// itself checks it, row-major), the prover's `a`/`b` match `A·z`/`B·z`,
    /// and `H_out` is the muxed compression.
    #[test]
    fn step_block_is_satisfied_and_outputs_the_muxed_hash() {
        let n_log = 3;
        let r1cs = build_block_r1cs(n_log);
        let rows: Vec<StepRow> = (0..8).map(sample_row).collect();
        let (z_b, a_b, b_b, _) = generate_witness_batch_major_partial(&rows, n_log);
        let z = to_row_major(&z_b, n_log);
        assert!(r1cs.satisfies_packed(&z), "the fused R1CS is satisfied");
        assert_eq!(
            r1cs.apply_a_packed(&z),
            to_row_major(&a_b, n_log),
            "a = A·z"
        );
        let bz = r1cs.apply_b_packed(&z);
        let bd = to_row_major(&b_b, n_log);
        let chunks0 = K / 128;
        for (i, (x, y)) in bz.iter().zip(bd.iter()).enumerate() {
            if x != y {
                let (o, c) = (i / chunks0, i % chunks0);
                panic!(
                    "b = B·z: instance {o} chunk {c} (bits {}..{}): B·z={x:?} driver={y:?}",
                    c * 128,
                    c * 128 + 128
                );
            }
        }

        let chunks = K / 128;
        for (o, row) in rows.iter().enumerate() {
            let blk = &z[o * chunks..(o + 1) * chunks];
            assert_eq!(
                word_bits(blk, sha2::H_OUT_BASE),
                row.output(),
                "row {o} H_out"
            );
            let (l, r) = row.mux();
            assert_eq!(word_bits(blk, M_BASE), l, "row {o} left");
            assert_eq!(word_bits(blk, M_BASE + SLOT_BITS), r, "row {o} right");
        }
    }

    /// A flipped bit anywhere in the swap region breaks satisfaction.
    #[test]
    fn step_block_rejects_a_tampered_swap() {
        let n_log = 3;
        let r1cs = build_block_r1cs(n_log);
        let rows: Vec<StepRow> = (0..8).map(sample_row).collect();
        let (z_b, _, _, _) = generate_witness_batch_major_partial(&rows, n_log);
        let mut z = to_row_major(&z_b, n_log);
        // Flip t_5 of instance 2.
        let g = 2 * K + T_BASE + 5;
        let e = &mut z[g >> 7];
        let s = g & 127;
        if s < 64 {
            e.lo ^= 1 << s;
        } else {
            e.hi ^= 1 << (s - 64);
        }
        assert!(!r1cs.satisfies_packed(&z));
    }
}
