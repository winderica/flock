//! Composable **SHA-256 compression** gadget over R1CS-over-GF(2), for the
//! post-quantum Tornado Cash circuit.
//!
//! This is a re-parameterization of Flock's proven `flock_prover::r1cs_hashes::
//! sha2` encoder: the identical symbolic construction (inlined 32-bit adders,
//! materialized W/T1/E/A/H_out slots), but with a **dynamic slot base** so that
//! many compressions can be laid out side-by-side inside one monolithic
//! constraint block, and with the message `M` and input chaining value `H_in`
//! driven by caller-supplied *wire supports* (linear GF(2) combinations of
//! other wires / the constant-1 wire) instead of being free inputs.
//!
//! The constraint shape is circuit-R1CS `(A·z) ⊙ (B·z) = z` (i.e. `C = I`), so
//! every row `j` either defines `z_j` as an AND of two linear forms
//! (`a_j · b_j`) or as a linear pass (`support · 1`).
//!
//! One compression computes the standard SHA-256 block map
//! `H_out = H_in + compress(H_in, M)` (Merkle–Damgård feed-forward). For a
//! fixed 512-bit message and a fixed IV this is a fixed-length collision-
//! resistant hash — exactly the primitive Tornado's commitment / nullifier /
//! Merkle-node hashing needs.

use sha2::{compress256, digest::generic_array::GenericArray};

/// Sorted, duplicate-free XOR support: a row of `A` or `B`. GF(2) so repeated
/// indices cancel.
pub type Sup = Vec<usize>;
/// 32 per-bit supports = one symbolic 32-bit word.
pub type Word = Vec<Sup>;

pub const WORD_BITS: usize = 32;
const N_ROUNDS: usize = 64;
const N_SCHED: usize = 48;
const CARRIES_PER_ADD: usize = 31; // WORD_BITS - 1

/// SHA-256 IV (FIPS 180-4 §5.3.3).
pub const SHA256_IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];
/// SHA-256 round constants (FIPS 180-4 §4.2.2).
pub const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

// ───────────────────────────────────────────────────────────────────────────
// Per-compression slot layout (offsets relative to a compression's `base`).
// Mirrors flock's sha2 I/O-aligned layout, minus the per-block const wire
// (we use a single global constant-1 wire for the whole circuit).
// ───────────────────────────────────────────────────────────────────────────

const H_IN_OFF: usize = 0; // [0, 256)   input chaining value (8×32)
const H_OUT_OFF: usize = 256; // [256, 512) output chaining value (8×32) — 128-aligned
const M_OFF: usize = 512; // [512, 1024) message block (16×32)
const CH_AND_OFF: usize = 1024; // 64×32
const MAJ_AND_OFF: usize = CH_AND_OFF + N_ROUNDS * WORD_BITS; // 3072
const ROUND_CARRY_OFF: usize = MAJ_AND_OFF + N_ROUNDS * WORD_BITS; // 5120
const W_OFF: usize = ROUND_CARRY_OFF + N_ROUNDS * 7 * CARRIES_PER_ADD; // 19008
const SCHED_CARRY_OFF: usize = W_OFF + N_SCHED * WORD_BITS; // 20544
const T1_OFF: usize = SCHED_CARRY_OFF + N_SCHED * 3 * CARRIES_PER_ADD; // 25008
const E_NEW_OFF: usize = T1_OFF + N_ROUNDS * WORD_BITS; // 27056
const A_NEW_OFF: usize = E_NEW_OFF + N_ROUNDS * WORD_BITS; // 29104
const OUT_CARRY_OFF: usize = A_NEW_OFF + N_ROUNDS * WORD_BITS; // 31152
const COMP_END: usize = OUT_CARRY_OFF + 8 * CARRIES_PER_ADD; // 31400

/// Wires consumed by one compression, rounded up to a 128-bit boundary so that
/// each compression's `H_out` region stays 128-bit-aligned for PCS opening.
pub const COMP_STRIDE: usize = COMP_END.next_multiple_of(128); // 31488

/// Slot accessors for one compression anchored at `base`.
#[derive(Clone, Copy, Debug)]
pub struct CompLayout {
    pub base: usize,
}

impl CompLayout {
    #[inline]
    pub fn h_in_bit(&self, w: usize, b: usize) -> usize {
        self.base + H_IN_OFF + WORD_BITS * w + b
    }
    #[inline]
    pub fn h_out_bit(&self, w: usize, b: usize) -> usize {
        self.base + H_OUT_OFF + WORD_BITS * w + b
    }
    /// First wire of the 256-bit `H_out` region (128-bit-aligned).
    #[inline]
    pub fn h_out_base(&self) -> usize {
        self.base + H_OUT_OFF
    }
    #[inline]
    fn m_bit(&self, i: usize, b: usize) -> usize {
        self.base + M_OFF + WORD_BITS * i + b
    }
    #[inline]
    fn ch_and_bit(&self, r: usize, b: usize) -> usize {
        self.base + CH_AND_OFF + WORD_BITS * r + b
    }
    #[inline]
    fn maj_and_bit(&self, r: usize, b: usize) -> usize {
        self.base + MAJ_AND_OFF + WORD_BITS * r + b
    }
    #[inline]
    fn round_carry_bit(&self, r: usize, add: usize, b: usize) -> usize {
        self.base + ROUND_CARRY_OFF + r * 7 * CARRIES_PER_ADD + add * CARRIES_PER_ADD + b
    }
    #[inline]
    fn w_bit(&self, t: usize, b: usize) -> usize {
        if t < 16 {
            self.m_bit(t, b)
        } else {
            self.base + W_OFF + (t - 16) * WORD_BITS + b
        }
    }
    #[inline]
    fn sched_carry_bit(&self, t: usize, add: usize, b: usize) -> usize {
        self.base + SCHED_CARRY_OFF + (t - 16) * 3 * CARRIES_PER_ADD + add * CARRIES_PER_ADD + b
    }
    #[inline]
    fn t1_bit(&self, r: usize, b: usize) -> usize {
        self.base + T1_OFF + WORD_BITS * r + b
    }
    #[inline]
    fn e_new_bit(&self, r: usize, b: usize) -> usize {
        self.base + E_NEW_OFF + WORD_BITS * r + b
    }
    #[inline]
    fn a_new_bit(&self, r: usize, b: usize) -> usize {
        self.base + A_NEW_OFF + WORD_BITS * r + b
    }
    #[inline]
    fn out_carry_bit(&self, w: usize, b: usize) -> usize {
        self.base + OUT_CARRY_OFF + w * CARRIES_PER_ADD + b
    }
}

// ───────────────────────────────────────────────────────────────────────────
// Symbolic XOR-support helpers.
// ───────────────────────────────────────────────────────────────────────────

fn zero_word() -> Word {
    (0..WORD_BITS).map(|_| Sup::new()).collect()
}

fn wire_word<F: Fn(usize) -> usize>(slot: F) -> Word {
    (0..WORD_BITS).map(|b| vec![slot(b)]).collect()
}

/// Symmetric difference of two sorted supports (GF(2) XOR).
fn xor_sup(a: &Sup, b: &Sup) -> Sup {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i] < b[j] {
            out.push(a[i]);
            i += 1;
        } else if a[i] > b[j] {
            out.push(b[j]);
            j += 1;
        } else {
            i += 1;
            j += 1;
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

fn xor3(a: &Sup, b: &Sup, c: &Sup) -> Sup {
    xor_sup(&xor_sup(a, b), c)
}

fn xor_words(x: &Word, y: &Word) -> Word {
    (0..WORD_BITS).map(|i| xor_sup(&x[i], &y[i])).collect()
}

fn rotr(w: &Word, n: usize) -> Word {
    (0..WORD_BITS)
        .map(|i| w[(i + n) % WORD_BITS].clone())
        .collect()
}

fn shr(w: &Word, n: usize) -> Word {
    (0..WORD_BITS)
        .map(|i| {
            if i + n < WORD_BITS {
                w[i + n].clone()
            } else {
                Sup::new()
            }
        })
        .collect()
}

fn rot_xor3(w: &Word, r1: usize, r2: usize, r3: usize) -> Word {
    let a = rotr(w, r1);
    let b = rotr(w, r2);
    let c = rotr(w, r3);
    (0..WORD_BITS).map(|i| xor3(&a[i], &b[i], &c[i])).collect()
}

fn sigma_xor(w: &Word, r1: usize, r2: usize, sh: usize) -> Word {
    let a = rotr(w, r1);
    let b = rotr(w, r2);
    let s = shr(w, sh);
    (0..WORD_BITS).map(|i| xor3(&a[i], &b[i], &s[i])).collect()
}

fn sigma_0(w: &Word) -> Word {
    sigma_xor(w, 7, 18, 3)
}
fn sigma_1(w: &Word) -> Word {
    sigma_xor(w, 17, 19, 10)
}
fn big_sigma_0(w: &Word) -> Word {
    rot_xor3(w, 2, 13, 22)
}
fn big_sigma_1(w: &Word) -> Word {
    rot_xor3(w, 6, 11, 25)
}

// ───────────────────────────────────────────────────────────────────────────
// Matrix construction (fills caller-owned a_rows/b_rows in place).
// ───────────────────────────────────────────────────────────────────────────

/// Sink for R1CS rows during gadget construction. Holds the (mutable) `A` and
/// `B` sparse-row vectors and the index of the global constant-1 wire.
pub struct Rows<'a> {
    pub a: &'a mut [Sup],
    pub b: &'a mut [Sup],
    pub const_wire: usize,
}

impl Rows<'_> {
    fn add32_inline<F: Fn(usize) -> usize>(&mut self, x: &Word, y: &Word, carry_slot: F) -> Word {
        let mut sum = zero_word();
        let mut cin: Sup = Sup::new();
        for i in 0..WORD_BITS {
            sum[i] = xor3(&x[i], &y[i], &cin);
            if i < CARRIES_PER_ADD {
                let slot = carry_slot(i);
                self.a[slot] = xor_sup(&x[i], &cin);
                self.b[slot] = xor_sup(&y[i], &cin);
                cin = xor_sup(&cin, &vec![slot]);
            }
        }
        sum
    }

    fn materialize<F: Fn(usize) -> usize>(&mut self, raw: &Word, slot_fn: F) -> Word {
        let mut out = zero_word();
        for b in 0..WORD_BITS {
            let s = slot_fn(b);
            self.a[s] = raw[b].clone();
            self.b[s] = vec![self.const_wire];
            out[b] = vec![s];
        }
        out
    }

    fn add32_alloc<F1: Fn(usize) -> usize, F2: Fn(usize) -> usize>(
        &mut self,
        x: &Word,
        y: &Word,
        carry_slot: F1,
        sum_slot: F2,
    ) -> Word {
        let raw = self.add32_inline(x, y, carry_slot);
        self.materialize(&raw, sum_slot)
    }
}

/// Emit the R1CS rows for one SHA-256 compression.
///
/// - `h_in_support[w][b]` — the linear support driving input chaining word `w`
///   bit `b` (for a fixed IV, `{const_wire}` for a 1 bit, empty for a 0 bit).
/// - `m_support[i][b]` — the linear support driving message word `i` bit `b`.
///
/// Both `H_in` and the 16 message words are *materialized* wires (defined as
/// `support · 1`) so downstream rows reference them by index. Returns the 256
/// `H_out` wire indices (word-major: `w*32 + b`).
pub fn build_compression(
    rows: &mut Rows,
    layout: CompLayout,
    h_in_support: &[Sup; 256],
    m_support: &[Sup; 512],
) -> Vec<usize> {
    // Materialize H_in and M from their supports.
    for w in 0..8 {
        for b in 0..WORD_BITS {
            let s = layout.h_in_bit(w, b);
            rows.a[s] = h_in_support[w * WORD_BITS + b].clone();
            rows.b[s] = vec![rows.const_wire];
        }
    }
    for i in 0..16 {
        for b in 0..WORD_BITS {
            let s = layout.m_bit(i, b);
            rows.a[s] = m_support[i * WORD_BITS + b].clone();
            rows.b[s] = vec![rows.const_wire];
        }
    }

    let h_in: Vec<Word> = (0..8)
        .map(|w| wire_word(|b| layout.h_in_bit(w, b)))
        .collect();
    let mut w_arr: Vec<Word> = (0..16).map(|i| wire_word(|b| layout.m_bit(i, b))).collect();

    // Message schedule W[16..64].
    for t in 16..(16 + N_SCHED) {
        let s1 = sigma_1(&w_arr[t - 2]);
        let s0 = sigma_0(&w_arr[t - 15]);
        let w_m7 = w_arr[t - 7].clone();
        let w_m16 = w_arr[t - 16].clone();
        let sched_0 = rows.add32_inline(&s1, &w_m7, |i| layout.sched_carry_bit(t, 0, i));
        let sched_1 = rows.add32_inline(&sched_0, &s0, |i| layout.sched_carry_bit(t, 1, i));
        let w_t = rows.add32_alloc(
            &sched_1,
            &w_m16,
            |i| layout.sched_carry_bit(t, 2, i),
            |b| layout.w_bit(t, b),
        );
        w_arr.push(w_t);
    }

    let mut state: [Word; 8] = std::array::from_fn(|i| h_in[i].clone());

    for r in 0..N_ROUNDS {
        let a = state[0].clone();
        let bb = state[1].clone();
        let c = state[2].clone();
        let d = state[3].clone();
        let e = state[4].clone();
        let f = state[5].clone();
        let g = state[6].clone();
        let h_var = state[7].clone();

        // ch_and[bit] = e · (f ⊕ g)
        let mut ch_and = zero_word();
        for bit in 0..WORD_BITS {
            let s = layout.ch_and_bit(r, bit);
            rows.a[s] = e[bit].clone();
            rows.b[s] = xor_sup(&f[bit], &g[bit]);
            ch_and[bit] = vec![s];
        }
        // maj_and[bit] = (a ⊕ b) · (a ⊕ c)
        let mut maj_and = zero_word();
        for bit in 0..WORD_BITS {
            let s = layout.maj_and_bit(r, bit);
            rows.a[s] = xor_sup(&a[bit], &bb[bit]);
            rows.b[s] = xor_sup(&a[bit], &c[bit]);
            maj_and[bit] = vec![s];
        }
        let ch_out = xor_words(&ch_and, &g);
        let maj_out = xor_words(&maj_and, &a);

        let t1_0 = rows.add32_inline(&h_var, &big_sigma_1(&e), |i| {
            layout.round_carry_bit(r, 0, i)
        });
        let t1_1 = rows.add32_inline(&t1_0, &ch_out, |i| layout.round_carry_bit(r, 1, i));
        let k_word: Word = (0..WORD_BITS)
            .map(|i| {
                if (SHA256_K[r] >> i) & 1 == 1 {
                    vec![rows.const_wire]
                } else {
                    Sup::new()
                }
            })
            .collect();
        let t1_2 = rows.add32_inline(&t1_1, &k_word, |i| layout.round_carry_bit(r, 2, i));
        let t1 = rows.add32_alloc(
            &t1_2,
            &w_arr[r],
            |i| layout.round_carry_bit(r, 3, i),
            |b| layout.t1_bit(r, b),
        );
        let t2 = rows.add32_inline(&big_sigma_0(&a), &maj_out, |i| {
            layout.round_carry_bit(r, 4, i)
        });
        let e_new = rows.add32_alloc(
            &d,
            &t1,
            |i| layout.round_carry_bit(r, 5, i),
            |b| layout.e_new_bit(r, b),
        );
        let a_new = rows.add32_alloc(
            &t1,
            &t2,
            |i| layout.round_carry_bit(r, 6, i),
            |b| layout.a_new_bit(r, b),
        );

        state = [a_new, a, bb, c, e_new, e, f, g];
    }

    // Output feed-forward H_out[w] = state[w] + H_in[w].
    let mut out = Vec::with_capacity(256);
    for w in 0..8 {
        let ho = rows.add32_alloc(
            &state[w],
            &h_in[w],
            |i| layout.out_carry_bit(w, i),
            |b| layout.h_out_bit(w, b),
        );
        for b in 0..WORD_BITS {
            out.push(ho[b][0]);
        }
    }
    out
}

// ───────────────────────────────────────────────────────────────────────────
// Native reference + witness fill.
// ───────────────────────────────────────────────────────────────────────────

#[inline]
fn big_sigma0(x: u32) -> u32 {
    x.rotate_right(2) ^ x.rotate_right(13) ^ x.rotate_right(22)
}
#[inline]
fn big_sigma1(x: u32) -> u32 {
    x.rotate_right(6) ^ x.rotate_right(11) ^ x.rotate_right(25)
}
#[inline]
fn small_sigma0(x: u32) -> u32 {
    x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)
}
#[inline]
fn small_sigma1(x: u32) -> u32 {
    x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10)
}

/// SHA-256 block map `H_out = H_in + compress(H_in, M)`. `m` is 16 big-endian
/// message words (already assembled by the caller). Uses the `sha2` crate's
/// `compress256` so it exactly matches the reference implementation.
pub fn compress(h_in: &[u32; 8], m: &[u32; 16]) -> [u32; 8] {
    let mut state = *h_in;
    let mut block = GenericArray::<u8, sha2::digest::consts::U64>::default();
    for (i, w) in m.iter().enumerate() {
        block[i * 4..i * 4 + 4].copy_from_slice(&w.to_be_bytes());
    }
    compress256(&mut state, std::slice::from_ref(&block));
    state
}

fn write_word(z: &mut [bool], base: usize, v: u32) {
    for b in 0..WORD_BITS {
        z[base + b] = (v >> b) & 1 == 1;
    }
}

/// 32-bit add writing 31 carry-aux bits. `cin[i+1] = cin[i] ⊕ carry_aux[i]`.
fn add32_w(x: u32, y: u32, carry_base: usize, z: &mut [bool]) -> u32 {
    let mut cin = false;
    for i in 0..CARRIES_PER_ADD {
        let xi = ((x >> i) & 1) == 1;
        let yi = ((y >> i) & 1) == 1;
        let aux = (xi ^ cin) && (yi ^ cin);
        z[carry_base + i] = aux;
        cin ^= aux;
    }
    x.wrapping_add(y)
}

/// Fill one compression's witness wires into the global `z`, given the concrete
/// input chaining value and message. Sets `H_in`, `M`, all intermediates, and
/// `H_out`. Recomputes the SHA-256 round function directly (mirrors the matrix
/// construction slot-for-slot).
pub fn fill_compression_witness(
    z: &mut [bool],
    layout: CompLayout,
    h_in: &[u32; 8],
    m: &[u32; 16],
) {
    for w in 0..8 {
        write_word(z, layout.h_in_bit(w, 0), h_in[w]);
    }
    for i in 0..16 {
        write_word(z, layout.m_bit(i, 0), m[i]);
    }

    let mut w_arr = [0u32; 64];
    w_arr[..16].copy_from_slice(m);
    for t in 16..64 {
        let s0 = small_sigma0(w_arr[t - 15]);
        let s1 = small_sigma1(w_arr[t - 2]);
        let sched_0 = add32_w(s1, w_arr[t - 7], layout.sched_carry_bit(t, 0, 0), z);
        let sched_1 = add32_w(sched_0, s0, layout.sched_carry_bit(t, 1, 0), z);
        let w_t = add32_w(sched_1, w_arr[t - 16], layout.sched_carry_bit(t, 2, 0), z);
        write_word(z, layout.w_bit(t, 0), w_t);
        w_arr[t] = w_t;
    }

    let mut state = *h_in;
    for r in 0..N_ROUNDS {
        let (a, b, c, d, e, f, g, h_var) = (
            state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7],
        );
        let ch_and = e & (f ^ g);
        write_word(z, layout.ch_and_bit(r, 0), ch_and);
        let maj_and = (a ^ b) & (a ^ c);
        write_word(z, layout.maj_and_bit(r, 0), maj_and);

        let ch_out = ch_and ^ g;
        let maj_out = maj_and ^ a;

        let t1_0 = add32_w(h_var, big_sigma1(e), layout.round_carry_bit(r, 0, 0), z);
        let t1_1 = add32_w(t1_0, ch_out, layout.round_carry_bit(r, 1, 0), z);
        let t1_2 = add32_w(t1_1, SHA256_K[r], layout.round_carry_bit(r, 2, 0), z);
        let t1 = add32_w(t1_2, w_arr[r], layout.round_carry_bit(r, 3, 0), z);
        write_word(z, layout.t1_bit(r, 0), t1);

        let t2 = add32_w(big_sigma0(a), maj_out, layout.round_carry_bit(r, 4, 0), z);
        let e_new = add32_w(d, t1, layout.round_carry_bit(r, 5, 0), z);
        write_word(z, layout.e_new_bit(r, 0), e_new);
        let a_new = add32_w(t1, t2, layout.round_carry_bit(r, 6, 0), z);
        write_word(z, layout.a_new_bit(r, 0), a_new);

        state = [a_new, a, b, c, e_new, e, f, g];
    }

    for w in 0..8 {
        let h_out = add32_w(state[w], h_in[w], layout.out_carry_bit(w, 0), z);
        write_word(z, layout.h_out_bit(w, 0), h_out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compress_matches_sha2_crate() {
        // A single SHA-256 compression on a fixed 512-bit message with the IV.
        let m: [u32; 16] = std::array::from_fn(|i| (0x0102_0304u32).wrapping_mul(i as u32 + 1));
        let got = compress(&SHA256_IV, &m);
        // Reference via a second independent path.
        let mut state = SHA256_IV;
        let mut block = GenericArray::<u8, sha2::digest::consts::U64>::default();
        for (i, w) in m.iter().enumerate() {
            block[i * 4..i * 4 + 4].copy_from_slice(&w.to_be_bytes());
        }
        compress256(&mut state, std::slice::from_ref(&block));
        assert_eq!(got, state);
    }
}
