//! The post-quantum Tornado Cash **withdrawal circuit**, as one monolithic
//! R1CS-over-GF(2) constraint block (circuit shape `(A·z) ⊙ (B·z) = z`).
//!
//! The circuit proves, in zero knowledge over the private witness
//! `(nullifier k, secret r, siblings[D], path-bits[D])`:
//!
//! 1. **Commitment opening.** `cm = SHA256_c(k ‖ r)`.
//! 2. **Nullifier hash.** `nh = SHA256_c(k ‖ DOMAIN_NF)` — later revealed and
//!    checked against the public nullifier hash (double-spend tag). `k` is the
//!    *same* wires as in (1), so the revealed `nh` is bound to the spent note.
//! 3. **Merkle membership.** Starting from `node₀ = cm`, for each level `d`
//!    a private path bit selects the left/right ordering of `(node_d, sib_d)`
//!    (an in-circuit multiplexer) and `node_{d+1} = SHA256_c(left ‖ right)`.
//!    The final `node_D` is revealed and checked against the public root.
//!
//! Every hash is one SHA-256 compression (see [`crate::sha256`]); all
//! intermediate values (`cm`, the internal nodes, the mux outputs) are private
//! wires wired together *inside* the single block, so nothing about *which*
//! leaf is spent leaks from the constraint system. Only the public root and
//! nullifier hash are exposed — via PCS openings in [`crate::pipeline`].
//!
//! The input chaining value of every compression is pinned to the SHA-256 IV
//! by construction (a constant support), which is what forces each hash to be
//! the honest fixed-IV compression rather than a prover-chosen map.

use flock_prover::r1cs::{BlockR1cs, SparseBinaryMatrix, WitnessLayout};

use crate::{
    note::{DOMAIN_NF, H256, Note, recompute_root},
    sha256::{
        COMP_STRIDE, CompLayout, Rows, SHA256_IV, Sup, build_compression, fill_compression_witness,
    },
};

/// Number of outer (block-diagonal) copies. Flock's lincheck requires
/// `n_outer ≥ 8`; the whole withdrawal statement is one block, so we tile the
/// identical block `2^N_BLOCKS_LOG` times.
pub const N_BLOCKS_LOG: usize = 3;
/// Univariate-skip dimension (matches Flock's hash encoders).
pub const K_SKIP: usize = 6;

/// Private withdrawal witness.
#[derive(Clone, Debug)]
pub struct Witness {
    pub note: Note,
    pub siblings: Vec<H256>,
    pub bits: Vec<bool>,
}

/// Public statement of a withdrawal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PublicInputs {
    pub root: H256,
    pub nullifier_hash: H256,
}

/// A built Tornado circuit: the R1CS instance plus the wire bookkeeping needed
/// to generate witnesses and to locate the public-output regions for opening.
pub struct TornadoCircuit {
    pub depth: usize,
    pub k_log: usize,
    pub r1cs: BlockR1cs,
    const_wire: usize,
    k_wires: Vec<usize>,
    r_wires: Vec<usize>,
    sib_wires: Vec<Vec<usize>>,
    bit_wires: Vec<usize>,
    t_wires: Vec<Vec<usize>>,
    comp_bases: Vec<usize>,
    /// Compression indices `[commitment, nullifier, level_0, .., level_{D-1}]`.
    nf_idx: usize,
    root_idx: usize,
}

/// Alloc a run of `n` consecutive wire indices.
fn alloc(next: &mut usize, n: usize) -> Vec<usize> {
    let v = (*next..*next + n).collect();
    *next += n;
    v
}

/// The support for a fixed 256-bit constant word-array: bit set → `{const}`,
/// clear → empty.
fn const_h256_support(v: &H256, const_wire: usize) -> Vec<Sup> {
    (0..256)
        .map(|i| {
            let (w, b) = (i / 32, i % 32);
            if (v[w] >> b) & 1 == 1 {
                vec![const_wire]
            } else {
                Sup::new()
            }
        })
        .collect()
}

impl TornadoCircuit {
    /// Build the withdrawal circuit for a Merkle tree of the given `depth`.
    pub fn build(depth: usize) -> Self {
        assert!(depth >= 1, "tree depth must be ≥ 1");
        let n_comp = 2 + depth; // commitment, nullifier, D levels
        let nf_idx = 1;
        let root_idx = 2 + depth - 1;

        // ---- Wire allocation.
        let mut next = 0usize;
        let const_wire = alloc(&mut next, 1)[0];
        let k_wires = alloc(&mut next, 256);
        let r_wires = alloc(&mut next, 256);
        let sib_wires: Vec<Vec<usize>> = (0..depth).map(|_| alloc(&mut next, 256)).collect();
        let bit_wires: Vec<usize> = (0..depth).map(|_| alloc(&mut next, 1)[0]).collect();
        let t_wires: Vec<Vec<usize>> = (0..depth).map(|_| alloc(&mut next, 256)).collect();

        // Compression regions must be 256-bit aligned so each H_out region
        // starts on an *even* 128-bit word (a clean 2-word PCS-openable pair).
        next = next.next_multiple_of(256);
        let comp_region_start = next;
        let comp_bases: Vec<usize> = (0..n_comp)
            .map(|c| comp_region_start + c * COMP_STRIDE)
            .collect();
        next = comp_region_start + n_comp * COMP_STRIDE;

        let useful_bits = next;
        // Enough for the witness, and large enough that m = k_log + N_BLOCKS_LOG
        // lands in the range of Flock's embedded Ligerito configs (m ≥ 22).
        let min_k_log = 22 - N_BLOCKS_LOG;
        let k_log = usize::max(
            min_k_log,
            useful_bits.next_power_of_two().trailing_zeros() as usize,
        );
        let k = 1usize << k_log;
        assert!(useful_bits <= k);

        // ---- Matrix rows.
        let mut a: Vec<Sup> = vec![Sup::new(); k];
        let mut b: Vec<Sup> = vec![Sup::new(); k];

        // Constant-1 wire: z[const]·z[const] = z[const] pins it to 1.
        a[const_wire] = vec![const_wire];
        b[const_wire] = vec![const_wire];

        // Free-input wires: z = z (tautology `z·1 = z`).
        for &w in k_wires
            .iter()
            .chain(r_wires.iter())
            .chain(sib_wires.iter().flatten())
            .chain(bit_wires.iter())
        {
            a[w] = vec![w];
            b[w] = vec![const_wire];
        }

        let iv_support: Vec<Sup> = {
            let words: H256 = SHA256_IV;
            const_h256_support(&words, const_wire)
        };
        let iv_support: [Sup; 256] = iv_support.try_into().unwrap();

        let mut rows = Rows {
            a: &mut a,
            b: &mut b,
            const_wire,
        };

        // (0) commitment: cm = H(k ‖ r).
        let comm_layout = CompLayout {
            base: comp_bases[0],
        };
        let comm_m: [Sup; 512] = {
            let mut m: Vec<Sup> = Vec::with_capacity(512);
            for i in 0..256 {
                m.push(vec![k_wires[i]]);
            }
            for i in 0..256 {
                m.push(vec![r_wires[i]]);
            }
            m.try_into().unwrap()
        };
        let cm_hout = build_compression(&mut rows, comm_layout, &iv_support, &comm_m);

        // (1) nullifier hash: nh = H(k ‖ DOMAIN_NF).
        let nf_layout = CompLayout {
            base: comp_bases[nf_idx],
        };
        let dom_support = const_h256_support(&DOMAIN_NF, const_wire);
        let nf_m: [Sup; 512] = {
            let mut m: Vec<Sup> = Vec::with_capacity(512);
            for i in 0..256 {
                m.push(vec![k_wires[i]]);
            }
            m.extend(dom_support);
            m.try_into().unwrap()
        };
        let _nf_hout = build_compression(&mut rows, nf_layout, &iv_support, &nf_m);

        // (2..) Merkle levels.
        let mut node = cm_hout; // node_0 = cm
        for d in 0..depth {
            let layout = CompLayout {
                base: comp_bases[2 + d],
            };
            // mux: t[i] = bit_d · (sib_i ⊕ node_i)  (AND row)
            for i in 0..256 {
                let t = t_wires[d][i];
                rows.a[t] = vec![bit_wires[d]];
                rows.b[t] = {
                    let mut s = vec![sib_wires[d][i], node[i]];
                    s.sort_unstable();
                    s.dedup();
                    s
                };
            }
            // left_i = node_i ⊕ t_i ; right_i = sib_i ⊕ t_i
            let m: [Sup; 512] = {
                let mut m: Vec<Sup> = Vec::with_capacity(512);
                for i in 0..256 {
                    let mut s = vec![node[i], t_wires[d][i]];
                    s.sort_unstable();
                    m.push(s);
                }
                for i in 0..256 {
                    let mut s = vec![sib_wires[d][i], t_wires[d][i]];
                    s.sort_unstable();
                    m.push(s);
                }
                m.try_into().unwrap()
            };
            node = build_compression(&mut rows, layout, &iv_support, &m);
        }

        let c_0 = SparseBinaryMatrix {
            num_rows: k,
            num_cols: k,
            rows: (0..k).map(|i| vec![i]).collect(),
        };
        let a_0 = SparseBinaryMatrix {
            num_rows: k,
            num_cols: k,
            rows: a,
        };
        let b_0 = SparseBinaryMatrix {
            num_rows: k,
            num_cols: k,
            rows: b,
        };

        let r1cs = BlockR1cs {
            m: k_log + N_BLOCKS_LOG,
            k_log,
            k_skip: K_SKIP,
            useful_bits,
            a_0,
            b_0,
            c_0,
            layout: WitnessLayout::RowMajor,
            const_pin: Some(const_wire),
            digest_cache: std::sync::OnceLock::new(),
            csc_cache: std::sync::OnceLock::new(),
        };

        // Warm the CSC transpose of (A_0, B_0) and the prover scratch pool, so
        // neither shows up on the prove/verify path. `CscCircuit` gathers per
        // column instead of scattering per row and is dramatically faster than
        // `SparseMatrixCircuit` at this block size.
        r1cs.csc_lincheck_circuit();
        flock_prover::scratch::prewarm_prover(r1cs.m);

        Self {
            depth,
            k_log,
            r1cs,
            const_wire,
            k_wires,
            r_wires,
            sib_wires,
            bit_wires,
            t_wires,
            comp_bases,
            nf_idx,
            root_idx,
        }
    }

    /// First wire (128-bit-aligned) of the revealed **root** region.
    pub fn root_hout_base(&self) -> usize {
        CompLayout {
            base: self.comp_bases[self.root_idx],
        }
        .h_out_base()
    }
    /// First wire (128-bit-aligned) of the revealed **nullifier-hash** region.
    pub fn nf_hout_base(&self) -> usize {
        CompLayout {
            base: self.comp_bases[self.nf_idx],
        }
        .h_out_base()
    }

    /// Compute the public outputs for a witness (native, no proving).
    pub fn public_inputs(&self, w: &Witness) -> PublicInputs {
        assert_eq!(w.siblings.len(), self.depth);
        assert_eq!(w.bits.len(), self.depth);
        let cm = w.note.commitment();
        PublicInputs {
            root: recompute_root(&cm, &w.siblings, &w.bits),
            nullifier_hash: w.note.nullifier_hash(),
        }
    }

    /// Generate the boolean witness for a single block (length `2^k_log`).
    pub fn generate_block_witness(&self, w: &Witness) -> Vec<bool> {
        assert_eq!(w.siblings.len(), self.depth);
        assert_eq!(w.bits.len(), self.depth);
        let k = 1usize << self.k_log;
        let mut z = vec![false; k];
        z[self.const_wire] = true;

        write_h256(&mut z, &self.k_wires, &w.note.nullifier);
        write_h256(&mut z, &self.r_wires, &w.note.secret);
        for d in 0..self.depth {
            write_h256(&mut z, &self.sib_wires[d], &w.siblings[d]);
            z[self.bit_wires[d]] = w.bits[d];
        }

        // Native values.
        let cm = w.note.commitment();
        let mut node = cm;
        // t and mux per level, plus compression witnesses.
        // commitment compression:
        {
            let mut m = [0u32; 16];
            m[..8].copy_from_slice(&w.note.nullifier);
            m[8..].copy_from_slice(&w.note.secret);
            fill_compression_witness(
                &mut z,
                CompLayout {
                    base: self.comp_bases[0],
                },
                &SHA256_IV,
                &m,
            );
        }
        // nullifier compression:
        {
            let mut m = [0u32; 16];
            m[..8].copy_from_slice(&w.note.nullifier);
            m[8..].copy_from_slice(&DOMAIN_NF);
            fill_compression_witness(
                &mut z,
                CompLayout {
                    base: self.comp_bases[self.nf_idx],
                },
                &SHA256_IV,
                &m,
            );
        }
        // Merkle levels:
        for d in 0..self.depth {
            // t[i] = bit & (sib_i ^ node_i)
            for i in 0..256 {
                let (wi, bi) = (i / 32, i % 32);
                let sib_bit = (w.siblings[d][wi] >> bi) & 1 == 1;
                let node_bit = (node[wi] >> bi) & 1 == 1;
                z[self.t_wires[d][i]] = w.bits[d] && (sib_bit ^ node_bit);
            }
            let (left, right) = if w.bits[d] {
                (w.siblings[d], node)
            } else {
                (node, w.siblings[d])
            };
            let mut m = [0u32; 16];
            m[..8].copy_from_slice(&left);
            m[8..].copy_from_slice(&right);
            fill_compression_witness(
                &mut z,
                CompLayout {
                    base: self.comp_bases[2 + d],
                },
                &SHA256_IV,
                &m,
            );
            node = crate::note::hash_pair(&left, &right);
        }
        z
    }
}

fn write_h256(z: &mut [bool], wires: &[usize], v: &H256) {
    for i in 0..256 {
        let (w, b) = (i / 32, i % 32);
        z[wires[i]] = (v[w] >> b) & 1 == 1;
    }
}
