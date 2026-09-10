//! Native (non-proving) circuit-satisfaction tests: fast to run and pinpoint
//! circuit/witness bugs without paying for the full PCS prove.

use pq_tornado::{
    circuit::{TornadoCircuit, Witness},
    note::{MerkleTree, Note},
};

fn sample_note(seed: u32) -> Note {
    Note {
        nullifier: std::array::from_fn(|i| seed.wrapping_mul(0x9E37_79B1).wrapping_add(i as u32)),
        secret: std::array::from_fn(|i| {
            seed.wrapping_mul(0x85EB_CA77)
                .wrapping_add(0x1000 + i as u32)
        }),
    }
}

fn build_scenario(depth: usize, leaf_index: usize) -> (TornadoCircuit, Witness) {
    let n = 1usize << depth;
    let notes: Vec<Note> = (0..n).map(|i| sample_note(i as u32 + 1)).collect();
    let leaves: Vec<_> = notes.iter().map(|n| n.commitment()).collect();
    let tree = MerkleTree::new(depth, &leaves);
    let (siblings, bits) = tree.path(leaf_index);
    let circuit = TornadoCircuit::build(depth);
    let w = Witness {
        note: notes[leaf_index],
        siblings,
        bits,
    };
    (circuit, w)
}

#[test]
fn circuit_satisfies_for_honest_witness() {
    for depth in [1usize, 4] {
        let (circuit, w) = build_scenario(depth, if depth == 1 { 1 } else { 5 });
        let block = circuit.generate_block_witness(&w);
        // Tile the identical block across the 2^N_BLOCKS_LOG outer copies.
        let n_blocks = 1usize << pq_tornado::circuit::N_BLOCKS_LOG;
        let mut z = Vec::with_capacity(block.len() * n_blocks);
        for _ in 0..n_blocks {
            z.extend_from_slice(&block);
        }
        assert_eq!(z.len(), circuit.r1cs.n());
        assert!(
            circuit.r1cs.satisfies(&z),
            "R1CS not satisfied at depth {depth}"
        );

        // The public outputs the circuit exposes must match the native tree.
        let pi = circuit.public_inputs(&w);
        let tree_root = {
            // recompute from the same leaf/path
            pq_tornado::note::recompute_root(&w.note.commitment(), &w.siblings, &w.bits)
        };
        assert_eq!(pi.root, tree_root);
        assert_eq!(pi.nullifier_hash, w.note.nullifier_hash());
    }
}

#[test]
fn tampered_witness_breaks_satisfaction() {
    let (circuit, w) = build_scenario(4, 3);
    let mut block = circuit.generate_block_witness(&w);
    // Flip a sibling bit — the recomputed root no longer matches, but more
    // importantly the level hash relation is violated at the mux input.
    // Corrupt the stored `t` for level 0 (index into the witness): flipping any
    // non-free intermediate must break a constraint.
    // Find first `true` bit past the constant and flip it.
    let pos = block.iter().position(|&b| b).unwrap();
    block[pos] = !block[pos];
    let n_blocks = 1usize << pq_tornado::circuit::N_BLOCKS_LOG;
    let mut z = Vec::with_capacity(block.len() * n_blocks);
    for _ in 0..n_blocks {
        z.extend_from_slice(&block);
    }
    assert!(!circuit.r1cs.satisfies(&z), "tamper should break R1CS");
}
