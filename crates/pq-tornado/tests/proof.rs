//! End-to-end prove/verify pipeline tests.

use pq_tornado::{
    circuit::{PublicInputs, TornadoCircuit, Witness},
    note::{MerkleTree, Note},
    pipeline,
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

fn scenario(depth: usize, idx: usize) -> (TornadoCircuit, Witness) {
    let n = 1usize << depth;
    let notes: Vec<Note> = (0..n).map(|i| sample_note(i as u32 + 1)).collect();
    let leaves: Vec<_> = notes.iter().map(|n| n.commitment()).collect();
    let tree = MerkleTree::new(depth, &leaves);
    let (siblings, bits) = tree.path(idx);
    (
        TornadoCircuit::build(depth),
        Witness {
            note: notes[idx],
            siblings,
            bits,
        },
    )
}

#[test]
fn honest_withdrawal_roundtrip() {
    let depth = 4;
    let (circuit, w) = scenario(depth, 5);
    let ctx = b"recipient=0xABCD;relayer=0x1234;fee=10";

    let (proof, pi) = pipeline::prove(&circuit, &w, ctx);

    // Public inputs must equal the real tree root / nullifier hash.
    let n = 1usize << depth;
    let notes: Vec<Note> = (0..n).map(|i| sample_note(i as u32 + 1)).collect();
    let leaves: Vec<_> = notes.iter().map(|nt| nt.commitment()).collect();
    let tree = MerkleTree::new(depth, &leaves);
    assert_eq!(pi.root, tree.root());
    assert_eq!(pi.nullifier_hash, w.note.nullifier_hash());

    pipeline::verify(&circuit, &proof, &pi, ctx).expect("honest proof must verify");
}

#[test]
fn wrong_root_rejected() {
    let (circuit, w) = scenario(4, 2);
    let ctx = b"ctx";
    let (proof, mut pi) = pipeline::prove(&circuit, &w, ctx);
    pi.root[0] ^= 1;
    assert!(pipeline::verify(&circuit, &proof, &pi, ctx).is_err());
}

#[test]
fn wrong_nullifier_hash_rejected() {
    let (circuit, w) = scenario(4, 2);
    let ctx = b"ctx";
    let (proof, mut pi) = pipeline::prove(&circuit, &w, ctx);
    pi.nullifier_hash[3] ^= 0x8000;
    assert!(pipeline::verify(&circuit, &proof, &pi, ctx).is_err());
}

#[test]
fn context_binding_rejected() {
    let (circuit, w) = scenario(4, 2);
    let (proof, pi) = pipeline::prove(&circuit, &w, b"recipient=A");
    // A relayer front-running with a different recipient must fail.
    assert!(pipeline::verify(&circuit, &proof, &pi, b"recipient=B").is_err());
}

#[test]
fn membership_against_foreign_root_rejected() {
    // Prove membership in tree T, then verify against an unrelated tree's root.
    let (circuit, w) = scenario(4, 1);
    let (proof, mut pi) = pipeline::prove(&circuit, &w, b"ctx");
    let other: PublicInputs = {
        let notes: Vec<Note> = (100..116u32).map(sample_note).collect();
        let leaves: Vec<_> = notes.iter().map(|n| n.commitment()).collect();
        let tree = MerkleTree::new(4, &leaves);
        PublicInputs {
            root: tree.root(),
            nullifier_hash: pi.nullifier_hash,
        }
    };
    pi.root = other.root;
    assert!(pipeline::verify(&circuit, &proof, &pi, b"ctx").is_err());
}
