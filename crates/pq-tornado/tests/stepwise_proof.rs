//! End-to-end tests for the stepwise (small-circuit + wiring) pipeline.

use pq_tornado::circuit::{PublicInputs, Witness};
use pq_tornado::note::{DOMAIN_NF, MerkleTree, Note};
use pq_tornado::step::StepInput;
use pq_tornado::stepwise::{self, StepwiseCircuit};

fn sample_note(seed: u32) -> Note {
    Note {
        nullifier: std::array::from_fn(|i| seed.wrapping_mul(0x9E37_79B1).wrapping_add(i as u32)),
        secret: std::array::from_fn(|i| {
            seed.wrapping_mul(0x85EB_CA77).wrapping_add(0x1000 + i as u32)
        }),
    }
}

fn tree(depth: usize) -> (Vec<Note>, MerkleTree) {
    let notes: Vec<Note> = (0..1usize << depth)
        .map(|i| sample_note(i as u32 + 1))
        .collect();
    let leaves: Vec<_> = notes.iter().map(|n| n.commitment()).collect();
    let t = MerkleTree::new(depth, &leaves);
    (notes, t)
}

fn scenario(depth: usize, idx: usize) -> (StepwiseCircuit, Witness) {
    let (notes, t) = tree(depth);
    let (siblings, bits) = t.path(idx);
    (
        StepwiseCircuit::build(depth),
        Witness {
            note: notes[idx],
            siblings,
            bits,
        },
    )
}

#[test]
fn r1cs_is_satisfied_and_wiring_holds_natively() {
    let (circuit, w) = scenario(4, 9);
    let inputs = circuit.step_inputs(&w);
    let z = circuit.step.generate_witness(&inputs);
    assert!(
        circuit.step.r1cs.satisfies(&z),
        "step R1CS must be satisfied"
    );

    // Every wiring constraint must hold on the honest witness.
    let public = circuit.public_inputs(&w);
    for c in circuit.constraints(&public) {
        for i in 0..256 {
            let lhs = c.slots.iter().fold(false, |acc, &s| acc ^ z[s * 256 + i]);
            let rhs = (c.rhs[i / 32] >> (i % 32)) & 1 == 1;
            assert_eq!(lhs, rhs, "constraint {c:?} bit {i}");
        }
    }
}

#[test]
fn honest_roundtrip() {
    let depth = 4;
    let idx = 5;
    let (circuit, w) = scenario(depth, idx);
    let ctx = b"recipient=0xABCD;relayer=0x1234;fee=10";

    let (proof, public) = stepwise::prove(&circuit, &w, ctx);

    let (_, t) = tree(depth);
    assert_eq!(public.root, t.root());
    assert_eq!(public.nullifier_hash, w.note.nullifier_hash());

    stepwise::verify(&circuit, &proof, &public, ctx).expect("honest proof must verify");
}

#[test]
fn wrong_root_rejected() {
    let (circuit, w) = scenario(4, 2);
    let ctx = b"ctx";
    let (proof, mut public) = stepwise::prove(&circuit, &w, ctx);
    public.root[0] ^= 1;
    assert!(stepwise::verify(&circuit, &proof, &public, ctx).is_err());
}

#[test]
fn wrong_nullifier_hash_rejected() {
    let (circuit, w) = scenario(4, 2);
    let ctx = b"ctx";
    let (proof, mut public) = stepwise::prove(&circuit, &w, ctx);
    public.nullifier_hash[3] ^= 0x8000;
    assert!(stepwise::verify(&circuit, &proof, &public, ctx).is_err());
}

#[test]
fn context_binding_rejected() {
    let (circuit, w) = scenario(4, 2);
    let (proof, public) = stepwise::prove(&circuit, &w, b"recipient=A");
    assert!(stepwise::verify(&circuit, &proof, &public, b"recipient=B").is_err());
}

#[test]
fn membership_against_foreign_root_rejected() {
    let (circuit, w) = scenario(4, 1);
    let ctx = b"ctx";
    let (proof, mut public) = stepwise::prove(&circuit, &w, ctx);
    public.root = {
        let notes: Vec<Note> = (100..116u32).map(sample_note).collect();
        let leaves: Vec<_> = notes.iter().map(|n| n.commitment()).collect();
        MerkleTree::new(4, &leaves).root()
    };
    assert!(stepwise::verify(&circuit, &proof, &public, ctx).is_err());
}

/// A prover who breaks a *link* between two compressions (rather than the
/// public outputs) must also be rejected: here the Merkle path is recomputed
/// from a foreign leaf while the commitment hash still opens the real note.
#[test]
fn forged_chain_link_rejected() {
    let depth = 4;
    let (notes, t) = tree(depth);
    let ctx = b"ctx";

    // Honest path for leaf 3 ...
    let (siblings, bits) = t.path(3);
    let w = Witness {
        note: notes[9], // ... but the note we can actually open is a different one
        siblings,
        bits,
    };
    let circuit = StepwiseCircuit::build(depth);
    let mut inputs = circuit.step_inputs(&w);
    // Splice: make the Merkle path start from leaf 3's commitment even though
    // instance 0 hashes note 9. This breaks `node(level 0) = H_out(cm)`.
    let mut node = notes[3].commitment();
    for d in 0..depth {
        inputs[2 + d].node = node;
        node = inputs[2 + d].output();
    }

    let public = PublicInputs {
        root: t.root(),
        nullifier_hash: notes[9].nullifier_hash(),
    };
    // The R1CS itself is satisfied (every instance is an honest compression) and
    // both public outputs are correct — only the `node(level 0) = H_out(cm)`
    // link is broken, so the rejection must come from the wiring layer.
    let z = circuit.step.generate_witness(&inputs);
    assert!(circuit.step.r1cs.satisfies(&z));
    let proof = stepwise::prove_raw(&circuit, &inputs, &public, ctx);
    assert!(matches!(
        stepwise::verify(&circuit, &proof, &public, ctx),
        Err(stepwise::VerifyError::Wiring(_))
    ));
}

/// A prover who swaps the mux on the commitment hash (spending `SHA256_c(k‖r)`
/// under nullifier `r` instead of `k`) must be rejected — that is what the
/// `t = 0` pins are for.
#[test]
fn swapped_commitment_mux_rejected() {
    let depth = 4;
    let (notes, t) = tree(depth);
    let (siblings, bits) = t.path(6);
    let note = notes[6];
    let ctx = b"ctx";

    let circuit = StepwiseCircuit::build(depth);
    // Spend the same commitment but declare `secret` as the nullifier: the
    // commitment instance uses bit = 1 so that H(sib‖node) = H(k‖r) still holds.
    let mut inputs = vec![
        StepInput {
            node: note.secret,
            sib: note.nullifier,
            bit: true, // mux swap → left = k, right = r  ⇒ same cm
        },
        StepInput {
            node: note.secret,
            sib: DOMAIN_NF,
            bit: false,
        },
    ];
    let mut node = note.commitment();
    for d in 0..depth {
        let inp = StepInput {
            node,
            sib: siblings[d],
            bit: bits[d],
        };
        node = inp.output();
        inputs.push(inp);
    }
    assert_eq!(inputs[0].output(), note.commitment(), "cm is unchanged");

    let public = PublicInputs {
        root: t.root(),
        nullifier_hash: pq_tornado::note::nullifier_hash(&note.secret),
    };
    let z = circuit.step.generate_witness(&inputs);
    assert!(circuit.step.r1cs.satisfies(&z));
    let proof = stepwise::prove_raw(&circuit, &inputs, &public, ctx);
    assert!(
        matches!(
            stepwise::verify(&circuit, &proof, &public, ctx),
            Err(stepwise::VerifyError::Wiring(_))
        ),
        "mux swap must be blocked by the t = 0 pin"
    );
}
