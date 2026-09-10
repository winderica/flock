//! Compare the two pq-tornado decompositions at equal security, one withdrawal
//! per proof:
//!
//! * **monolithic** — [`pq_tornado::circuit`] + [`pq_tornado::pipeline`]:
//!   all `D + 2` SHA-256 compressions wired inside one R1CS block.
//! * **stepwise** — [`pq_tornado::step`] + [`pq_tornado::wiring`] +
//!   [`pq_tornado::stepwise`]: `2^n_log` copies of a single-compression block,
//!   linked by the slot-wiring sumcheck (the paper's glue circuit `G`).
//!
//! ```sh
//! cargo run --release -p pq-tornado --example bench_compare -- [DEPTH] [RUNS]
//! ```

use std::time::Instant;

use pq_tornado::circuit::{TornadoCircuit, Witness};
use pq_tornado::note::{H256, Note, recompute_root};
use pq_tornado::stepwise::{self, StepwiseCircuit};

const CTX: &[u8] = b"recipient=0xABCD;relayer=0x1234;fee=10";

fn h256_from_seed(seed: &mut u64) -> H256 {
    std::array::from_fn(|_| {
        *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) as u32
    })
}

/// A withdrawal witness with a synthesized authentication path (O(D), so the
/// depth-32 case is feasible without materializing 2^32 leaves).
fn synth_witness(depth: usize, seed: &mut u64) -> Witness {
    let note = Note {
        nullifier: h256_from_seed(seed),
        secret: h256_from_seed(seed),
    };
    let siblings: Vec<H256> = (0..depth).map(|_| h256_from_seed(seed)).collect();
    let bits: Vec<bool> = (0..depth).map(|d| (*seed >> (d % 63)) & 1 == 1).collect();
    let _ = recompute_root(&note.commitment(), &siblings, &bits);
    Witness {
        note,
        siblings,
        bits,
    }
}

fn best_of(runs: usize, mut f: impl FnMut() -> f64) -> f64 {
    (0..runs).map(|_| f()).fold(f64::INFINITY, f64::min)
}

struct Row {
    name: &'static str,
    m: usize,
    k_log: usize,
    useful_frac: f64,
    build_s: f64,
    prove_s: f64,
    verify_s: f64,
    proof_kib: f64,
}

impl Row {
    fn print(&self) {
        println!(
            "| {:<12} | {:>2} | {:>2} | {:>5.1}% | {:>8.3} | {:>8.1} | {:>8.1} | {:>7.0} |",
            self.name,
            self.m,
            self.k_log,
            self.useful_frac * 100.0,
            self.build_s,
            self.prove_s * 1e3,
            self.verify_s * 1e3,
            self.proof_kib,
        );
    }
}

fn bench_monolithic(depth: usize, runs: usize) -> Row {
    let t = Instant::now();
    let circuit = TornadoCircuit::build(depth);
    let build_s = t.elapsed().as_secs_f64();

    let mut seed = 0xC0FF_EE00_1234_5678;
    let w = synth_witness(depth, &mut seed);

    // warm-up (also validates)
    let (proof, public) = pq_tornado::pipeline::prove(&circuit, &w, CTX);
    pq_tornado::pipeline::verify(&circuit, &proof, &public, CTX).expect("monolithic verify");
    let proof_kib = bincode::serialize(&proof).unwrap().len() as f64 / 1024.0;

    let prove_s = best_of(runs, || {
        let t = Instant::now();
        let _ = pq_tornado::pipeline::prove(&circuit, &w, CTX);
        t.elapsed().as_secs_f64()
    });
    let verify_s = best_of(runs, || {
        let t = Instant::now();
        pq_tornado::pipeline::verify(&circuit, &proof, &public, CTX).unwrap();
        t.elapsed().as_secs_f64()
    });

    let useful = (2 + depth) as f64 * pq_tornado::sha256::COMP_STRIDE as f64;
    Row {
        name: "monolithic",
        m: circuit.r1cs.m,
        k_log: circuit.k_log,
        useful_frac: useful / (1u64 << circuit.r1cs.m) as f64,
        build_s,
        prove_s,
        verify_s,
        proof_kib,
    }
}

fn bench_stepwise(depth: usize, runs: usize) -> Row {
    let t = Instant::now();
    let circuit = StepwiseCircuit::build(depth);
    let build_s = t.elapsed().as_secs_f64();

    let mut seed = 0xC0FF_EE00_1234_5678;
    let w = synth_witness(depth, &mut seed);

    let (proof, public) = stepwise::prove(&circuit, &w, CTX);
    stepwise::verify(&circuit, &proof, &public, CTX).expect("stepwise verify");
    let proof_kib = bincode::serialize(&proof).unwrap().len() as f64 / 1024.0;

    let prove_s = best_of(runs, || {
        let t = Instant::now();
        let _ = stepwise::prove(&circuit, &w, CTX);
        t.elapsed().as_secs_f64()
    });
    let verify_s = best_of(runs, || {
        let t = Instant::now();
        stepwise::verify(&circuit, &proof, &public, CTX).unwrap();
        t.elapsed().as_secs_f64()
    });

    let useful =
        StepwiseCircuit::n_compressions(depth) as f64 * pq_tornado::step::USEFUL_BITS as f64;
    Row {
        name: "stepwise",
        m: circuit.m(),
        k_log: pq_tornado::step::K_LOG,
        useful_frac: useful / (1u64 << circuit.m()) as f64,
        build_s,
        prove_s,
        verify_s,
        proof_kib,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let depth: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(32);
    let runs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);

    println!(
        "pq-tornado: monolithic vs stepwise   depth={depth} ({} SHA-256 compressions), best of {runs}",
        depth + 2
    );
    println!("threads: {}\n", rayon::current_num_threads());

    let header = format!(
        "| {:<12} | {:>2} | {:>2} | {:>6} | {:>8} | {:>8} | {:>8} | {:>7} |",
        "variant", "m", "kl", "useful", "build s", "prove ms", "verif ms", "proof K"
    );
    println!("{header}");
    println!("|{}|", "-".repeat(header.len() - 2));

    let mono = bench_monolithic(depth, runs);
    mono.print();
    let step = bench_stepwise(depth, runs);
    step.print();

    println!("\nstepwise speed-up over monolithic:");
    println!("  build  {:>6.1}x", mono.build_s / step.build_s);
    println!("  prove  {:>6.1}x", mono.prove_s / step.prove_s);
    println!("  verify {:>6.1}x", mono.verify_s / step.verify_s);
    println!("  proof  {:>6.2}x", mono.proof_kib / step.proof_kib);
}
