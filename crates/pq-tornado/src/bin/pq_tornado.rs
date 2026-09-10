//! `pq_tornado` — CLI for the post-quantum Tornado Cash prover/verifier.
//!
//! Subcommands:
//!   deposit  [--seed N] [--out note.json]
//!       Create a note (nullifier, secret); print its commitment (the leaf).
//!   withdraw --note note.json --depth D --index I [--context STR]
//!            [--out proof.bin] [--public public.json]
//!       Build a depth-D tree with the note at leaf I (other leaves filler),
//!       produce a withdrawal proof, and write the proof + public inputs.
//!   verify   --proof proof.bin --public public.json --depth D [--context STR]
//!       Verify a withdrawal proof.
//!   demo     [--depth D] [--index I] [--context STR]
//!       Self-contained end-to-end run (deposit → withdraw → verify) with
//!       timing and size reporting.

use std::{collections::HashMap, time::Instant};

use pq_tornado::{
    circuit::{PublicInputs, TornadoCircuit, Witness},
    note::{H256, Note},
    pipeline,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cmd = args.get(1).map(String::as_str).unwrap_or("demo");
    let opts = parse_opts(&args[2.min(args.len())..]);
    let r = match cmd {
        "deposit" => cmd_deposit(&opts),
        "withdraw" => cmd_withdraw(&opts),
        "verify" => cmd_verify(&opts),
        "demo" => cmd_demo(&opts),
        "fast-demo" => cmd_fast_demo(&opts),
        "help" | "-h" | "--help" => {
            print_help();
            Ok(())
        }
        other => Err(format!("unknown subcommand `{other}` (try `help`)")),
    };
    if let Err(e) = r {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn print_help() {
    print!(
        "pq_tornado — post-quantum Tornado Cash (Flock backend)\n\n\
         USAGE:\n\
         \x20 pq_tornado deposit  [--seed N] [--out note.json]\n\
         \x20 pq_tornado withdraw --note note.json --depth D --index I \\\n\
         \x20                     [--context STR] [--out proof.bin] [--public public.json]\n\
         \x20 pq_tornado verify   --proof proof.bin --public public.json --depth D [--context STR]\n\
         \x20 pq_tornado demo     [--depth D] [--index I] [--context STR]\n\
         \x20 pq_tornado fast-demo [--depth D] [--context STR]\n"
    );
}

// ---------------------------------------------------------------------------
// Option parsing
// ---------------------------------------------------------------------------

fn parse_opts(args: &[String]) -> HashMap<String, String> {
    let mut m = HashMap::new();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if let Some(key) = a.strip_prefix("--") {
            let val = args.get(i + 1).cloned().unwrap_or_default();
            m.insert(key.to_string(), val);
            i += 2;
        } else {
            i += 1;
        }
    }
    m
}

fn opt<'a>(o: &'a HashMap<String, String>, k: &str) -> Option<&'a str> {
    o.get(k).map(String::as_str)
}
fn opt_usize(o: &HashMap<String, String>, k: &str, default: usize) -> usize {
    opt(o, k).and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn context_of(o: &HashMap<String, String>) -> Vec<u8> {
    opt(o, "context")
        .unwrap_or("pq-tornado-demo")
        .as_bytes()
        .to_vec()
}

// ---------------------------------------------------------------------------
// Hex helpers for H256 (8 big-endian u32 words → 64 hex chars)
// ---------------------------------------------------------------------------

fn h256_hex(v: &H256) -> String {
    v.iter().map(|w| format!("{w:08x}")).collect()
}

// ---------------------------------------------------------------------------
// Deterministic / OS randomness for notes
// ---------------------------------------------------------------------------

/// SplitMix64 → fill an H256.
fn h256_from_seed(seed: &mut u64) -> H256 {
    std::array::from_fn(|_| {
        *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) as u32
    })
}

fn random_seed() -> u64 {
    // Prefer OS randomness (read exactly 8 bytes — /dev/urandom never hits EOF).
    use std::io::Read;
    let mut buf = [0u8; 8];
    if let Ok(mut f) = std::fs::File::open("/dev/urandom")
        && f.read_exact(&mut buf).is_ok()
    {
        return u64::from_le_bytes(buf);
    }
    0xC0FF_EE12_3456_789A
}

fn make_note(seed: &mut u64) -> Note {
    Note {
        nullifier: h256_from_seed(seed),
        secret: h256_from_seed(seed),
    }
}

// ---------------------------------------------------------------------------
// Subcommands
// ---------------------------------------------------------------------------

fn cmd_deposit(o: &HashMap<String, String>) -> Result<(), String> {
    let mut seed = opt(o, "seed")
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(random_seed);
    let note = make_note(&mut seed);
    let cm = note.commitment();
    let json = serde_json::to_string_pretty(&note).map_err(|e| e.to_string())?;
    if let Some(path) = opt(o, "out") {
        std::fs::write(path, &json).map_err(|e| e.to_string())?;
        println!("note written to {path}");
    } else {
        println!("{json}");
    }
    println!("commitment (leaf): {}", h256_hex(&cm));
    println!("nullifier hash   : {}", h256_hex(&note.nullifier_hash()));
    Ok(())
}

/// Synthesize an authentication path for `note` at leaf `index` in a depth-`D`
/// tree, in O(D): random sibling hashes, path bits from the leaf index, and the
/// resulting root. A real prover likewise holds only its path — never the whole
/// `2^D`-leaf tree (which is infeasible to materialize for large `D`).
fn synth_path(note: &Note, depth: usize, index: usize) -> (Vec<H256>, Vec<bool>, H256) {
    let mut seed = 0xABCD_0000_0000_0001u64 ^ ((index as u64).wrapping_mul(0x9E37_79B9));
    let siblings: Vec<H256> = (0..depth).map(|_| h256_from_seed(&mut seed)).collect();
    let bits: Vec<bool> = (0..depth).map(|d| (index >> d) & 1 == 1).collect();
    let root = pq_tornado::note::recompute_root(&note.commitment(), &siblings, &bits);
    (siblings, bits, root)
}

fn cmd_withdraw(o: &HashMap<String, String>) -> Result<(), String> {
    let note_path = opt(o, "note").ok_or("withdraw needs --note")?;
    let depth = opt_usize(o, "depth", 8);
    let index = opt_usize(o, "index", 0);
    let ctx = context_of(o);

    let note: Note =
        serde_json::from_str(&std::fs::read_to_string(note_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;

    let (siblings, bits, _root) = synth_path(&note, depth, index);
    let witness = Witness {
        note,
        siblings,
        bits,
    };

    println!("building circuit (depth {depth}) ...");
    let t = Instant::now();
    let circuit = TornadoCircuit::build(depth);
    println!(
        "  circuit: {} compressions, k_log={}, m={}  ({:.2}s)",
        2 + depth,
        circuit.k_log,
        circuit.r1cs.m,
        t.elapsed().as_secs_f64()
    );

    let t = Instant::now();
    let (proof, pi) = pipeline::prove(&circuit, &witness, &ctx);
    println!("  prove: {:.2}s", t.elapsed().as_secs_f64());

    let proof_bytes = bincode::serialize(&proof).map_err(|e| e.to_string())?;
    let out = opt(o, "out").unwrap_or("proof.bin");
    std::fs::write(out, &proof_bytes).map_err(|e| e.to_string())?;
    let public_path = opt(o, "public").unwrap_or("public.json");
    std::fs::write(
        public_path,
        serde_json::to_string_pretty(&pi).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;

    println!("root          : {}", h256_hex(&pi.root));
    println!("nullifier hash: {}", h256_hex(&pi.nullifier_hash));
    println!("proof ({} bytes) → {out}", proof_bytes.len());
    println!("public inputs → {public_path}");
    Ok(())
}

fn cmd_verify(o: &HashMap<String, String>) -> Result<(), String> {
    let proof_path = opt(o, "proof").ok_or("verify needs --proof")?;
    let public_path = opt(o, "public").ok_or("verify needs --public")?;
    let depth = opt_usize(o, "depth", 8);
    let ctx = context_of(o);

    let proof: pipeline::WithdrawProof =
        bincode::deserialize(&std::fs::read(proof_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let pi: PublicInputs =
        serde_json::from_str(&std::fs::read_to_string(public_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;

    let circuit = TornadoCircuit::build(depth);
    let t = Instant::now();
    match pipeline::verify(&circuit, &proof, &pi, &ctx) {
        Ok(()) => {
            println!("VERIFIED ✓  ({:.3}s)", t.elapsed().as_secs_f64());
            println!("root          : {}", h256_hex(&pi.root));
            println!("nullifier hash: {}", h256_hex(&pi.nullifier_hash));
            Ok(())
        }
        Err(e) => Err(format!("verification FAILED: {e:?}")),
    }
}

/// End-to-end run of the **stepwise** pipeline: the withdrawal proved over
/// `2^n_log` copies of the single-compression step circuit, linked by the
/// slot-wiring argument (the Flock paper's glue circuit `G`).
fn cmd_fast_demo(o: &HashMap<String, String>) -> Result<(), String> {
    use pq_tornado::stepwise::{self, StepwiseCircuit};

    let depth = opt_usize(o, "depth", 32);
    let index = opt_usize(o, "index", 3);
    let ctx = context_of(o);

    println!("== post-quantum Tornado Cash — stepwise demo ==");
    println!("hash: SHA-256   proof system: Flock/Ligerito (transparent, PQ)");

    let t = Instant::now();
    let circuit = StepwiseCircuit::build(depth);
    let build_s = t.elapsed().as_secs_f64();
    println!(
        "tree depth {depth}  ->  {} SHA-256 compressions",
        StepwiseCircuit::n_compressions(depth)
    );
    println!(
        "circuit: k_log={} (one compression), {} instances, m={}  ({build_s:.3}s)",
        pq_tornado::step::K_LOG,
        1usize << circuit.step.n_log,
        circuit.m()
    );

    let mut seed = random_seed();
    let note = make_note(&mut seed);
    println!("\n[deposit]");
    println!("  commitment (leaf): {}", h256_hex(&note.commitment()));

    let (siblings, bits, expected_root) = synth_path(&note, depth, index);
    let witness = pq_tornado::circuit::Witness {
        note,
        siblings,
        bits,
    };

    println!("\n[withdraw]");
    let t = Instant::now();
    let (proof, public) = stepwise::prove(&circuit, &witness, &ctx);
    let prove_s = t.elapsed().as_secs_f64();
    let bytes = bincode::serialize(&proof).map_err(|e| e.to_string())?;
    println!(
        "  proved in {:.1} ms, proof {} KiB",
        prove_s * 1e3,
        bytes.len() / 1024
    );
    println!("  root          : {}", h256_hex(&public.root));
    println!("  nullifier hash: {}", h256_hex(&public.nullifier_hash));
    assert_eq!(public.root, expected_root);

    println!("\n[verify]");
    let t = Instant::now();
    stepwise::verify(&circuit, &proof, &public, &ctx).map_err(|e| format!("{e:?}"))?;
    println!("  VERIFIED \u{2713}  ({:.2} ms)", t.elapsed().as_secs_f64() * 1e3);

    let mut bad = public;
    bad.root[0] ^= 1;
    let rejected = stepwise::verify(&circuit, &proof, &bad, &ctx).is_err();
    println!("  tampered-root proof rejected: {rejected}");
    assert!(rejected);

    println!("\nOK");
    Ok(())
}

fn cmd_demo(o: &HashMap<String, String>) -> Result<(), String> {
    let depth = opt_usize(o, "depth", 8);
    let index = opt_usize(o, "index", 3.min((1 << depth) - 1));
    let ctx = context_of(o);

    println!("== post-quantum Tornado Cash demo ==");
    println!("hash: SHA-256   proof system: Flock/Ligerito (transparent, PQ)");
    println!("tree depth: {depth}  (capacity {} notes)", 1usize << depth);

    let mut seed = random_seed();
    let note = make_note(&mut seed);
    println!("\n[deposit]");
    println!("  commitment (leaf): {}", h256_hex(&note.commitment()));

    let (siblings, bits, expected_root) = synth_path(&note, depth, index);
    let witness = Witness {
        note,
        siblings,
        bits,
    };

    println!("\n[withdraw]");
    let t = Instant::now();
    let circuit = TornadoCircuit::build(depth);
    println!(
        "  circuit built: {} SHA-256 compressions, k_log={}, m={}  ({:.2}s)",
        2 + depth,
        circuit.k_log,
        circuit.r1cs.m,
        t.elapsed().as_secs_f64()
    );
    let t = Instant::now();
    let (proof, pi) = pipeline::prove(&circuit, &witness, &ctx);
    let prove_s = t.elapsed().as_secs_f64();
    let proof_bytes = bincode::serialize(&proof).map_err(|e| e.to_string())?;
    println!(
        "  proved in {prove_s:.2}s, proof size {} KiB",
        proof_bytes.len() / 1024
    );
    println!("  root          : {}", h256_hex(&pi.root));
    println!("  nullifier hash: {}", h256_hex(&pi.nullifier_hash));
    assert_eq!(pi.root, expected_root);

    println!("\n[verify]");
    let t = Instant::now();
    pipeline::verify(&circuit, &proof, &pi, &ctx).map_err(|e| format!("{e:?}"))?;
    println!("  VERIFIED ✓  ({:.3}s)", t.elapsed().as_secs_f64());

    // Negative check: tampering with the public root must be rejected.
    let mut bad = pi;
    bad.root[0] ^= 1;
    let rejected = pipeline::verify(&circuit, &proof, &bad, &ctx).is_err();
    println!("  tampered-root proof rejected: {rejected}");
    assert!(rejected);

    println!("\nOK");
    Ok(())
}
