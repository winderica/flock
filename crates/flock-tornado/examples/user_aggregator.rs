//! **User / aggregator benchmark.** Each user proves ONE withdrawal on their
//! own machine; an aggregator verifies every proof on intake and folds them
//! into one recursive root.
//!
//! ```sh
//! cargo run --release -p flock-tornado --example user_aggregator -- [depth] [n,n,...] [user_runs]
//! ```
//!
//! Defaults: depth 20 (Tornado Cash's tree), n = 4,8,16,32,64,128, 20 runs.

use std::{env, io::Write, time::Instant};

use flock_prover::tower::{TowerConfig, verify_root_bytes};
use flock_tornado::{
    Aggregator, TornadoStatement, UserProof, generate_tornado_vk, prove_withdrawal,
    testing::sample_withdrawal, verify_withdrawal,
};

const CFG: TowerConfig = TowerConfig::Chain128;

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn flush() {
    std::io::stdout().flush().ok();
}

fn main() {
    let mut args = env::args().skip(1);
    let depth: usize = args.next().map_or(20, |v| v.parse().expect("depth"));
    let ns: Vec<usize> = args.next().map_or(vec![4, 8, 16, 32, 64, 128], |v| {
        v.split(',').map(|x| x.parse().expect("n")).collect()
    });
    let runs: usize = args.next().map_or(20, |v| v.parse().expect("user_runs"));
    let threads = rayon::current_num_threads();

    println!(
        "== USER: one withdrawal, depth {depth} ({} SHA-256 compressions) ==",
        depth + 2
    );

    // First proof in a fresh process: includes building the circuit shape
    // and the step table (a wallet would ship these precomputed).
    let w0 = sample_withdrawal(1, depth);
    let t = Instant::now();
    let (p0, _) = prove_withdrawal(CFG, depth, &w0);
    let cold = ms(t);

    // Steady state, all cores.
    let mut warm = Vec::with_capacity(runs);
    for i in 0..runs {
        let w = sample_withdrawal(10 + i as u32, depth);
        let t = Instant::now();
        let _ = prove_withdrawal(CFG, depth, &w);
        warm.push(ms(t));
    }
    // Steady state, ONE thread — a proxy for a weak device.
    let pool1 = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("1-thread pool");
    let mut one = Vec::with_capacity(runs.min(10));
    pool1.install(|| {
        let _ = prove_withdrawal(CFG, depth, &w0);
        for i in 0..runs.min(10) {
            let w = sample_withdrawal(500 + i as u32, depth);
            let t = Instant::now();
            let _ = prove_withdrawal(CFG, depth, &w);
            one.push(ms(t));
        }
    });
    let mut ver = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = Instant::now();
        verify_withdrawal(CFG, depth, &p0).expect("verifies");
        ver.push(ms(t));
    }
    let size = bincode::serialize(&p0).expect("serialize").len();
    println!(
        "  prove   : {:.1} ms median ({threads} threads) | {:.1} ms median (1 thread) | first in process {:.0} ms\n  \
         verify  : {:.1} ms median\n  \
         proof   : {:.1} KiB on the wire",
        median(warm),
        median(one),
        cold,
        median(ver),
        size as f64 / 1024.0,
    );
    flush();

    println!("\n== AGGREGATOR: fold n user proofs into one root ==");
    let t = Instant::now();
    let vk = generate_tornado_vk(CFG, depth, 1);
    println!(
        "  verification key (one-time, per depth): {:.1} s",
        ms(t) / 1e3
    );
    flush();

    let mut rows = Vec::new();
    for &n in &ns {
        // Each user proves their own withdrawal (timed individually; in a
        // deployment these run in parallel on the users' own machines).
        let mut user_ms = Vec::with_capacity(n);
        let proofs: Vec<UserProof> = (0..n)
            .map(|i| {
                let w = sample_withdrawal(10_000 + i as u32, depth);
                let t = Instant::now();
                let (p, _) = prove_withdrawal(CFG, depth, &w);
                user_ms.push(ms(t));
                p
            })
            .collect();
        let user_avg = user_ms.iter().sum::<f64>() / n as f64;

        // The aggregator: check every proof, then merge them all.
        let t = Instant::now();
        let mut agg = Aggregator::new(CFG, depth);
        for p in proofs {
            agg.admit(p).expect("honest proof admitted");
        }
        let want = agg.statement();
        let tower = agg.aggregate();
        let agg_wall = ms(t);

        // The final proof, as a verifier receives it.
        let bytes = tower.root_bundle_bytes();
        let t = Instant::now();
        let got = verify_root_bytes(&vk, &bytes).expect("final proof verifies");
        let verify = ms(t);
        assert_eq!(
            TornadoStatement::from_app(&got),
            want,
            "statement rode the wire"
        );

        println!(
            "  {n:>3} users: user {user_avg:.1} ms avg | aggregator {:.1} s | final proof {:.1} KiB | verify {verify:.0} ms",
            agg_wall / 1e3,
            bytes.len() as f64 / 1024.0,
        );
        flush();
        rows.push((n, user_avg, agg_wall, bytes.len(), verify));
    }

    println!(
        "\n| users | user prove (avg) | aggregator wall time | final proof size | verify |"
    );
    println!("|---:|---:|---:|---:|---:|");
    for (n, ua, aw, b, v) in rows {
        println!(
            "| {n} | {ua:.1} ms | {:.1} s | {:.1} KiB | {v:.0} ms |",
            aw / 1e3,
            b as f64 / 1024.0,
        );
    }
}
