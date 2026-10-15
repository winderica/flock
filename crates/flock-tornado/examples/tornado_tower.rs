//! The recursion tower over a post-quantum Tornado Cash withdrawal set:
//! prove `n_leaves` batches of `per_leaf` depth-`depth` withdrawals, fold
//! them to ONE recursable root, discharge the root-side residue, and
//! (optionally) verify the root STANDALONE from its wire bytes.
//!
//! ```sh
//! cargo run --release --example tornado_tower -- [n_leaves] [per_leaf] [depth] [chain128|chain100] [verify]
//! ```
//!
//! Defaults: 4 leaves x 32 withdrawals x depth 20 (Tornado Cash's tree).

use std::{env, time::Instant};

use flock_prover::tower::{TowerConfig, TowerVk, verify_root_bytes};
use flock_tornado::{Note, TornadoStatement, Withdrawal, generate_tornado_vk, prove_tornado};

fn splitmix(seed: &mut u64) -> [u32; 8] {
    std::array::from_fn(|_| {
        *seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) as u32
    })
}

fn synth(depth: usize, seed: &mut u64) -> Withdrawal {
    let note = Note {
        nullifier: splitmix(seed),
        secret: splitmix(seed),
    };
    let siblings = (0..depth).map(|_| splitmix(seed)).collect();
    let bits = (0..depth).map(|d| (*seed >> (d % 63)) & 1 == 1).collect();
    Withdrawal {
        note,
        siblings,
        bits,
    }
}

fn main() {
    let mut args = env::args().skip(1);
    let n_leaves: usize = args
        .next()
        .map_or(4, |v| v.parse().expect("n_leaves: an even integer >= 4"));
    let per_leaf: usize = args.next().map_or(32, |v| v.parse().expect("per_leaf"));
    let depth: usize = args.next().map_or(20, |v| v.parse().expect("depth"));
    let cfg = match args.next().as_deref() {
        None | Some("chain128") => TowerConfig::Chain128,
        Some("chain100") => TowerConfig::Chain100,
        Some(other) => panic!("unknown tower config {other:?}"),
    };
    let do_verify = matches!(args.next().as_deref(), Some("verify"));

    let mut seed = 0xC0FF_EE00_1234_5678u64;
    let leaves: Vec<Vec<Withdrawal>> = (0..n_leaves)
        .map(|_| (0..per_leaf).map(|_| synth(depth, &mut seed)).collect())
        .collect();

    let t0 = Instant::now();
    let tower = prove_tornado(cfg, depth, &leaves);
    let prove_s = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    tower
        .discharge_root()
        .expect("the root-side residue discharges");
    let discharge_s = t1.elapsed().as_secs_f64();

    let st = &TornadoStatement::from_app(tower.statement());
    println!(
        "\nTORNADO TOWER {cfg:?}: {n_leaves} leaves x {per_leaf} withdrawals x depth {depth} \
         = {} withdrawals ({} SHA-256 compressions)\n  \
         digest = {:08x}{:08x}..\n  \
         ONE root proof: {:.1} KiB | prove {prove_s:.1}s ({:.1} ms/withdrawal) | root discharge {discharge_s:.3}s",
        st.n_withdrawals,
        st.n_withdrawals * (depth + 2),
        st.digest[0],
        st.digest[1],
        tower.root_proof_kib(),
        prove_s * 1e3 / st.n_withdrawals as f64,
    );

    let c = tower.costs();
    println!(
        "  COSTS: online {:.1}s = leaves {}x{:.0}ms + FLs {}x{:.0}ms + nodes {}x{:.0}ms  ({:.2} ms/withdrawal)\n  \
         per-shape SETUP (circuit build, cacheable, not yet cached by the driver): {:.1}s",
        c.online_ms() / 1e3,
        c.leaves,
        c.leaf_online_ms / c.leaves.max(1) as f64,
        c.fls,
        c.fl_online_ms / c.fls.max(1) as f64,
        c.nodes,
        c.node_online_ms / c.nodes.max(1) as f64,
        c.online_ms() / st.n_withdrawals as f64,
        c.setup_ms() / 1e3,
    );

    if do_verify {
        let t2 = Instant::now();
        let vk: TowerVk = generate_tornado_vk(cfg, depth, per_leaf);
        let gen_s = t2.elapsed().as_secs_f64();
        let bytes = tower.root_bundle_bytes();
        let t3 = Instant::now();
        let got = verify_root_bytes(&vk, &bytes).expect("the wire bundle verifies standalone");
        let verify_s = t3.elapsed().as_secs_f64();
        assert_eq!(
            &TornadoStatement::from_app(&got),
            st,
            "the statement rode the wire"
        );
        println!(
            "  VERIFIED STANDALONE (consumer path, over the wire): bundle {:.1} KiB \
             | vk generate {gen_s:.1}s (one-time) | verify {verify_s:.3}s | statement certified",
            bytes.len() as f64 / 1024.0,
        );
    }
}
