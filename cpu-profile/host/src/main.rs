use anyhow::{ensure, Result};
use sha2::{Digest, Sha256};
use sp1_sdk::{CpuProver, Elf, Prover, SP1Stdin};
use std::{fs, path::PathBuf, time::Instant};
use zkdrl_profile_dqn_common::{Journal, PrivateInput};

fn hash(bytes: &[u8]) -> [u8; 32] { Sha256::digest(bytes).into() }

#[tokio::main]
async fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    ensure!(a.len() == 4,
        "native control|instrumented INPUT | execute original|control|instrumented INPUT");
    let variant = a[2].as_str();
    let input_bytes = fs::read(&a[3])?;
    let input: PrivateInput = serde_json::from_slice(&input_bytes)?;
    let input_hash = hex::encode(hash(&input_bytes));
    if a[1] == "native" {
        ensure!(matches!(variant, "control" | "instrumented"), "native variant");
        ensure!((variant == "instrumented") == cfg!(feature = "markers"),
            "use host-control for native control and host-instrumented for native instrumented");
        let output = zkdrl_profile_dqn_common::run(&input, hash, false);
        let journal = bincode::serialize(&output.journal)?;
        ensure!(journal.len() == 148, "journal ABI");
        println!("{}", serde_json::json!({
            "cryptographic_proof": false, "operation": "native", "variant": variant,
            "input_sha256": input_hash, "capture": false,
            "markers_feature": cfg!(feature = "markers"),
            "journal_hex": hex::encode(&journal), "journal_bytes": journal.len(),
            "output": output
        }));
        return Ok(());
    }
    ensure!(a[1] == "execute", "operation");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let elf_path = match variant {
        "original" => root.join("../dqn/elf/zkdrl-ctc-dqn-guest"),
        "control" => root.join("elf-control/zkdrl-profile-dqn-guest"),
        "instrumented" => root.join("elf-instrumented/zkdrl-profile-dqn-guest"),
        _ => anyhow::bail!("execute variant"),
    };
    let elf_bytes = fs::read(&elf_path)?;
    let elf_hash = hex::encode(hash(&elf_bytes));
    let init = Instant::now();
    // SDK profiling selects the portable CPU executor. No setup/prove entrypoint.
    let client = CpuProver::new().await;
    let init_seconds = init.elapsed().as_secs_f64();
    let mut stdin = SP1Stdin::new();
    stdin.write(&input);
    let start = Instant::now();
    let (values, report) = client.execute(Elf::from(elf_bytes), stdin)
        .cycle_limit(8_000_000_000).await
        .map_err(|e| anyhow::anyhow!("execution: {e:?}"))?;
    let execute_seconds = start.elapsed().as_secs_f64();
    ensure!(report.exit_code == 0, "nonzero guest exit");
    let bytes = values.as_slice();
    ensure!(bytes.len() == 148, "actual guest journal ABI");
    let journal: Journal = bincode::deserialize(bytes)?;
    ensure!(journal.schema == 202 && bincode::serialize(&journal)? == bytes,
        "actual guest journal canonical encoding/schema");
    println!("{}", serde_json::json!({
        "cryptographic_proof": false, "operation": "execute", "variant": variant,
        "input_sha256": input_hash, "elf": elf_path, "elf_sha256": elf_hash,
        "journal_hex": hex::encode(bytes), "journal_bytes": bytes.len(),
        "journal": journal, "instruction_cycles": report.total_instruction_count(),
        "syscalls": report.total_syscall_count(), "report": report,
        "client_init_seconds": init_seconds, "execute_seconds": execute_seconds,
        "cycle_limit": 8_000_000_000u64, "sdk": "6.9.0",
        "sdk_default_features": false, "sdk_features": ["profiling"]
    }));
    Ok(())
}
