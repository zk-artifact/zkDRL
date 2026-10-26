use anyhow::{ensure, Context, Result};
use serde_json::json;
use sha2::{Digest, Sha256};
use sp1_sdk::{CpuProver, Elf, HashableKey, ProveRequest, Prover, ProverClient, ProvingKey};
use std::time::{Duration, Instant};

mod approval;
mod artifact;
mod cli;
mod input;
mod journal;
mod proof;
mod registry;
#[cfg(test)] mod tests;

use approval::Approval;
use artifact::read_bounded;
use cli::Command;
use input::Input;
use journal::Journal;
use registry::{Program, CIRCUIT, SDK};

fn hash(bytes: &[u8]) -> [u8; 32] { Sha256::digest(bytes).into() }
fn sha(bytes: &[u8]) -> String { hex::encode(hash(bytes)) }
fn input_file(p: Program, path: &str) -> Result<Input> { Input::parse(p, &read_bounded(path, proof::MAX_BUNDLE)?) }
fn approval_file(p: Program, path: &str) -> Result<Approval> { Approval::parse(p, &read_bounded(path, 16 * 1024)?) }

fn cuda_environment_values(prover: &str, visible: &str, uuid: &str) -> Result<()> {
    ensure!(prover == "cuda", "SP1_PROVER must equal cuda; CPU/mock/network fallback refused");
    ensure!(visible == "0", "CUDA_VISIBLE_DEVICES must equal 0");
    ensure!(!uuid.trim().is_empty(), "ZKDRL_APPROVED_GPU_UUID must be nonempty");
    Ok(())
}
fn cuda_environment() -> Result<(String, String)> {
    let prover = std::env::var("SP1_PROVER").context("SP1_PROVER required")?;
    let visible = std::env::var("CUDA_VISIBLE_DEVICES").context("CUDA_VISIBLE_DEVICES required")?;
    let uuid = std::env::var("ZKDRL_APPROVED_GPU_UUID").context("ZKDRL_APPROVED_GPU_UUID required")?;
    cuda_environment_values(&prover, &visible, &uuid)?;
    Ok((visible, uuid))
}

#[tokio::main]
async fn main() -> Result<()> {
    let total = Instant::now();
    let (p, command) = Command::parse(&std::env::args().collect::<Vec<_>>())?;
    if let Command::Native { input, capture } = &command {
        println!("{}", input_file(p, input)?.native_json(*capture)?);
        return Ok(());
    }
    // No logger on native: stdout is exactly one serialized core Output.
    sp1_sdk::utils::setup_logger();
    if let Command::Execute { input } = &command {
        let input = input_file(p, input)?;
        let t = Instant::now();
        let client = CpuProver::new().await;
        let init = t.elapsed().as_secs_f64();
        ensure!(client.version() == CIRCUIT, "SDK circuit version differs");
        let t = Instant::now();
        let (public_values, report) = client.execute(Elf::Static(p.elf()), input.stdin()).await
            .map_err(|e| anyhow::anyhow!("CPU guest execution: {e:?}"))?;
        let execute = t.elapsed().as_secs_f64();
        ensure!(report.exit_code == 0, "guest exit code {}", report.exit_code);
        let journal = Journal::decode(p, public_values.as_slice())?.value()?;
        let t = Instant::now();
        drop(client);
        let cleanup = t.elapsed().as_secs_f64();
        println!("{}", json!({"operation":"execute","algorithm":p.id(),"client":"CpuProver",
            "cryptographic_proof":false,"sdk":SDK,"circuit_version":CIRCUIT,"profile":p.profile(),
            "journal":journal,"journal_bytes":public_values.as_slice().len(),
            "journal_hex":hex::encode(public_values.as_slice()),"instruction_cycles":report.total_instruction_count(),
            "syscalls":report.total_syscall_count(),"exit_code":report.exit_code,"report":report,
            "client_init_seconds":init,"execute_seconds":execute,"cleanup_seconds":cleanup,
            "elapsed_seconds":total.elapsed().as_secs_f64()}));
        return Ok(());
    }
    // Check input/approval/config binding and CUDA authorization before CUDA
    // construction. CPU setup is also required to pin every verification key.
    let mut proving_input = None;
    let mut loaded_proof = None;
    let mut approved = None;
    let mut gpu = None;
    match &command {
        Command::Prove { input, approval, proof } => {
            let a = approval_file(p, approval)?;
            let i = input_file(p, input)?;
            i.check_approval(&a)?;
            artifact::refuse_existing(proof)?;
            gpu = Some(cuda_environment()?);
            proving_input = Some(i);
            approved = Some(a);
        },
        Command::Verify { proof, approval } | Command::Tamper { proof, approval } | Command::Audit { proof, approval } => {
            approved = Some(approval_file(p, approval)?);
            loaded_proof = Some(proof::decode(&read_bounded(proof, proof::MAX_BUNDLE)?)?);
        },
        Command::Identity => {},
        _ => unreachable!(),
    }
    let t = Instant::now();
    let cpu = CpuProver::new().await;
    let cpu_init = t.elapsed().as_secs_f64();
    ensure!(cpu.version() == CIRCUIT, "SDK circuit version differs");
    let t = Instant::now();
    let cpu_key = cpu.setup(Elf::Static(p.elf())).await
        .map_err(|e| anyhow::anyhow!("CPU setup: {e:?}"))?;
    let cpu_setup = t.elapsed().as_secs_f64();
    let vk = cpu_key.verifying_key().bytes32();
    if let Some(a) = &approved { a.pin_key(&vk)?; }
    let mut metrics = json!({"operation":command.name(),"algorithm":p.id(),"sdk":SDK,
        "circuit_version":CIRCUIT,"profile":p.profile(),"elf_sha256":sha(p.elf()),"elf_bytes":p.elf().len(),
        "vk_bytes32":vk,"release_vk_sha256":sha(&sp1_verifier::GROTH16_VK_BYTES),
        "cpu_client_init_seconds":cpu_init,"cpu_setup_seconds":cpu_setup});
    if matches!(&command, Command::Identity) {
        metrics["client"] = json!("CpuProver");
        metrics["vk_hash_u32"] = json!(cpu_key.verifying_key().hash_u32());
        metrics["vk_sha256"] = json!(sha(&bincode::serialize(cpu_key.verifying_key())?));
    }
    if let Command::Audit {..} = &command {
        let bundle = loaded_proof.as_ref().context("proof")?;
        let a = approved.as_ref().context("approval")?;
        // Structural and approval checks only; do NOT invoke light verification
        // on this independent audit path. None requests default SUCCESS.
        let journal = proof::policy(bundle, p, a, &vk)?;
        let t = Instant::now();
        cpu.verify(bundle, cpu_key.verifying_key(), None)?;
        metrics["verification_seconds"] = json!(t.elapsed().as_secs_f64());
        metrics["verified"] = json!(true);
        metrics["client"] = json!("CpuProver");
        metrics["journal"] = journal.value()?;
        metrics["run_id"] = json!(a.run_id);
        metrics["group_id"] = json!(a.group_id);
    }
    let t = Instant::now();
    drop(cpu_key);
    drop(cpu);
    let cpu_cleanup = t.elapsed().as_secs_f64();
    metrics["cpu_cleanup_seconds"] = json!(cpu_cleanup);
    match &command {
        Command::Identity | Command::Audit {..} => {},
        Command::Verify {..} | Command::Tamper {..} => {
            let a = approved.as_ref().context("approval")?;
            let bundle = loaded_proof.as_ref().context("proof")?;
            let t = Instant::now();
            let journal = proof::verify(bundle, p, a, &vk)?;
            metrics["verification_seconds"] = json!(t.elapsed().as_secs_f64());
            if matches!(&command, Command::Tamper {..}) {
                let t = Instant::now();
                metrics["checks"] = proof::tamper(bundle, p, a, &vk)?;
                metrics["tamper_seconds"] = json!(t.elapsed().as_secs_f64());
            }
            metrics["client"] = json!("sp1-verifier");
            metrics["verified"] = json!(true);
            metrics["journal"] = journal.value()?;
            metrics["run_id"] = json!(a.run_id);
            metrics["group_id"] = json!(a.group_id);
        },
        Command::Prove { proof: out, .. } => {
            let a = approved.as_ref().context("approval")?;
            let i = proving_input.as_ref().context("input")?;
            let (visible, uuid) = gpu.context("CUDA environment")?;
            metrics["client"] = json!("CudaProver");
            metrics["mode"] = json!("groth16");
            metrics["device_id"] = json!(0);
            metrics["cuda_visible_devices"] = json!(visible);
            metrics["approved_gpu_uuid"] = json!(uuid);
            metrics["run_id"] = json!(a.run_id);
            metrics["group_id"] = json!(a.group_id);
            metrics["verified"] = json!(false);
            let t = Instant::now();
            // Concrete CUDA builder: SP1_PROVER cannot redirect to CPU/mock/network.
            let client = ProverClient::builder().cuda().with_device_id(0).build().await;
            metrics["client_init_seconds"] = json!(t.elapsed().as_secs_f64());
            let mut key = None;
            let outcome: Result<()> = async {
                ensure!(client.version() == CIRCUIT, "CUDA circuit version differs");
                metrics["stage"] = json!("setup");
                let t = Instant::now();
                let setup = client.setup(Elf::Static(p.elf())).await;
                metrics["setup_seconds"] = json!(t.elapsed().as_secs_f64());
                key = Some(setup.map_err(|e| anyhow::anyhow!("CUDA setup: {e:?}"))?);
                let pk = key.as_ref().context("CUDA key")?;
                ensure!(pk.verifying_key().bytes32() == vk, "CUDA key differs from embedded ELF CPU key");
                let stdin = i.stdin();
                metrics["stage"] = json!("prove");
                let t = Instant::now();
                let result = client.prove(pk, stdin).groth16().await;
                metrics["prove_including_execution_witness_seconds"] = json!(t.elapsed().as_secs_f64());
                let bundle = result.map_err(|e| anyhow::anyhow!("CUDA Groth16 proving: {e:?}"))?;
                metrics["stage"] = json!("persist");
                let bytes = bincode::serialize(&bundle)?;
                let seal = proof::wrapper(&bundle)?;
                // Persist BEFORE inline verification, keeping failed verification
                // artifacts for diagnosis. No proof is ever overwritten.
                let t = Instant::now();
                artifact::write(out, &bytes, &seal, bundle.public_values.as_slice())?;
                metrics["artifact_write_seconds"] = json!(t.elapsed().as_secs_f64());
                metrics["artifacts"] = json!(artifact::paths(out));
                metrics["artifact_bytes"] = json!(bytes.len());
                metrics["artifact_sha256"] = json!(sha(&bytes));
                metrics["wrapper_bytes"] = json!(seal.len());
                metrics["journal_bytes"] = json!(bundle.public_values.as_slice().len());
                metrics["stage"] = json!("verify");
                let t = Instant::now();
                let verification = proof::decode(&bytes).and_then(|saved| proof::verify(&saved, p, a, &vk));
                metrics["verification_seconds"] = json!(t.elapsed().as_secs_f64());
                metrics["journal"] = verification?.value()?;
                metrics["verified"] = json!(true);
                metrics["cryptographic_proof"] = json!(true);
                metrics["stage"] = json!("complete");
                Ok(())
            }.await;
            let t = Instant::now();
            drop(key);
            metrics["key_cleanup_seconds"] = json!(t.elapsed().as_secs_f64());
            let t = Instant::now();
            tokio::time::sleep(Duration::from_millis(250)).await;
            drop(client);
            tokio::time::sleep(Duration::from_millis(500)).await;
            metrics["client_cleanup_seconds"] = json!(t.elapsed().as_secs_f64());
            metrics["cleanup_seconds"] = json!(cpu_cleanup
                + metrics["key_cleanup_seconds"].as_f64().unwrap_or(0.0)
                + metrics["client_cleanup_seconds"].as_f64().unwrap_or(0.0));
            if let Err(e) = outcome {
                metrics["error"] = json!(format!("{e:#}"));
                metrics["elapsed_seconds"] = json!(total.elapsed().as_secs_f64());
                println!("{metrics}");
                return Err(e);
            }
        },
        _ => unreachable!(),
    }
    metrics["elapsed_seconds"] = json!(total.elapsed().as_secs_f64());
    println!("{metrics}");
    Ok(())
}
