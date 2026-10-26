use anyhow::{ensure, Context, Result};
use bincode::Options;
use serde::{de::DeserializeOwned, Serialize};
use serde_json::{json, Value};
use sp1_sdk::{SP1Proof, SP1ProofWithPublicValues};
use sp1_verifier::{Groth16Verifier, GROTH16_VK_BYTES};
use crate::{approval::Approval, journal::Journal, registry::{Program, CIRCUIT}};

pub const MAX_BUNDLE: u64 = 1024 * 1024;

pub fn strict_decode<T: DeserializeOwned + Serialize>(bytes: &[u8]) -> Result<T> {
    ensure!(bytes.len() as u64 <= MAX_BUNDLE, "artifact exceeds 1 MiB");
    let value: T = bincode::DefaultOptions::new().with_fixint_encoding()
        .with_limit(MAX_BUNDLE).reject_trailing_bytes().deserialize(bytes).context("strict bincode decode")?;
    ensure!(bincode::serialize(&value)? == bytes, "noncanonical bincode encoding");
    Ok(value)
}
pub fn wrapper(p: &SP1ProofWithPublicValues) -> Result<Vec<u8>> {
    ensure!(matches!(&p.proof, SP1Proof::Groth16(_)) && p.tee_proof.is_none(), "Groth16 without TEE required");
    ensure!(p.sp1_version == CIRCUIT, "wrong circuit version");
    let bytes = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| p.bytes()))
        .map_err(|_| anyhow::anyhow!("malformed Groth16 wrapper encoding"))?;
    ensure!(bytes.len() == 356, "canonical 356-byte wrapper required");
    Ok(bytes)
}
pub fn decode(bytes: &[u8]) -> Result<SP1ProofWithPublicValues> {
    let p = strict_decode(bytes)?;
    wrapper(&p)?;
    Ok(p)
}
pub fn policy(p: &SP1ProofWithPublicValues, program: Program, a: &Approval, actual_vk: &str) -> Result<Journal> {
    a.validate(program)?;
    a.pin_key(actual_vk)?;
    wrapper(p)?;
    let journal = Journal::decode(program, p.public_values.as_slice())?;
    journal.policy(program, a)?;
    Ok(journal)
}
pub fn verify(p: &SP1ProofWithPublicValues, program: Program, a: &Approval, actual_vk: &str) -> Result<Journal> {
    let journal = policy(p, program, a, actual_vk)?;
    // The default lightweight API enforces SUCCESS and the release VK root.
    // Always authenticate the ACTUAL public_values bytes, with the key derived
    // from the embedded ELF, not an expected full journal or an arbitrary key.
    Groth16Verifier::verify(&wrapper(p)?, p.public_values.as_slice(), actual_vk, &GROTH16_VK_BYTES)
        .map_err(|e| anyhow::anyhow!("cryptographic verification: {e:?}"))?;
    Ok(journal)
}
fn crypto_reject(seal: &[u8], journal: &[u8], vk: &str) -> Result<()> {
    let error = Groth16Verifier::verify(seal, journal, vk, &GROTH16_VK_BYTES)
        .err().context("tamper accepted by cryptographic verifier")?;
    ensure!(format!("{error:?}") == "ProofVerificationFailed", "wrong rejection stage: {error:?}");
    Ok(())
}
pub fn tamper(p: &SP1ProofWithPublicValues, program: Program, a: &Approval, actual_vk: &str) -> Result<Value> {
    let journal = verify(p, program, a, actual_vk)?;
    let honest = journal.value()?;
    let seal = wrapper(p)?;
    let bytes = p.public_values.as_slice();
    let mut fields = Vec::new();
    for (name, value) in honest.as_object().context("journal object")? {
        let mut changed = honest.clone();
        if value.is_array() {
            changed[name][0] = json!(value[0].as_u64().context("byte")? ^ 1);
        } else if name == "return_units" {
            changed[name] = json!(value.as_i64().context("return")? ^ 1);
        } else {
            changed[name] = json!(value.as_u64().context("counter")? ^ 1);
        }
        let altered = Journal::from_json(program, &serde_json::to_vec(&changed)?)?.encode()?;
        ensure!(altered != bytes, "ineffective mutation: {name}");
        // Bypass all policy gates: every public field must fail cryptography.
        crypto_reject(&seal, &altered, actual_vk)?;
        fields.push(json!({"field":name,"cryptographic_error":"ProofVerificationFailed"}));
    }
    ensure!(fields.len() == if program == Program::Dqn {15} else {14}, "journal field coverage");
    for field in ["challenge", "config_commit"] {
        let mut alternate = a.clone();
        if field == "challenge" { alternate.challenge[0] ^= 1; } else { alternate.config_commit[0] ^= 1; }
        ensure!(policy(p, program, &alternate, actual_vk).is_err(), "alternate approval {field} accepted");
    }
    let mut wrong_key = hex::decode(&actual_vk[2..])?;
    wrong_key[31] ^= 1;
    crypto_reject(&seal, bytes, &format!("0x{}", hex::encode(wrong_key)))?;
    // SP1 v6.1.0: byte 99 is the low nonce byte; keep curve encodings,
    // exit code and VK root intact so the failure must be cryptographic.
    let mut corrupt = seal.clone();
    corrupt[99] ^= 1;
    crypto_reject(&corrupt, bytes, actual_vk)?;
    let serialized = bincode::serialize(p)?;
    let mut trailing = serialized.clone(); trailing.push(0);
    ensure!(decode(&trailing).is_err(), "trailing proof accepted");
    ensure!(decode(&serialized[..serialized.len()-1]).is_err(), "truncated proof accepted");
    let mut trailing_journal = bytes.to_vec(); trailing_journal.push(0);
    ensure!(Journal::decode(program, &trailing_journal).is_err(), "trailing journal accepted");
    ensure!(Journal::decode(program, &bytes[..bytes.len()-1]).is_err(), "truncated journal accepted");
    Ok(json!({"honest_accepted":true,"fields":fields,
        "alternate_challenge_policy_rejected":true,"alternate_config_commit_policy_rejected":true,
        "wrong_key_error":"ProofVerificationFailed","nonce_error":"ProofVerificationFailed",
        "corruption_method":"wrapper byte99 low-bit flip",
        "trailing_rejected":true,"truncated_rejected":true,"journal_trailing_rejected":true,"journal_truncated_rejected":true}))
}
