use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use crate::{registry::Program, sha};

// Trusted auditor/controller input, NEVER authorization minted by the prover.
// Ledger registration, approval issuance and one-time challenge use live outside
// this host. The caller must obtain this path through that trusted channel.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Approval {
    pub schema: String,
    pub run_id: String,
    pub group_id: String,
    pub algorithm: String,
    pub elf_sha256: String,
    pub vk_bytes32: String,
    pub config_commit: [u8; 32],
    pub challenge: [u8; 32],
    pub kind: u32,
    pub episodes: u32,
    pub scale: u32,
    pub width: u32,
    // Require the JSON field even for A2C (null); serde otherwise treats a
    // missing Option field as None. Parent and host share the exact schema.
    #[serde(deserialize_with = "required_option")]
    pub capacity: Option<u32>,
}
fn required_option<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<u32>, D::Error> {
    Option::<u32>::deserialize(d)
}
pub fn lower_hex(s: &str, len: usize) -> bool {
    s.len() == len && s.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}
impl Approval {
    pub fn parse(program: Program, bytes: &[u8]) -> Result<Self> {
        // Direct typed deserialization rejects duplicate and unknown fields.
        let a: Self = serde_json::from_slice(bytes)?;
        a.validate(program)?;
        Ok(a)
    }
    pub fn validate(&self, p: Program) -> Result<()> {
        ensure!(self.schema == "zkdrl-ctc-approval-v1", "approval schema");
        for id in [&self.run_id, &self.group_id] {
            ensure!(!id.trim().is_empty() && id.len() <= 256 && !id.chars().any(char::is_control),
                "approval IDs must be nonblank, at most 256 UTF-8 bytes, without control characters");
        }
        ensure!(self.algorithm == p.id(), "approval algorithm");
        ensure!(lower_hex(&self.elf_sha256, 64), "approval ELF hash encoding");
        ensure!(self.vk_bytes32.starts_with("0x") && lower_hex(&self.vk_bytes32[2..], 64), "approval key encoding");
        ensure!(self.kind <= 1 && (1..=4000).contains(&self.episodes), "approval task/budget");
        ensure!(match p { Program::Dqn => matches!(self.scale, 16 | 256), Program::A2c => self.scale == 65536 },
            "approval scale");
        ensure!(self.width == p.width() && self.capacity == p.capacity(), "approval profile");
        ensure!(self.elf_sha256 == sha(p.elf()), "approval ELF differs from embedded ELF");
        Ok(())
    }
    pub fn pin_key(&self, actual_vk: &str) -> Result<()> {
        ensure!(self.vk_bytes32 == actual_vk, "approval key differs from embedded ELF CPU setup");
        Ok(())
    }
}
