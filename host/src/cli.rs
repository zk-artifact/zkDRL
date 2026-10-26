use anyhow::{bail, ensure, Result};
use crate::registry::Program;

pub const USAGE: &str = "identity ALG | native ALG INPUT [--capture] | execute ALG INPUT | prove ALG INPUT APPROVAL PROOF | verify|tamper|audit ALG PROOF APPROVAL";
#[derive(Debug)]
pub enum Command {
    Identity,
    Native { input: String, capture: bool },
    Execute { input: String },
    Prove { input: String, approval: String, proof: String },
    Verify { proof: String, approval: String },
    Tamper { proof: String, approval: String },
    Audit { proof: String, approval: String },
}
impl Command {
    pub fn parse(args: &[String]) -> Result<(Program, Self)> {
        ensure!(args.len() >= 3, "{USAGE}");
        let p = Program::parse(&args[2])?;
        let command = match args[1].as_str() {
            "identity" if args.len() == 3 => Self::Identity,
            "native" if args.len() == 4 || (args.len() == 5 && args[4] == "--capture") =>
                Self::Native { input: args[3].clone(), capture: args.len() == 5 },
            "execute" if args.len() == 4 => Self::Execute { input: args[3].clone() },
            "prove" if args.len() == 6 => Self::Prove { input: args[3].clone(), approval: args[4].clone(), proof: args[5].clone() },
            "verify" if args.len() == 5 => Self::Verify { proof: args[3].clone(), approval: args[4].clone() },
            "tamper" if args.len() == 5 => Self::Tamper { proof: args[3].clone(), approval: args[4].clone() },
            "audit" if args.len() == 5 => Self::Audit { proof: args[3].clone(), approval: args[4].clone() },
            _ => bail!("invalid command/arguments; {USAGE}"),
        };
        Ok((p, command))
    }
    pub fn name(&self) -> &'static str {
        match self { Self::Identity => "identity", Self::Native {..} => "native", Self::Execute {..} => "execute",
            Self::Prove {..} => "prove", Self::Verify {..} => "verify", Self::Tamper {..} => "tamper", Self::Audit {..} => "audit" }
    }
}
