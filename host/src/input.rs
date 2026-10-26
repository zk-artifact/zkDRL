use anyhow::{ensure, Result};
use sp1_sdk::SP1Stdin;
use crate::{approval::Approval, hash, registry::Program};
use zkdrl_ctc_a2c as a;
use zkdrl_ctc_dqn as d;

pub enum Input { Dqn(d::PrivateInput), A2c(a::PrivateInput) }
impl Input {
    pub fn parse(p: Program, bytes: &[u8]) -> Result<Self> {
        let input = match p {
            Program::Dqn => Self::Dqn(serde_json::from_slice(bytes)?),
            Program::A2c => Self::A2c(serde_json::from_slice(bytes)?),
        };
        input.validate()?;
        Ok(input)
    }
    pub fn validate(&self) -> Result<()> {
        // Mirror the core's public input bounds without running training.
        match self {
            Self::Dqn(i) => {
                ensure!(i.schema == d::SCHEMA && i.kind <= 1 && i.train <= 1, "DQN schema/kind/train");
                ensure!(matches!(i.scale, 16 | 256), "DQN scale");
                ensure!(i.slots >= d::HORIZON && i.slots <= d::MAX_SLOTS && i.slots % d::HORIZON == 0, "DQN slots");
                ensure!(i.initial_coarse.len() == d::P, "DQN parameter count");
                ensure!(i.initial_coarse.iter().all(|&x| -(1i64 << 20) <= x && x < (1i64 << 20)), "DQN signed21 parameters");
                let bound = 16 * i64::from(i.scale);
                ensure!(i.params.iter().all(|&x| (-bound..=bound).contains(&x)), "DQN reward parameters");
                ensure!(i.kind != 0 || i.params == [0; 4], "DQN login parameters must be zero");
            },
            Self::A2c(i) => {
                ensure!(i.schema == a::SCHEMA && i.kind <= 1 && (1..=a::MAX_EPISODES as u32).contains(&i.episodes), "A2C schema/kind/episodes");
                ensure!(i.initial_actor.len() == a::ACTOR && i.initial_critic.len() == a::CRITIC, "A2C parameter count");
                ensure!(i.initial_actor.iter().chain(&i.initial_critic).all(|&x|
                    x > -256*a::G && x < 256*a::G && (x >> 24) > -256*a::S && (x >> 24) < 256*a::S), "A2C master/coarse bounds");
            },
        }
        Ok(())
    }
    pub fn check_approval(&self, approval: &Approval) -> Result<()> {
        self.validate()?;
        let (commit, challenge, kind, episodes, scale) = match self {
            Self::Dqn(i) => {
                ensure!(approval.algorithm == "dqn" && i.train == 1, "approved DQN training required");
                (hash(&d::config_bytes(i)), i.challenge, i.kind, i.slots/d::HORIZON, i.scale)
            },
            Self::A2c(i) => {
                ensure!(approval.algorithm == "a2c", "approved algorithm");
                (hash(&a::config_bytes(i)), i.challenge, i.kind, i.episodes, a::S as u32)
            },
        };
        ensure!(commit == approval.config_commit, "input config commitment differs from approval");
        ensure!(challenge == approval.challenge, "input challenge differs from approval");
        ensure!(kind == approval.kind && episodes == approval.episodes && scale == approval.scale, "input task/budget/scale differs from approval");
        Ok(())
    }
    pub fn stdin(&self) -> SP1Stdin {
        let mut stdin = SP1Stdin::new();
        match self { Self::Dqn(i) => stdin.write(i), Self::A2c(i) => stdin.write(i) }
        stdin
    }
    pub fn native_json(&self, capture: bool) -> Result<String> {
        // Serialize the core Output directly: no metadata or field omissions.
        match self {
            Self::Dqn(i) => Ok(serde_json::to_string(&d::run(i, hash, capture))?),
            Self::A2c(i) => Ok(serde_json::to_string(&a::run(i, hash, capture))?),
        }
    }
}
