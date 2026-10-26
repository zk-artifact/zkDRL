use anyhow::{ensure, Result};
use serde_json::Value;
use crate::{approval::Approval, proof::strict_decode, registry::Program};
use zkdrl_ctc_a2c::Journal as AJ;
use zkdrl_ctc_dqn::Journal as DJ;

pub enum Journal { Dqn(DJ), A2c(AJ) }
impl Journal {
    pub fn decode(p: Program, bytes: &[u8]) -> Result<Self> {
        ensure!(bytes.len() == p.journal_size(), "canonical journal length");
        let journal = match p {
            Program::Dqn => Self::Dqn(strict_decode(bytes)?),
            Program::A2c => Self::A2c(strict_decode(bytes)?),
        };
        ensure!(match &journal {
            Self::Dqn(j) => j.schema == zkdrl_ctc_dqn::SCHEMA,
            Self::A2c(j) => j.schema == zkdrl_ctc_a2c::SCHEMA,
        }, "journal schema");
        Ok(journal)
    }
    pub fn from_json(p: Program, bytes: &[u8]) -> Result<Self> {
        // Never parse untrusted JSON through Value: that collapses duplicates.
        Ok(match p {
            Program::Dqn => Self::Dqn(serde_json::from_slice(bytes)?),
            Program::A2c => Self::A2c(serde_json::from_slice(bytes)?),
        })
    }
    pub fn value(&self) -> Result<Value> {
        Ok(match self { Self::Dqn(j) => serde_json::to_value(j)?, Self::A2c(j) => serde_json::to_value(j)? })
    }
    pub fn encode(&self) -> Result<Vec<u8>> {
        Ok(match self { Self::Dqn(j) => bincode::serialize(j)?, Self::A2c(j) => bincode::serialize(j)? })
    }
    pub fn policy(&self, p: Program, a: &Approval) -> Result<()> {
        a.validate(p)?;
        let slots = 6 * a.episodes;
        match self {
            Self::Dqn(j) => {
                ensure!(p == Program::Dqn && j.schema == zkdrl_ctc_dqn::SCHEMA, "DQN journal schema");
                ensure!(j.config_commit == a.config_commit && j.challenge == a.challenge, "DQN approved config/challenge");
                ensure!(j.kind == a.kind && j.scale == a.scale && j.train == 1, "DQN approved task/scale/train");
                ensure!(j.slots == slots && j.resets == a.episodes, "DQN approved budget/resets");
                counters(j.interactions, j.padded, slots, a.episodes, a.kind)?;
                ensure!(j.updates == j.interactions.saturating_sub(zkdrl_ctc_dqn::WARMUP - 1)
                    && j.target_copies == j.updates / zkdrl_ctc_dqn::TARGET_EVERY
                    && j.replay_wraps == j.interactions.saturating_sub(zkdrl_ctc_dqn::CAP as u32), "DQN learner counters");
            },
            Self::A2c(j) => {
                ensure!(p == Program::A2c && j.schema == zkdrl_ctc_a2c::SCHEMA, "A2C journal schema");
                ensure!(j.config_commit == a.config_commit && j.challenge == a.challenge, "A2C approved config/challenge");
                ensure!(j.kind == a.kind && j.episodes == a.episodes && j.slots == slots
                    && j.updates == a.episodes && j.resets == a.episodes, "A2C approved task/budget/counters");
                counters(j.interactions, j.padded, slots, a.episodes, a.kind)?;
                ensure!(j.successes <= a.episodes && (a.kind != 1 || j.successes == 0), "A2C success counter");
            },
        }
        // return_units and final/initial commitments are authenticated computed
        // outputs. No preknown output or final hash enters the approval policy.
        Ok(())
    }
}
fn counters(interactions: u32, padded: u32, slots: u32, episodes: u32, kind: u32) -> Result<()> {
    ensure!(interactions.checked_add(padded) == Some(slots) && interactions >= episodes, "interaction/padding counters");
    ensure!(kind != 1 || (interactions == slots && padded == 0), "exposure counters");
    Ok(())
}
