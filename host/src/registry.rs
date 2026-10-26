use anyhow::{bail, Result};
use serde_json::{json, Value};

// The formal campaign has five separate DQN builds, without crossed H/C profiles.
// The common core independently supports those crossed profiles.
#[cfg(any(
    all(feature = "h32", feature = "h64"),
    all(feature = "h32", feature = "c128"),
    all(feature = "h32", feature = "c256"),
    all(feature = "h64", feature = "c128"),
    all(feature = "h64", feature = "c256"),
    all(feature = "c128", feature = "c256"),
))]
compile_error!("at most one host profile feature may be enabled: h32, h64, c128, c256");

pub const SDK: &str = "6.9.0";
pub const CIRCUIT: &str = "v6.1.0";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Program { Dqn, A2c }
impl Program {
    pub fn parse(s: &str) -> Result<Self> {
        match s { "dqn" => Ok(Self::Dqn), "a2c" => Ok(Self::A2c), _ => bail!("algorithm must be dqn or a2c") }
    }
    pub fn id(self) -> &'static str { match self { Self::Dqn => "dqn", Self::A2c => "a2c" } }
    pub fn elf(self) -> &'static [u8] {
        match self {
            #[cfg(not(any(feature = "h32", feature = "h64", feature = "c128", feature = "c256")))]
            Self::Dqn => include_bytes!("../../dqn/elf/zkdrl-ctc-dqn-guest"),
            #[cfg(feature = "h32")]
            Self::Dqn => include_bytes!("../../dqn/elf-h32/zkdrl-ctc-dqn-guest"),
            #[cfg(feature = "h64")]
            Self::Dqn => include_bytes!("../../dqn/elf-h64/zkdrl-ctc-dqn-guest"),
            #[cfg(feature = "c128")]
            Self::Dqn => include_bytes!("../../dqn/elf-c128/zkdrl-ctc-dqn-guest"),
            #[cfg(feature = "c256")]
            Self::Dqn => include_bytes!("../../dqn/elf-c256/zkdrl-ctc-dqn-guest"),
            Self::A2c => include_bytes!("../../a2c/elf/zkdrl-ctc-a2c-guest"),
        }
    }
    pub fn width(self) -> u32 {
        match self { Self::Dqn => zkdrl_ctc_dqn::H as u32, Self::A2c => zkdrl_ctc_a2c::H as u32 }
    }
    pub fn capacity(self) -> Option<u32> {
        match self { Self::Dqn => Some(zkdrl_ctc_dqn::CAP as u32), Self::A2c => None }
    }
    pub fn journal_size(self) -> usize { match self { Self::Dqn => 148, Self::A2c => 172 } }
    pub fn profile(self) -> Value { json!({"width":self.width(),"capacity":self.capacity()}) }
}
