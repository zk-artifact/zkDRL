#![no_main]
sp1_zkvm::entrypoint!(main);
use sha2::{Digest, Sha256};
fn hash(bytes: &[u8]) -> [u8; 32] { Sha256::digest(bytes).into() }
pub fn main() {
    // Only initial configuration. No externally supplied interaction/learning log.
    let input = sp1_zkvm::io::read::<zkdrl_profile_dqn_common::PrivateInput>();
    // Every slot, transition and learning update executes inside this program.
    // false suppresses native audit hashing, NOT any environment/learner work.
    let output = zkdrl_profile_dqn_common::run(&input, hash, false);
    sp1_zkvm::io::commit(&output.journal);
}
