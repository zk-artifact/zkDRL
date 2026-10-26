#![no_main]
sp1_zkvm::entrypoint!(main);
use sha2::{Digest, Sha256};
fn hash(bytes: &[u8]) -> [u8; 32] { Sha256::digest(bytes).into() }
pub fn main() {
    // Only starting fixture/configuration, NEVER a supplied trace or result.
    let input = sp1_zkvm::io::read::<zkdrl_ctc_a2c::PrivateInput>();
    // Environment/rewards/reset/termination/RNG/returns/actor+critic SGD all
    // execute here. false suppresses native audit hashes, NOT computation.
    let output = zkdrl_ctc_a2c::run(&input, hash, false);
    sp1_zkvm::io::commit(&output.journal);
}
