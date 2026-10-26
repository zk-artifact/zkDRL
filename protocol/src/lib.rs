//! SHA-256 commit-then-challenge random tape. No externally supplied tape/cache.

/// Callers must supply SHA-256, including in the guest.
pub type Hash = fn(&[u8]) -> [u8; 32];
pub const DQN: u32 = 1;
pub const A2C: u32 = 2;
const KEY_DOMAIN: &[u8] = b"zkDRL/ctc/rng-key/v1\0";
const BLOCK_DOMAIN: &[u8] = b"zkDRL/ctc/rng-block/v1\0";

/// Private derived cache; position is the u64 index of the next u32 word.
/// Construction always starts at zero. This type is never deserialized.
pub struct CounterRng {
    hash: Hash,
    key: [u8; 32],
    position: u64,
    cached_block: Option<u64>,
    block: [u8; 32],
}

impl CounterRng {
    pub fn new(algorithm: u32, rng_seed: [u8; 32], challenge: [u8; 32], hash: Hash) -> Self {
        assert!(algorithm == DQN || algorithm == A2C, "unsupported RNG algorithm");
        let mut input = [0u8; KEY_DOMAIN.len() + 4 + 32 + 32];
        let n = KEY_DOMAIN.len();
        input[..n].copy_from_slice(KEY_DOMAIN);
        input[n..n + 4].copy_from_slice(&algorithm.to_le_bytes());
        input[n + 4..n + 36].copy_from_slice(&rng_seed);
        input[n + 36..].copy_from_slice(&challenge);
        Self { hash, key: hash(&input), position: 0, cached_block: None, block: [0; 32] }
    }

    pub fn position(&self) -> u64 { self.position }

    pub fn next_u32(&mut self) -> u32 {
        // Check BEFORE changing the cache or cursor, including in release mode.
        let next = self.position.checked_add(1).expect("RNG word cursor overflow");
        let j = self.position / 8;
        if self.cached_block != Some(j) {
            let mut input = [0u8; BLOCK_DOMAIN.len() + 32 + 8];
            let n = BLOCK_DOMAIN.len();
            input[..n].copy_from_slice(BLOCK_DOMAIN);
            input[n..n + 32].copy_from_slice(&self.key);
            input[n + 32..].copy_from_slice(&j.to_le_bytes());
            self.block = (self.hash)(&input);
            self.cached_block = Some(j);
        }
        let offset = (self.position % 8) as usize * 4; // in 0..=28
        let word = u32::from_le_bytes(self.block[offset..offset + 4].try_into().unwrap());
        self.position = next;
        word
    }
}

#[cfg(test)]
mod tests;
