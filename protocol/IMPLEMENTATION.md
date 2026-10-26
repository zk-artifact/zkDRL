# Protocol, randomness, and guest encodings

Only shared Rust RNG and learner common/guest crates are implemented here. Each learner is a workspace with `common` and `guest`; no host or external tape input is provided. `Hash = fn(&[u8]) -> [u8; 32]` must be SHA-256; both guests pass `Sha256::digest`. SP1 is exactly 6.9.0; learner workspaces pin SHA-256 patch commit `75b15faa7ce44d25435d0a3209285455540f0b7f` (the old guest lockfiles' `patch-sha2-0.10.8-sp1-6.2.0`). Release uses opt-level 3, fat LTO, one codegen unit, overflow checks off with the old explicit arithmetic guards retained.

All canonical encodings below concatenate fields without lengths, separators, alignment, or serde framing. `U32`, `U64`, and `I64` mean little-endian fixed-width integers; signed integers use two's complement. Arrays concatenate elements. Every quoted tag includes its final NUL byte. Digests, seeds, salts, and challenges are raw 32-byte values. `kappa = SHA256(config_bytes)` is the journal's `config_commit`.

## Random tape

```
key     = SHA256(b"zkDRL/ctc/rng-key/v1\0" || U32(algorithm) || rng_seed || challenge)
block_j = SHA256(b"zkDRL/ctc/rng-block/v1\0" || key || U64(j))
word_n  = LE_u32(block_(n//8)[4*(n%8) : 4*(n%8)+4])
```

Algorithm 1 is DQN; 2 is A2C. `CounterRng::new(algorithm, rng_seed, challenge, hash)` starts at next-word cursor zero. `next_u32()` checks cursor + 1 before producing a word; `position()` returns the next-word cursor. Exhaustion at `u64::MAX` panics without advancing. Key and cached block are private derived data, never deserialized. No initialization metadata seed, kappa, budget, width, precision, salts, or LUT enters either RNG domain. Paired runs share `rng_seed` and `challenge` per algorithm, so their word prefixes agree. The caller supplies an independent private 32-byte `rng_seed`; it is never derived from the old metadata `seed`.

DQN draws ten words per planned slot: exploration `word & 15` (explore iff training and <4), exploratory action `word & 3`, then eight replay words. Replay indices are `(U64(word)*count)>>32`, where `count=min(interactions_after_insert,CAP)`. All ten words are consumed during evaluation, warmup, and terminal padding; final cursor = `10*slots`. A2C consumes one full u32 per planned slot, sampled by the unchanged integer CDF; padding consumes it too. Final cursor = `6*episodes`. Each run constructs its tape locally, with no continuation witness.

Known vectors for all-zero seed and challenge (three blocks and patterned seed/challenge cases are frozen in `src/tests.rs`):

- Algorithm 1: key `77b5e977b322ab6eb46aede05902cb267946924466fa4facf35bf767ab226e50`; block 0 `f592b44132d1f0c83bf39edcd497fc21aa42286f02d126fd4f2231d8830c6891`.
- Algorithm 2: key `8dab15bd59e9f8efff9de485144fd1c8a7e95038a51604cbd3857708bb49a10d`; block 0 `4c5e77710e6d355fe58b9b402dfa94cfcf0720a5d8cdd199a1aca5e26b12bb0b`.


## DQN (schema 202)

`H=16` by default, `h32` selects 32, `h64` selects 64. `CAP=64` by default, `c128` selects 128, `c256` selects 256. Width features mutually exclude each other, as do capacity features. All nine valid combinations use one source; guest features forward to common. `P=9*H+4`, batch 8, warmup 32, L=16, target copy interval 32, horizon 6. All old integer math, guards, and environment rules are retained (the forward dot-product safety bound now accounts for H<=64).

Config, exactly `216+8*P` bytes (1400 at default width):

```
b"zkDRL/ctc/dqn/config/v3\0"
U32[202, kind, scale, slots, train, H, CAP, 8, 32, 16, 32, 6, 4, 1, 2, seed]
rng_seed
I64[P] initial_coarse
I64[4] params
config_salt || final_salt
```

Persistent state, exactly `16*P+88*CAP+76` bytes (8076 by default):

```
I64[P] master || I64[P] target
I64[CAP][11] replay buffer (physical ring order, including unused zero rows)
U32 interactions || U32 updates || U64 rng_next_word
I64[4] state || U32 done || I64 total
U32[4] [resets, padded, target_copies, replay_wraps]
```

Each replay row is `[before[0..4], action, reward, successor[0..4], terminal]`, all I64. Boolean values in canonical encodings are 0 or 1. The cursor starts at zero and is copied into CoreState/State/Audit after the slot's ten draws.

```
final_commit = SHA256(b"zkDRL/ctc/dqn/final/v3\0" || kappa || challenge || final_salt
                     || canonical_state || I64[P] online_coarse || I64[P] target_coarse)
```

Coarse exports are unchanged arithmetic shifts of fine masters by `7+log2(scale)`. The final preimage has `STATE_BYTES+16*P+119` bytes. The public journal stays 148 bytes, in this order:

```
U32 schema || config_commit || final_commit || challenge
U32[4] [kind, scale, slots, train] || I64 return_units
U32[6] [interactions, updates, resets, padded, target_copies, replay_wraps]
```

Optional native capture retains one Audit per planned slot, including padding. Both chains start at 32 zero bytes; neither enters any commitment. After the environment/learning work for each slot:

```
trajectory_chain = SHA256(b"zkRL/risc0/trajectory/v1\0" || previous_trajectory_chain
    || I64[3] [slot, slot//6, slot%6] || I64[4] before
    || I64[3] [start_done, action, reward] || I64[4] successor || I64 terminal
    || I64[8] replay_indices || I64[2] [interactions, updates] || U64 rng_next_word)
state_chain = SHA256(b"zkRL/risc0/state-chain/v1\0" || previous_state_chain || canonical_state)
```

Trajectory preimages are 265 bytes; state-chain preimages are `58+STATE_BYTES` bytes. Replay indices are all -1 unless a training update occurs. Audit object order: `slot:u32, action:u32, reward:i64, start_done:bool, terminal:bool, state:[i64;4], successor:[i64;4], replay_indices:[i64;8], interactions:u32, updates:u32, rng:u64, state_chain:[u8;32], trajectory_chain:[u8;32]`. The cursor is encoded as U64, never cast to i64. Quiet runs have no audits or chain hashes; they perform `3+ceil(10*slots/8)` total SHA-256 calls, versus that count plus `2*slots` when captured.

## A2C (schema 203)

The old Q16 forward/Q40 master/Q32 mass/Q48 LUT arithmetic and all guards remain. `exp_table.rs` is byte-for-byte unchanged. Config, exactly 2188 bytes:

```
b"zkDRL/ctc/a2c/config/v2\0"
U32[17] [203, kind, episodes, 16, 40, 32, 48, 16, 6, 7, 8, 1, 32, 64, 128, 256, 32]
U64 seed || rng_seed
I64[148] initial_actor || I64[97] initial_critic
config_salt || final_salt || TABLE_SHA
```

`TABLE_SHA` is the old frozen JSON table digest `39786b3a4a9ffc46842ae90d55beedaba84f1156cfd365e7eeb8b3f43b0f348d`, not the Rust source file digest. Challenge is absent from config. Initial and final commitments are:

```
initial_commit = SHA256(b"zkDRL/ctc/a2c/initial/v2\0" || kappa || config_salt
                       || I64[148] initial_actor || I64[97] initial_critic)
final_commit   = SHA256(b"zkDRL/ctc/a2c/final/v2\0" || kappa || challenge || final_salt
                       || canonical_state)
```

Initial preimage length is 2049; final is 2155. Persistent state remains 2036 bytes:

```
I64[148] actor || I64[97] critic || U64 rng_next_word
I64[4] state || U32 done || I64 total
U32[6] [episodes, updates, interactions, resets, padded, successes]
```

The cursor is at byte offset 1960, state at 1968, done at 2000, total at 2004, counters at 2012. Journal remains 172 bytes:

```
U32 schema || config_commit || initial_commit || final_commit || challenge
U32[8] [kind, episodes, slots, interactions, updates, resets, padded, successes]
I64 return_units
```

Capture retains one Audit per episode. Its `trace` has six `[i64;16]` rows:
`[episode, t, before[0..4], action, reward, successor[0..4], previous_done, terminal, updates_before_episode_update, full_u32_draw]`. The last field is the sampled word, not the cursor (every u32 fits i64). Padding uses action 4, zero reward, unchanged terminal state, and a fresh draw. Audit fields are `episode:u32, trace:Vec<[i64;16]>, model_sha256:[u8;32], stages_sha256:[u8;32]`.

```
model_sha256 = SHA256(b"zkRL/A2C-fixed/state/v1\0" || U32(16) || U64(updates_after_update)
                     || I64[148] actor_after_update || I64[97] critic_after_update)
stages_sha256 = SHA256(b"zkRL/A2C-SP1/stages/v1\0" || U32(n_active)
                      || active_row_stages_in_order || I64[148] ga || I64[97] gc)
```

Each active row's stages concatenate: actor logits I64[4], critic output I64[1], masses U64[4], target I64, advantage I64, actor upstream I64[4], critic upstream I64[1], actor preactivations I64[16], actor hidden I64[16], critic preactivations I64[16], critic hidden I64[16]. That is 640 bytes per active row. Model preimages are 1996 bytes; stage preimages are `1987+640*n_active` bytes. Gradients are accumulated before the unchanged atomic two-network update; model hashes use post-update masters. Quiet runs omit all audit allocation/hashing and make `4+ceil(6*episodes/8)` SHA-256 calls; captured runs add `2*episodes`. Audit data is never a witness or final-commit component.

## Serialization and checks for the parent

PrivateInput adds `rng_seed:[u8;32]` immediately after the metadata `seed`, retaining every old field. DQN field order: `schema, kind, scale, slots, train, seed, rng_seed, initial_coarse, params, config_salt, final_salt, challenge`. A2C: `schema, kind, episodes, seed, rng_seed, initial_actor, initial_critic, config_salt, final_salt, challenge`. All deserializable structs reject unknown named fields. For ordinary bincode 1.3 fixed-width serialization, vectors have U64 length prefixes and bools have one byte; canonical hash encodings above deliberately have neither vector prefixes nor one-byte bools. Journals have no vectors/bools, so their fixed-width bincode bytes are exactly the listed journal layouts. SP1 input transport remains the old `io::read::<PrivateInput>()`; the parent must use its transport framing.

Rust tests cover frozen independent RNG vectors, cache boundaries/exhaustion, repeatability and separation, commitment/config encodings and field sensitivity, terminal-padding cursors, raw draw/replay mapping, metadata/budget-independent tape prefixes, audit encodings/capture equality, journal ABI, and preserved numerical boundary checks.
