//! Complete, bounded, from-initialization integer-master DQN + computational MDP.
//! Feature-selected 4-H-4 ReLU, replay CAP, batch 8, warmup 32, L=16,
//! gamma=1/2, epsilon=1/4, target copy every 32 updates, six-slot episodes.
//! The STE gradient is a DECLARED surrogate, not the derivative of floor.
//! Schema 202 accepts any bounded initial coarse vector; it makes no provenance
//! claim. Only the final state is committed; legacy audit chains are optional.
use serde::{Deserialize, Serialize};

#[cfg(all(feature = "h32", feature = "h64"))]
compile_error!("h32 and h64 are mutually exclusive");
#[cfg(all(feature = "c128", feature = "c256"))]
compile_error!("c128 and c256 are mutually exclusive");

pub const H: usize = if cfg!(feature = "h64") { 64 } else if cfg!(feature = "h32") { 32 } else { 16 };
pub const P: usize = 9 * H + 4;
pub const CAP: usize = if cfg!(feature = "c256") { 256 } else if cfg!(feature = "c128") { 128 } else { 64 };
pub const B: usize = 8;
pub const WARMUP: u32 = 32;
pub const L: i64 = 16;
pub const TARGET_EVERY: u32 = 32;
pub const HORIZON: u32 = 6;
pub const MAX_SLOTS: u32 = 24000;
pub const SCHEMA: u32 = 202;
pub const STATE_BYTES: usize = 16 * P + 88 * CAP + 76;
pub use zkdrl_ctc_protocol::Hash;
use zkdrl_ctc_protocol::{CounterRng, DQN};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrivateInput {
    pub schema: u32,
    pub kind: u32, // 0 = Boolean login, 1 = four-item exposure MDP
    pub scale: u32,
    pub slots: u32,
    pub train: u32,
    /// Initializer metadata, bound in config but not used to draw randomness.
    pub seed: u32,
    /// Independent private seed for the SHA-256 counter tape.
    pub rng_seed: [u8; 32],
    pub initial_coarse: Vec<i64>,
    pub params: [i64; 4],
    pub config_salt: [u8; 32],
    pub final_salt: [u8; 32],
    pub challenge: [u8; 32],
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Journal {
    pub schema: u32,
    pub config_commit: [u8; 32],
    pub final_commit: [u8; 32],
    pub challenge: [u8; 32],
    pub kind: u32,
    pub scale: u32,
    pub slots: u32,
    pub train: u32,
    pub return_units: i64,
    pub interactions: u32,
    pub updates: u32,
    pub resets: u32,
    pub padded: u32,
    pub target_copies: u32,
    pub replay_wraps: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub master: Vec<i64>,
    pub target: Vec<i64>,
    pub buffer: Vec<[i64; 11]>,
    pub interactions: u32,
    pub updates: u32,
    /// Next-word cursor; ten words are consumed per planned slot.
    pub rng: u64,
    pub state: [i64; 4],
    pub done: bool,
    pub total: i64,
    pub resets: u32,
    pub padded: u32,
    pub target_copies: u32,
    pub replay_wraps: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Audit {
    pub slot: u32,
    pub action: u32,
    pub reward: i64,
    pub start_done: bool,
    pub terminal: bool,
    pub state: [i64; 4],
    pub successor: [i64; 4],
    pub replay_indices: [i64; B],
    pub interactions: u32,
    pub updates: u32,
    /// Next-word cursor; ten words are consumed per planned slot.
    pub rng: u64,
    pub state_chain: [u8; 32],
    pub trajectory_chain: [u8; 32],
}

#[derive(Debug, Serialize)]
pub struct Output {
    pub journal: Journal,
    pub final_state: State,
    pub coarse_export: Vec<i64>,
    pub state_chain: Option<[u8; 32]>,
    pub trajectory_chain: Option<[u8; 32]>,
    pub audit: Vec<Audit>, // empty in guest; never a witness input
}

// Fixed-size working memory: no allocation in observe or in an uncaptured slot.
// Coarse caches are derived internally, never deserialized as witness inputs.
struct CoreState {
    master: [i64; P],
    target: [i64; P],
    online: [i64; P],
    target_coarse: [i64; P],
    buffer: [[i64; 11]; CAP],
    interactions: u32,
    updates: u32,
    rng: u64,
    state: [i64; 4],
    done: bool,
    total: i64,
    resets: u32,
    padded: u32,
    target_copies: u32,
    replay_wraps: u32,
}

impl CoreState {
    fn into_state(self) -> State {
        State {
            master: self.master.to_vec(), target: self.target.to_vec(),
            buffer: self.buffer.to_vec(), interactions: self.interactions,
            updates: self.updates, rng: self.rng, state: self.state, done: self.done,
            total: self.total, resets: self.resets, padded: self.padded,
            target_copies: self.target_copies, replay_wraps: self.replay_wraps,
        }
    }

    fn state_bytes(&self) -> Vec<u8> {
        canonical_state_bytes(
            &self.master, &self.target, &self.buffer,
            [self.interactions, self.updates], self.rng, &self.state, self.done,
            self.total, [self.resets, self.padded, self.target_copies, self.replay_wraps],
        )
    }
}

fn u32s(dst: &mut Vec<u8>, values: &[u32]) {
    for v in values { dst.extend_from_slice(&v.to_le_bytes()); }
}
fn i64s(dst: &mut Vec<u8>, values: &[i64]) {
    for v in values { dst.extend_from_slice(&v.to_le_bytes()); }
}
fn add(a: i64, b: i64) -> i64 { a.checked_add(b).expect("addition overflow") }
fn mul(a: i64, b: i64) -> i64 { a.checked_mul(b).expect("multiplication overflow") }
fn s32(x: i64) -> i64 { assert!((-1i64 << 31) <= x && x < (1i64 << 31), "signed-32"); x }
fn abs_bound(x: i64, bits: u32) { assert!(-(1i64 << bits) < x && x < (1i64 << bits), "strict magnitude bound"); }

fn scale_bits(scale: i64) -> u32 {
    match scale { 16 => 4, 256 => 8, _ => panic!("unsupported scale") }
}

fn refresh_coarse(master: &[i64; P], weights: &mut [i64; P], bits: u32) {
    for k in 0..P {
        abs_bound(master[k], 50);
        // Arithmetic right shift is Python floor, including negative residues.
        let w = master[k] >> (7 + bits);
        assert!(-(1 << 20) <= w && w < (1 << 20), "coarse signed-21");
        weights[k] = w;
    }
}

/// Positive divisor: EXACT Python //, including negative non-multiples.
pub fn floor_div(x: i64, d: i64) -> i64 {
    assert!(d > 0);
    x.div_euclid(d)
}

pub fn initial_state(kind: u32) -> [i64; 4] {
    if kind == 0 { [0, 0, 0, 1] } else { [0; 4] }
}

pub fn env_step(kind: u32, state: [i64; 4], action: u32, params: &[i64; 4], scale: i64, done: bool)
    -> ([i64; 4], i64, bool)
{
    assert!(scale == 16 || scale == 256);
    if done { assert_eq!(action, 4); return (state, 0, true); }
    assert!(action < 4);
    let mut s = state;
    if kind == 0 {
        match action {
            0 => s[0] = 1,
            1 => s[0] = 0,
            2 => s[1] = 1,
            3 => { s[2] = 1; return (s, if s[0] != 0 && s[1] != 0 { scale } else { -scale }, true); }
            _ => unreachable!(),
        }
        (s, -(scale >> 4), false)
    } else {
        assert_eq!(kind, 1);
        let a = action as usize;
        let r = params[a].checked_sub(mul(scale >> 2, s[a])).expect("reward overflow");
        s[a] = add(s[a], 1);
        (s, r, false)
    }
}

pub fn coarse(master: &[i64], scale: i64) -> Vec<i64> {
    assert_eq!(master.len(), P);
    let bits = scale_bits(scale);
    master.iter().map(|&m| {
        abs_bound(m, 50);
        let q = m >> (7 + bits);
        assert!(-(1 << 20) <= q && q < (1 << 20), "coarse signed-21");
        q
    }).collect()
}

pub fn forward(weights: &[i64], state: &[i64; 4], scale: i64) -> ([i64; 4], [i64; H]) {
    assert_eq!(weights.len(), P);
    assert!(weights.iter().all(|&w| -(1 << 20) <= w && w < (1 << 20)), "coarse signed-21");
    forward_bounded(weights, state, scale_bits(scale))
}

// Called only with validated coarse caches (or by the validating public wrapper).
// |w| <= 2^20, 0 <= feature <= 6, so |raw| <= 25*2^20.
// Positive activations are strictly below 25*2^20, and each partial dot is
// bounded by H*25*2^40 <= 64*25*2^40 < 2^51. All unchecked i64 adds/products
// below are safe even if the workspace release profile disables overflow checks.
fn forward_bounded(weights: &[i64], state: &[i64; 4], bits: u32) -> ([i64; 4], [i64; H]) {
    assert!(state.iter().all(|&x| (0..=6).contains(&x)), "state domain");
    let mut act = [0; H];
    for j in 0..H {
        let mut raw = weights[4 * H + j];
        for i in 0..4 { raw += weights[4 * j + i] * state[i]; }
        act[j] = s32(raw).max(0);
    }
    let mut q = [0; 4];
    for a in 0..4 {
        let mut dot = 0;
        for j in 0..H { dot += weights[5 * H + a * H + j] * act[j]; }
        q[a] = s32(s32(dot >> bits) + weights[9 * H + a]);
    }
    (q, act)
}

fn argmax(q: &[i64; 4]) -> u32 {
    let mut best = 0;
    for a in 1..4 { if q[a] > q[best] { best = a; } }
    best as u32 // earliest index wins ties, as in NumPy
}

fn observe(st: &mut CoreState, row: [i64; 11], draws: &[u32; B], bits: u32) -> [i64; B] {
    // All rows originate here, AFTER a checked in-guest environment transition.
    let at = st.interactions as usize % CAP;
    if st.interactions >= CAP as u32 { st.replay_wraps += 1; }
    st.buffer[at] = row;
    st.interactions += 1;
    if st.interactions < WARMUP { return [-1; B]; }
    let count = st.interactions.min(CAP as u32);
    let indices = draws.map(|x| ((x as u64 * count as u64) >> 32) as i64);
    let qw = &st.online;
    let target = &st.target_coarse;
    let mut grad = [0; P];
    for &index in &indices {
        assert!(index >= 0 && index < count as i64);
        let r = st.buffer[index as usize];
        let s: [i64; 4] = r[..4].try_into().unwrap();
        let ns: [i64; 4] = r[6..10].try_into().unwrap();
        assert!((0..4).contains(&r[4]) && (r[10] == 0 || r[10] == 1));
        let a = r[4] as usize;
        let (q, act) = forward_bounded(qw, &s, bits);
        let (future, _) = forward_bounded(target, &ns, bits);
        let y = add(r[5], if r[10] != 0 { 0 } else { *future.iter().max().unwrap() >> 1 });
        let err = s32(q[a].checked_sub(y).expect("TD subtraction overflow"));
        for j in 0..H {
            // signed32 err times coarse is at most 2^51 in magnitude;
            // err times activation is strictly below 25*2^51 < 2^56.
            let dh = if act[j] > 0 { err * qw[5 * H + a * H + j] } else { 0 };
            let g2 = err * act[j];
            abs_bound(dh, 46);
            abs_bound(g2, 50);
            // Preserve the Python reference's additional diagnostic domain checks.
            s32(dh >> bits); // mask BEFORE the signed32 diagnostic, as Python does
            s32(g2 >> bits);
            // Bounds above precede accumulation: eight samples yield
            // |g1| < 8*6*2^46 < 2^52, |gb1| < 2^49, |g2sum| < 2^53.
            // Partial sums obey the same bounds even if final terms cancel.
            for i in 0..4 { grad[4 * j + i] += dh * s[i]; }
            grad[4 * H + j] += dh;
            let k = 5 * H + a * H + j;
            grad[k] += g2;
        }
        // Eight signed32 errors times S <= 256 have magnitude <= 2^42.
        grad[9 * H + a] += (1i64 << bits) * err;
    }
    for (m, g) in st.master.iter_mut().zip(grad) {
        abs_bound(g, 50);
        *m = m.checked_sub(g).expect("master subtraction overflow");
    }
    // Enforce strict fine-master and signed21 coarse bounds after EVERY update.
    refresh_coarse(&st.master, &mut st.online, bits);
    st.updates += 1;
    if st.updates % TARGET_EVERY == 0 {
        st.target = st.master; // copies fine masters, NOT rounded weights
        st.target_coarse = st.online;
        st.target_copies += 1;
    }
    indices
}

/// Canonical bytes independent of serde/Rust layouts, all integers LE.
pub fn state_bytes(st: &State) -> Vec<u8> {
    canonical_state_bytes(
        &st.master, &st.target, &st.buffer,
        [st.interactions, st.updates], st.rng, &st.state, st.done,
        st.total, [st.resets, st.padded, st.target_copies, st.replay_wraps],
    )
}

fn canonical_state_bytes(
    master: &[i64], target: &[i64], buffer: &[[i64; 11]],
    counters: [u32; 2], rng: u64, state: &[i64; 4], done: bool, total: i64, stats: [u32; 4],
) -> Vec<u8> {
    assert_eq!(master.len(), P);
    assert_eq!(target.len(), P);
    assert_eq!(buffer.len(), CAP);
    let mut b = Vec::with_capacity(STATE_BYTES);
    i64s(&mut b, master);
    i64s(&mut b, target);
    for r in buffer { i64s(&mut b, r); }
    u32s(&mut b, &counters);
    b.extend_from_slice(&rng.to_le_bytes());
    i64s(&mut b, state);
    u32s(&mut b, &[done as u32]);
    i64s(&mut b, &[total]);
    u32s(&mut b, &stats);
    assert_eq!(b.len(), STATE_BYTES);
    b
}

pub fn config_bytes(input: &PrivateInput) -> Vec<u8> {
    let mut b = b"zkDRL/ctc/dqn/config/v3\0".to_vec();
    u32s(&mut b, &[input.schema, input.kind, input.scale, input.slots, input.train,
        H as u32, CAP as u32, B as u32, WARMUP, L as u32, TARGET_EVERY, HORIZON,
        4, 1, 2, input.seed]);
    b.extend_from_slice(&input.rng_seed);
    i64s(&mut b, &input.initial_coarse);
    i64s(&mut b, &input.params);
    b.extend_from_slice(&input.config_salt);
    b.extend_from_slice(&input.final_salt);
    b
}

pub fn run(input: &PrivateInput, hash: Hash, capture: bool) -> Output {
    assert_eq!(input.schema, SCHEMA);
    assert!(input.kind <= 1 && input.train <= 1);
    assert!(input.scale == 16 || input.scale == 256);
    assert!(input.slots >= HORIZON && input.slots <= MAX_SLOTS && input.slots % HORIZON == 0);
    assert_eq!(input.initial_coarse.len(), P);
    assert!(input.initial_coarse.iter().all(|&x| -(1 << 20) <= x && x < (1 << 20)));
    let scale = input.scale as i64;
    let bits = scale_bits(scale);
    assert!(input.params.iter().all(|&x| -16 * scale <= x && x <= 16 * scale));
    if input.kind == 0 { assert_eq!(input.params, [0; 4]); }
    let config_commit = hash(&config_bytes(input));
    let mut rng = CounterRng::new(DQN, input.rng_seed, input.challenge, hash);
    let mut master = [0; P];
    for (m, &w) in master.iter_mut().zip(&input.initial_coarse) {
        // Validated signed21 w times 128*S <= 2^15: |m| <= 2^35.
        *m = w * (1i64 << (7 + bits));
    }
    let mut online = [0; P];
    refresh_coarse(&master, &mut online, bits);
    let mut st = CoreState {
        target: master, master, online, target_coarse: online,
        buffer: [[0; 11]; CAP], interactions: 0,
        updates: 0, rng: 0, state: initial_state(input.kind), done: false, total: 0,
        resets: 0, padded: 0, target_copies: 0, replay_wraps: 0,
    };
    let mut state_chain = [0; 32];
    let mut trajectory_chain = [0; 32];
    let mut audit = Vec::with_capacity(if capture { input.slots as usize } else { 0 });
    // Counters increment at most once per slot: all remain <= 24000.
    // Resets <= 4000, target copies <= 749; all u32 arithmetic is bounded.
    for slot in 0..input.slots {
        if slot % HORIZON == 0 {
            st.state = initial_state(input.kind);
            st.done = false;
            st.resets += 1;
        }
        // A fixed ten draws per slot, even in padding/evaluation. No host RNG.
        let explore = rng.next_u32() & 15;
        let random_action = rng.next_u32() & 3;
        let mut replay_draws = [0; B];
        for x in &mut replay_draws { *x = rng.next_u32(); }
        st.rng = rng.position();
        let (q, _) = forward_bounded(&st.online, &st.state, bits);
        let done_before = st.done;
        let before = st.state;
        let action = if done_before { 4 } else if input.train != 0 && explore < 4 { random_action } else { argmax(&q) };
        let (successor, reward, terminal) = env_step(input.kind, before, action, &input.params, scale, done_before);
        assert!(successor.iter().all(|&x| (0..=6).contains(&x)));
        let mut indices = [-1; B];
        if !done_before && input.train != 0 {
            let row = [before[0], before[1], before[2], before[3], action as i64, reward,
                successor[0], successor[1], successor[2], successor[3], terminal as i64];
            indices = observe(&mut st, row, &replay_draws, bits);
        }
        if done_before { st.padded += 1; }
        st.total = add(st.total, reward);
        st.state = successor;
        st.done = terminal;
        // Native comparison only. The guest skips all per-slot serialization
        // and both legacy audit hashes; neither chain enters the final commit.
        if capture {
            let mut trace = b"zkRL/risc0/trajectory/v1\0".to_vec();
            trace.extend_from_slice(&trajectory_chain);
            i64s(&mut trace, &[slot as i64, slot as i64 / 6, slot as i64 % 6]);
            i64s(&mut trace, &before);
            i64s(&mut trace, &[done_before as i64, action as i64, reward]);
            i64s(&mut trace, &successor);
            i64s(&mut trace, &[terminal as i64]);
            i64s(&mut trace, &indices);
            i64s(&mut trace, &[st.interactions as i64, st.updates as i64]);
            trace.extend_from_slice(&st.rng.to_le_bytes());
            trajectory_chain = hash(&trace);
            let mut chain = b"zkRL/risc0/state-chain/v1\0".to_vec();
            chain.extend_from_slice(&state_chain);
            chain.extend_from_slice(&st.state_bytes());
            state_chain = hash(&chain);
            audit.push(Audit {slot, action, reward, start_done: done_before, terminal,
                state: before, successor, replay_indices: indices, interactions: st.interactions,
                updates: st.updates, rng: st.rng, state_chain, trajectory_chain});
        }
    }
    let coarse_export = st.online.to_vec();
    let mut final_bytes = Vec::with_capacity(STATE_BYTES + 2 * P * 8 + 128);
    final_bytes.extend_from_slice(b"zkDRL/ctc/dqn/final/v3\0");
    final_bytes.extend_from_slice(&config_commit);
    final_bytes.extend_from_slice(&input.challenge);
    final_bytes.extend_from_slice(&input.final_salt);
    final_bytes.extend_from_slice(&st.state_bytes());
    i64s(&mut final_bytes, &coarse_export);
    i64s(&mut final_bytes, &st.target_coarse);
    let journal = Journal { schema: SCHEMA, config_commit, final_commit: hash(&final_bytes),
        challenge: input.challenge, kind: input.kind, scale: input.scale, slots: input.slots,
        train: input.train, return_units: st.total, interactions: st.interactions, updates: st.updates,
        resets: st.resets, padded: st.padded, target_copies: st.target_copies, replay_wraps: st.replay_wraps };
    Output { journal, final_state: st.into_state(), coarse_export,
        state_chain: capture.then_some(state_chain),
        trajectory_chain: capture.then_some(trajectory_chain), audit }
}

#[cfg(test)]
mod tests;
