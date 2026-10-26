//! Bounded Q16-forward/Q40-master integer A2C, matching A2C-FX v1.
//!
//! Invalid inputs or rejected numerical bounds panic; there is no clipping.
//! Rollouts and training arithmetic use fixed arrays. Vec is used only at the
//! input, output, and serialization/capture boundaries. All signed divisions
//! are floors. Randomness is a derived SHA-256 counter tape.

use serde::{Deserialize, Serialize};

mod exp_table;

pub const S: i64 = 1 << 16;
pub const G: i64 = 1 << 40;
pub const P: u64 = 1 << 32;
pub const H: usize = 16;
pub const ACTOR: usize = 148;
pub const CRITIC: usize = 97;
pub const MAX_EPISODES: usize = 4000;
pub const STATE_BYTES: usize = 2036;
pub const SCHEMA: u32 = 203;
pub use zkdrl_ctc_protocol::Hash;
use zkdrl_ctc_protocol::{CounterRng, A2C};

const D: usize = 6;
const RATIO: i64 = G / S;
const MASTER_LIMIT: i64 = 256 * G;
const PARAM_LIMIT: i64 = 256 * S;
const FORWARD_LIMIT: i64 = 1024 * S;
const GRAD_LIMIT: i64 = 65536 * G;
const TABLE_SHA: [u8; 32] = [
    0x39, 0x78, 0x6b, 0x3a, 0x4a, 0x9f, 0xfc, 0x46,
    0x84, 0x2a, 0xe9, 0x0d, 0x55, 0xbe, 0xed, 0xab,
    0xa8, 0x4f, 0x11, 0x56, 0xcf, 0xd3, 0x65, 0xe7,
    0xee, 0xb8, 0xb3, 0xf4, 0x3b, 0x0f, 0x34, 0x8d,
];

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PrivateInput {
    pub schema: u32,
    pub kind: u32,
    pub episodes: u32,
    /// Initializer metadata, bound in config but not used to draw randomness.
    pub seed: u64,
    /// Independent private seed for the SHA-256 counter tape.
    pub rng_seed: [u8; 32],
    pub initial_actor: Vec<i64>,
    pub initial_critic: Vec<i64>,
    pub config_salt: [u8; 32],
    pub final_salt: [u8; 32],
    pub challenge: [u8; 32],
}

/// Field order is part of the public journal ABI (172 fixed-width bytes).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Journal {
    pub schema: u32,
    pub config_commit: [u8; 32],
    pub initial_commit: [u8; 32],
    pub final_commit: [u8; 32],
    pub challenge: [u8; 32],
    pub kind: u32,
    pub episodes: u32,
    pub slots: u32,
    pub interactions: u32,
    pub updates: u32,
    pub resets: u32,
    pub padded: u32,
    pub successes: u32,
    pub return_units: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub actor: Vec<i64>,
    pub critic: Vec<i64>,
    /// Next-word cursor; one word is consumed per planned slot, including padding.
    pub rng: u64,
    pub state: [i64; 4],
    pub done: bool,
    pub total: i64,
    pub episodes: u32,
    pub updates: u32,
    pub interactions: u32,
    pub resets: u32,
    pub padded: u32,
    pub successes: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Audit {
    /// Zero-based episode index, as in collect_fixed.
    pub episode: u32,
    pub trace: Vec<[i64; 16]>,
    pub model_sha256: [u8; 32],
    pub stages_sha256: [u8; 32],
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Output {
    pub journal: Journal,
    pub final_state: State,
    /// Final quantized Q16 parameters, in the same flat layout as the masters.
    pub actor_export: Vec<i64>,
    pub critic_export: Vec<i64>,
    pub audit: Vec<Audit>,
}

fn bounded(value: i128, limit: i64, name: &str) -> i64 {
    // Comparing signed endpoints also rejects i128::MIN without abs overflow.
    assert!(value > -(limit as i128) && value < limit as i128, "{name}: strict bound");
    value as i64
}

pub fn floor_div(value: i128, positive_divisor: i128) -> i128 {
    assert!(positive_divisor > 0, "positive divisor required");
    value.div_euclid(positive_divisor)
}

fn validate_masters(values: &[i64]) {
    for &v in values {
        bounded(v as i128, MASTER_LIMIT, "master");
        bounded((v >> 24) as i128, PARAM_LIMIT, "quantized master");
    }
}

fn validate_input(input: &PrivateInput) {
    assert_eq!(input.schema, SCHEMA, "schema");
    assert!(input.kind <= 1, "kind");
    assert!((1..=MAX_EPISODES as u32).contains(&input.episodes), "episodes");
    assert_eq!(input.initial_actor.len(), ACTOR, "actor length");
    assert_eq!(input.initial_critic.len(), CRITIC, "critic length");
    validate_masters(&input.initial_actor);
    validate_masters(&input.initial_critic);
}

pub fn quantize<const N: usize>(masters: &[i64; N]) -> [i64; N] {
    validate_masters(masters);
    masters.map(|v| v >> 24)
}

pub fn exp_weight(shifted: i64) -> u64 {
    assert!(shifted <= 0, "nonpositive shifted logit required");
    let q = -(shifted as i128);
    if q > 32 * S as i128 { return 0; }
    if q == 32 * S as i128 { return exp_table::EXP[8192]; }
    let k = (q / 256) as usize;
    let r = (q % 256) as u64;
    // Both table values <=2^48; interpolation numerator <=2^56.
    ((256 - r) * exp_table::EXP[k] + r * exp_table::EXP[k + 1]) / 256
}

pub fn masses(logits: &[i64; 4]) -> [u64; 4] {
    for &z in logits { bounded(z as i128, FORWARD_LIMIT, "logits"); }
    let peak = *logits.iter().max().unwrap();
    let weights = logits.map(|z| exp_weight(z - peak));
    let total: u64 = weights.iter().sum(); // <=4*2^48; at least one weight is 2^48.
    assert!(total > 0);
    let mut result = [0; 4];
    let mut remainders = [0u128; 4];
    for i in 0..4 {
        let numerator = P as u128 * weights[i] as u128; // <=2^80.
        result[i] = (numerator / total as u128) as u64;
        remainders[i] = numerator % total as u128;
    }
    let residual = P - result.iter().sum::<u64>();
    assert!(residual < 4);
    let mut order = [0usize, 1, 2, 3];
    // Fixed four-entry stable insertion sort: lower indices win ties.
    for i in 1..4 {
        let mut j = i;
        while j > 0 && remainders[order[j]] > remainders[order[j - 1]] {
            order.swap(j, j - 1);
            j -= 1;
        }
    }
    for i in 0..residual as usize { result[order[i]] += 1; }
    validate_mass(&result);
    for i in 0..4 { assert!(weights[i] != 0 || result[i] == 0); }
    result
}

fn validate_mass(mass: &[u64; 4]) {
    for &v in mass { assert!(v <= P, "mass bound"); }
    assert_eq!(mass.iter().sum::<u64>(), P, "mass sum");
}

pub fn categorical(mass: &[u64; 4], draw: u32) -> usize {
    validate_mass(mass);
    let mut cumulative = 0;
    for (i, &v) in mass.iter().enumerate() {
        cumulative += v;
        if (draw as u64) < cumulative { return i; }
    }
    unreachable!("validated CDF covers every u32")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Forward<const O: usize> {
    pub state: [i64; 4],
    pub pre: [i64; H],
    pub hidden: [i64; H],
    pub output: [i64; O],
}

impl<const O: usize> Forward<O> {
    const ZERO: Self = Self { state: [0; 4], pre: [0; H], hidden: [0; H], output: [0; O] };
}

fn validate_params<const O: usize, const N: usize>(params: &[i64; N]) {
    assert!(O == 1 || O == 4, "network outputs");
    assert_eq!(N, 80 + 17 * O, "network layout");
    for &v in params { bounded(v as i128, PARAM_LIMIT, "forward parameter"); }
}

fn validate_state(state: &[i64; 4]) {
    assert!(state.iter().all(|&v| (0..=6).contains(&v)), "state domain");
}

pub fn forward<const O: usize, const N: usize>(params: &[i64; N], state: [i64; 4]) -> Forward<O> {
    validate_params::<O, N>(params);
    validate_state(&state);
    let mut cache = Forward { state, ..Forward::ZERO };
    for j in 0..H {
        // Four terms each <6*2^24 plus bias: every partial sum fits i64.
        let mut pre = params[64 + j];
        for k in 0..4 { pre += params[4 * j + k] * state[k]; }
        cache.pre[j] = bounded(pre as i128, FORWARD_LIMIT, "preactivation");
        cache.hidden[j] = pre.max(0);
    }
    for o in 0..O {
        // 16 terms, each <2^24*2^26: absolute partial sum <2^54.
        let mut dot = 0i64;
        for j in 0..H { dot += params[80 + 16 * o + j] * cache.hidden[j]; }
        cache.output[o] = bounded(((dot >> 16) + params[80 + 16 * O + o]) as i128, FORWARD_LIMIT, "output");
    }
    cache
}

/// Sum rows before flooring output-weight gradients; mask dh after its floor.
pub fn backward<const O: usize, const N: usize>(
    params: &[i64; N], caches: &[Forward<O>], upstream: &[[i64; O]],
) -> [i64; N] {
    validate_params::<O, N>(params);
    assert!((1..=D).contains(&caches.len()), "active row count");
    assert_eq!(caches.len(), upstream.len());
    for (cache, dy) in caches.iter().zip(upstream) {
        validate_state(&cache.state);
        for j in 0..H {
            bounded(cache.pre[j] as i128, FORWARD_LIMIT, "cached preactivation");
            assert_eq!(cache.hidden[j], cache.pre[j].max(0), "cached ReLU");
        }
        for &v in dy { bounded(v as i128, GRAD_LIMIT, "upstream"); }
    }
    let mut dh = [[0i64; H]; D];
    for t in 0..caches.len() {
        for j in 0..H {
            // <=4 terms <2^56*2^24: every partial sum <2^82.
            let mut dot = 0i128;
            for o in 0..O { dot += upstream[t][o] as i128 * params[80 + 16 * o + j] as i128; }
            let raw = dot >> 16;
            let masked = if caches[t].pre[j] > 0 { raw } else { 0 };
            dh[t][j] = bounded(masked, GRAD_LIMIT, "hidden VJP");
        }
    }
    let mut grad = [0; N];
    for j in 0..H {
        for k in 0..4 {
            let mut sum = 0i64;
            // Six terms <6*2^56; partial sums <36*2^56 <2^62.
            for t in 0..caches.len() { sum += dh[t][j] * caches[t].state[k]; }
            grad[4 * j + k] = bounded(sum as i128, GRAD_LIMIT, "gradient W1");
        }
        let mut sum = 0i64;
        for row in &dh[..caches.len()] { sum += row[j]; }
        grad[64 + j] = bounded(sum as i128, GRAD_LIMIT, "gradient b1");
    }
    for o in 0..O {
        for j in 0..H {
            let mut sum = 0i128;
            // Six terms <2^56*2^26: partial sums <2^85, before floor.
            for t in 0..caches.len() { sum += upstream[t][o] as i128 * caches[t].hidden[j] as i128; }
            grad[80 + 16 * o + j] = bounded(sum >> 16, GRAD_LIMIT, "gradient W2");
        }
        let mut sum = 0i64;
        for dy in upstream { sum += dy[o]; } // <6*2^56.
        grad[80 + 16 * O + o] = bounded(sum as i128, GRAD_LIMIT, "gradient b2");
    }
    grad
}

pub fn actor_upstream(logits: &[i64; 4], mass: &[u64; 4], action: usize, advantage: i64) -> [i64; 4] {
    assert!(action < 4, "action");
    validate_mass(mass);
    bounded(advantage as i128, GRAD_LIMIT, "advantage");
    for &z in logits { bounded(z as i128, FORWARD_LIMIT, "logits"); }
    let peak = *logits.iter().max().unwrap();
    let shifted = logits.map(|z| z - peak); // strict absolute bound 2^27.
    let mut moment = 0i128;
    for i in 0..4 { moment += mass[i] as i128 * shifted[i] as i128; }
    let mut dy = [0; 4];
    for i in 0..4 {
        let delta = mass[i] as i128 - if i == action { P as i128 } else { 0 };
        let policy = (advantage as i128 * delta) >> 32;
        // Cancel the exact 2^24 factor BEFORE multiplication: original shift
        // is 71, so the canceled shift is 47. Numerator <2^91 (i128 safe).
        let entropy = (mass[i] as i128 * (shifted[i] as i128 * P as i128 - moment)) >> 47;
        dy[i] = bounded(policy + entropy, GRAD_LIMIT, "actor upstream");
    }
    dy
}

pub fn finite_returns(rewards: &[i64], terminals: &[bool], trace_continue: &[bool]) -> [i64; D] {
    assert!((1..=D).contains(&rewards.len()), "active row count");
    assert_eq!(rewards.len(), terminals.len());
    assert_eq!(rewards.len(), trace_continue.len());
    assert!(!trace_continue[rewards.len() - 1], "last trace must cut");
    for i in 0..rewards.len() {
        bounded(rewards[i] as i128, 4 * S, "reward");
        assert!(!(terminals[i] && trace_continue[i]), "terminal trace must cut");
    }
    let mut result = [0; D];
    let mut following = 0i64;
    for i in (0..rewards.len()).rev() {
        let discounted = if trace_continue[i] { (7 * following as i128) >> 3 } else { 0 };
        following = bounded(rewards[i] as i128 * RATIO as i128 + discounted, GRAD_LIMIT, "target");
        result[i] = following;
    }
    result
}

/// Compute and validate BOTH candidates before changing either network.
pub fn apply_update(actor: &mut [i64; ACTOR], critic: &mut [i64; CRITIC], ga: &[i64; ACTOR], gc: &[i64; CRITIC]) {
    validate_masters(actor);
    validate_masters(critic);
    fn candidate<const N: usize>(old: &[i64; N], grad: &[i64; N], denominator: i128) -> [i64; N] {
        let mut result = [0; N];
        for i in 0..N {
            bounded(grad[i] as i128, GRAD_LIMIT, "gradient");
            result[i] = bounded(old[i] as i128 - floor_div(grad[i] as i128, denominator), MASTER_LIMIT, "candidate master");
        }
        validate_masters(&result); // Includes the asymmetric negative Q16 edge.
        result
    }
    let next_actor = candidate(actor, ga, 192);
    let next_critic = candidate(critic, gc, 384);
    *actor = next_actor;
    *critic = next_critic;
}

fn transition(kind: u32, state: [i64; 4], action: usize, done: bool) -> ([i64; 4], i64, bool) {
    assert!(kind <= 1);
    validate_state(&state);
    if done { return (state, 0, true); }
    assert!(action < 4);
    let mut next = state;
    let (reward, terminal) = if kind == 0 {
        match action {
            0 => next[0] = 1,
            1 => next[0] = 0,
            2 => next[1] = 1,
            3 => next[2] = 1,
            _ => unreachable!(),
        }
        if action == 3 { (if next[0] != 0 && next[1] != 0 { S } else { -S }, true) }
        else { (-S / 16, false) }
    } else {
        let reward = [6, 23, 11, 7][action] * (S / 16) - (S / 4) * next[action];
        next[action] += 1;
        (reward, false)
    };
    validate_state(&next);
    bounded(reward as i128, 4 * S, "reward");
    (next, reward, terminal)
}

fn append_i64(bytes: &mut Vec<u8>, values: &[i64]) {
    for &v in values { bytes.extend_from_slice(&v.to_le_bytes()); }
}

fn append_u32(bytes: &mut Vec<u8>, values: &[u32]) {
    for &v in values { bytes.extend_from_slice(&v.to_le_bytes()); }
}

pub fn config_bytes(input: &PrivateInput) -> Vec<u8> {
    validate_input(input);
    let mut bytes = Vec::with_capacity(2200);
    bytes.extend_from_slice(b"zkDRL/ctc/a2c/config/v2\0");
    append_u32(&mut bytes, &[SCHEMA, input.kind, input.episodes, 16, 40, 32, 48, 16, 6, 7, 8, 1, 32, 64, 128, 256, 32]);
    bytes.extend_from_slice(&input.seed.to_le_bytes());
    bytes.extend_from_slice(&input.rng_seed);
    append_i64(&mut bytes, &input.initial_actor);
    append_i64(&mut bytes, &input.initial_critic);
    bytes.extend_from_slice(&input.config_salt);
    bytes.extend_from_slice(&input.final_salt);
    bytes.extend_from_slice(&TABLE_SHA);
    bytes
}

pub fn state_bytes(state: &State) -> Vec<u8> {
    assert_eq!(state.actor.len(), ACTOR);
    assert_eq!(state.critic.len(), CRITIC);
    let mut bytes = Vec::with_capacity(STATE_BYTES);
    append_i64(&mut bytes, &state.actor);
    append_i64(&mut bytes, &state.critic);
    bytes.extend_from_slice(&state.rng.to_le_bytes());
    append_i64(&mut bytes, &state.state);
    append_u32(&mut bytes, &[u32::from(state.done)]);
    bytes.extend_from_slice(&state.total.to_le_bytes());
    append_u32(&mut bytes, &[state.episodes, state.updates, state.interactions, state.resets, state.padded, state.successes]);
    assert_eq!(bytes.len(), STATE_BYTES);
    bytes
}

fn model_hash(actor: &[i64; ACTOR], critic: &[i64; CRITIC], updates: u32, hash: Hash) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(2000);
    bytes.extend_from_slice(b"zkRL/A2C-fixed/state/v1\0");
    append_u32(&mut bytes, &[16]);
    bytes.extend_from_slice(&(updates as u64).to_le_bytes());
    append_i64(&mut bytes, actor);
    append_i64(&mut bytes, critic);
    hash(&bytes)
}

/// Execute exactly six scheduled slots and one atomic update per episode.
pub fn run(input: &PrivateInput, hash: Hash, capture: bool) -> Output {
    validate_input(input);
    let mut actor: [i64; ACTOR] = input.initial_actor.as_slice().try_into().unwrap();
    let mut critic: [i64; CRITIC] = input.initial_critic.as_slice().try_into().unwrap();
    let config_commit = hash(&config_bytes(input));
    let initial_commit = {
        let mut bytes = Vec::with_capacity(2050);
        bytes.extend_from_slice(b"zkDRL/ctc/a2c/initial/v2\0");
        bytes.extend_from_slice(&config_commit);
        bytes.extend_from_slice(&input.config_salt);
        append_i64(&mut bytes, &actor);
        append_i64(&mut bytes, &critic);
        hash(&bytes)
    };
    let mut tape = CounterRng::new(A2C, input.rng_seed, input.challenge, hash);
    let mut rng = 0;
    let mut state = [0; 4];
    let mut done = false;
    let mut total = 0i64;
    let mut interactions = 0u32;
    let mut padded = 0u32;
    let mut successes = 0u32;
    let mut updates = 0u32;
    let mut resets = 0u32;
    let mut audit = if capture { Vec::with_capacity(input.episodes as usize) } else { Vec::new() };
    for episode in 0..input.episodes {
        let aq = quantize(&actor);
        let cq = quantize(&critic);
        state = if input.kind == 0 { [0, 0, 0, 1] } else { [0; 4] };
        done = false;
        // Every episode starts with exactly one environment reset, including 0.
        resets += 1;
        let mut ac = [Forward::<4>::ZERO; D];
        let mut cc = [Forward::<1>::ZERO; D];
        let mut mass = [[0u64; 4]; D];
        let mut actions = [0usize; D];
        let mut rewards = [0i64; D];
        let mut terminals = [false; D];
        let mut trace_continue = [false; D];
        let mut trace = if capture { Some([[0i64; 16]; D]) } else { None };
        let mut n = 0;
        for t in 0..D {
            let u = tape.next_u32(); // Padding consumes a draw too.
            rng = tape.position();
            let before = state;
            let previous_done = done;
            let action = if done { 4 } else {
                ac[n] = forward(&aq, state);
                cc[n] = forward(&cq, state);
                mass[n] = masses(&ac[n].output);
                categorical(&mass[n], u)
            };
            let (successor, reward, terminal) = transition(input.kind, state, action, done);
            if !previous_done {
                actions[n] = action;
                rewards[n] = reward;
                terminals[n] = terminal;
                trace_continue[n] = !terminal && t + 1 < D;
                n += 1;
                interactions += 1;
                if input.kind == 0 && terminal && reward > 0 { successes += 1; }
            } else { padded += 1; }
            // <=24000 rewards, each strictly <4*S: |total| <2^33.
            total += reward;
            if let Some(rows) = trace.as_mut() {
                rows[t] = [episode as i64, t as i64, before[0], before[1], before[2], before[3],
                    action as i64, reward, successor[0], successor[1], successor[2], successor[3],
                    i64::from(previous_done), i64::from(terminal), updates as i64, u as i64];
            }
            state = successor;
            done = terminal;
        }
        let targets = finite_returns(&rewards[..n], &terminals[..n], &trace_continue[..n]);
        let mut advantages = [0i64; D];
        let mut dz = [[0i64; 4]; D];
        let mut dv = [[0i64; 1]; D];
        for t in 0..n {
            advantages[t] = bounded(targets[t] as i128 - cc[t].output[0] as i128 * RATIO as i128, GRAD_LIMIT, "advantage");
            dz[t] = actor_upstream(&ac[t].output, &mass[t], actions[t], advantages[t]);
            dv[t][0] = -advantages[t];
        }
        let ga = backward(&aq, &ac[..n], &dz[..n]);
        let gc = backward(&cq, &cc[..n], &dv[..n]);
        apply_update(&mut actor, &mut critic, &ga, &gc);
        updates += 1;
        if let Some(trace) = trace {
            let mut bytes = Vec::with_capacity(32 + 4 + n * 640 + (ACTOR + CRITIC) * 8);
            bytes.extend_from_slice(b"zkRL/A2C-SP1/stages/v1\0");
            append_u32(&mut bytes, &[n as u32]);
            for t in 0..n {
                append_i64(&mut bytes, &ac[t].output);
                append_i64(&mut bytes, &cc[t].output);
                for &m in &mass[t] { bytes.extend_from_slice(&m.to_le_bytes()); }
                append_i64(&mut bytes, &[targets[t], advantages[t]]);
                append_i64(&mut bytes, &dz[t]);
                append_i64(&mut bytes, &dv[t]);
                append_i64(&mut bytes, &ac[t].pre);
                append_i64(&mut bytes, &ac[t].hidden);
                append_i64(&mut bytes, &cc[t].pre);
                append_i64(&mut bytes, &cc[t].hidden);
            }
            append_i64(&mut bytes, &ga);
            append_i64(&mut bytes, &gc);
            audit.push(Audit { episode, trace: trace.to_vec(),
                model_sha256: model_hash(&actor, &critic, updates, hash),
                stages_sha256: hash(&bytes) });
        }
    }
    // All counters are bounded by 6*MAX_EPISODES=24000; no wrapping is needed.
    let slots = input.episodes * D as u32;
    assert_eq!(interactions + padded, slots);
    assert_eq!(updates, input.episodes);
    assert_eq!(resets, input.episodes);
    let actor_export = quantize(&actor).to_vec();
    let critic_export = quantize(&critic).to_vec();
    let final_state = State { actor: actor.to_vec(), critic: critic.to_vec(), rng, state, done, total,
        episodes: input.episodes, updates, interactions, resets, padded, successes };
    let final_commit = {
        let mut bytes = Vec::with_capacity(STATE_BYTES + 128);
        bytes.extend_from_slice(b"zkDRL/ctc/a2c/final/v2\0");
        bytes.extend_from_slice(&config_commit);
        bytes.extend_from_slice(&input.challenge);
        bytes.extend_from_slice(&input.final_salt);
        bytes.extend_from_slice(&state_bytes(&final_state));
        hash(&bytes)
    };
    let journal = Journal { schema: SCHEMA, config_commit, initial_commit, final_commit,
        challenge: input.challenge, kind: input.kind, episodes: input.episodes, slots,
        interactions, updates, resets, padded, successes, return_units: total };
    Output { journal, final_state, actor_export, critic_export, audit }
}

#[cfg(test)]
mod tests;
