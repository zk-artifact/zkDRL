use super::*;
use std::cell::RefCell;

thread_local! {
    static HASH_INPUTS: RefCell<Vec<Vec<u8>>> = const { RefCell::new(Vec::new()) };
}

use sha2::{Digest, Sha256};

fn test_hash(bytes: &[u8]) -> [u8; 32] { Sha256::digest(bytes).into() }

fn recording_hash(bytes: &[u8]) -> [u8; 32] {
    HASH_INPUTS.with(|v| v.borrow_mut().push(bytes.to_vec()));
    test_hash(bytes)
}

fn input(kind: u32, scale: u32, slots: u32, train: u32) -> PrivateInput {
    let mut weights = vec![0; P];
    // Small, nontrivial, mixed-sign initial weights; not a provenance claim.
    for (k, w) in weights.iter_mut().enumerate() {
        *w = ((k % 7) as i64 - 3) * i64::from(scale) / 16;
    }
    for w in &mut weights[4 * H..5 * H] { *w = i64::from(scale) / 4; }
    for w in &mut weights[9 * H..] { *w = 0; }
    PrivateInput {
        schema: SCHEMA, kind, scale, slots, train, seed: 30, rng_seed: [37; 32], initial_coarse: weights,
        params: if kind == 0 { [0; 4] } else { [scale as i64, 0, -(scale as i64), scale as i64 / 2] },
        config_salt: [11; 32], final_salt: [19; 32], challenge: [23; 32],
    }
}

#[test]
fn shifts_equal_python_floor_at_boundaries() {
    for bits in [1, 4, 8, 11, 15] {
        let d = 1i64 << bits;
        for x in [i64::MIN, -(1i64 << 50), -d - 1, -d, -d + 1, -1, 0, 1,
                  d - 1, d, d + 1, (1i64 << 50) - 1, i64::MAX] {
            assert_eq!(x >> bits, floor_div(x, d));
        }
    }
}

#[test]
fn capture_quiet_repeatability_and_cursor_all_modes() {
    for kind in [0, 1] {
        for scale in [16, 256] {
            for train in [0, 1] {
                for slots in [6, 30, 36, 72, 1536] {
                    let i = input(kind, scale, slots, train);
                    let new = run(&i, test_hash, true);
                    let quiet = run(&i, test_hash, false);
                    let repeat = run(&i, test_hash, true);
                    assert_eq!(new.journal, repeat.journal);
                    assert_eq!(new.audit, repeat.audit);
                    assert_eq!(new.journal, quiet.journal);
                    assert_eq!(new.final_state, quiet.final_state);
                    assert_eq!(new.coarse_export, quiet.coarse_export);
                    assert_eq!(new.final_state.rng, 10 * slots as u64);
                    assert!(quiet.state_chain.is_none() && quiet.trajectory_chain.is_none());
                    assert!(quiet.audit.is_empty());
                    assert_eq!(quiet.audit.capacity(), 0);
                    for (slot, row) in new.audit.iter().enumerate() {
                        assert_eq!(row.rng, 10 * (slot as u64 + 1));
                    }
                }
            }
        }
    }
}

#[test]
fn uncaptured_hashes_and_exact_final_preimage() {
    let i = input(1, 16, 72, 1);
    HASH_INPUTS.with(|v| v.borrow_mut().clear());
    let out = run(&i, recording_hash, false);
    HASH_INPUTS.with(|v| {
        let inputs = v.borrow();
        assert_eq!(inputs.len(), 3 + (10 * i.slots as usize + 7) / 8);
        assert_eq!(inputs[0], config_bytes(&i));
        let mut key = b"zkDRL/ctc/rng-key/v1\0".to_vec();
        key.extend_from_slice(&1u32.to_le_bytes());
        key.extend_from_slice(&i.rng_seed);
        key.extend_from_slice(&i.challenge);
        assert_eq!(inputs[1], key);
        let key = test_hash(&key);
        for (j, block) in inputs[2..inputs.len() - 1].iter().enumerate() {
            let mut expected = b"zkDRL/ctc/rng-block/v1\0".to_vec();
            expected.extend_from_slice(&key);
            expected.extend_from_slice(&(j as u64).to_le_bytes());
            assert_eq!(*block, expected);
        }
        let mut expected = b"zkDRL/ctc/dqn/final/v3\0".to_vec();
        expected.extend_from_slice(&out.journal.config_commit);
        expected.extend_from_slice(&i.challenge);
        expected.extend_from_slice(&i.final_salt);
        let state = state_bytes(&out.final_state);
        assert_eq!(state.len(), 16 * P + 88 * CAP + 76);
        expected.extend_from_slice(&state);
        i64s(&mut expected, &out.coarse_export);
        i64s(&mut expected, &coarse(&out.final_state.target, i.scale as i64));
        assert_eq!(*inputs.last().unwrap(), expected);
    });
}

#[test]
fn full_4000_episodes_have_bounded_counters() {
    for kind in [0, 1] {
        for scale in [16, 256] {
            for train in [0, 1] {
                let mut i = input(kind, scale, MAX_SLOTS, train);
                // Isolate counter limits with a stable bias-only learner.
                i.initial_coarse.fill(0);
                let o = run(&i, test_hash, false);
                assert_eq!(o.final_state.rng, 10 * MAX_SLOTS as u64);
                assert_eq!(o.journal.resets, 4000);
                assert_eq!(o.journal.updates, o.journal.interactions.saturating_sub(31));
                assert_eq!(o.journal.replay_wraps, o.journal.interactions.saturating_sub(CAP as u32));
                assert_eq!(o.journal.target_copies, o.journal.updates / 32);
                if train == 1 { assert_eq!(o.journal.interactions + o.journal.padded, MAX_SLOTS); }
                else { assert_eq!(o.journal.interactions, 0); }
                if kind == 1 { assert_eq!(o.journal.padded, 0); }
            }
        }
    }
}

#[test]
fn fine_target_copy_preserves_residues_and_stays_frozen_between_copies() {
    let mut i = input(1, 16, 66, 1);
    let after = run(&i, test_hash, true);
    let mut before_target = None;
    // At interaction 63, update 32 copies the full fine master.
    // Use the same challenge and salts for all prefixes: slots do not seed RNG.
    i.slots = 60;
    let before = run(&i, test_hash, false);
    assert_eq!(before.journal.updates, 29);
    assert_eq!(after.journal.updates, 35);
    assert_eq!(before.final_state.target, input(1, 16, 60, 1).initial_coarse.iter()
        .map(|w| w * 2048).collect::<Vec<_>>());
    // Directly exercise observe to inspect the non-six-slot copy boundary.
    let master = std::array::from_fn(|k| i.initial_coarse[k] * 2048);
    let mut online = [0; P];
    refresh_coarse(&master, &mut online, 4);
    let mut st = CoreState { master, target: master, online, target_coarse: online,
        buffer: [[0; 11]; CAP], interactions: 0, updates: 0, rng: 0, state: [0; 4],
        done: false, total: 0, resets: 0, padded: 0, target_copies: 0, replay_wraps: 0 };
    for n in 1..=65 {
        observe(&mut st, [0, 0, 0, 0, 0, 7, 1, 0, 0, 0, 0], &[0; B], 4);
        if n == 63 {
            assert_eq!(st.updates, 32);
            assert_eq!(st.target, st.master);
            assert!(st.target.iter().any(|m| m.rem_euclid(2048) != 0));
            before_target = Some(st.target);
        }
        if n > 63 { assert_eq!(st.target, before_target.unwrap()); }
    }
}

#[test]
fn masked_diagnostic_does_not_reject_inactive_hidden_unit() {
    // Raw err*W2/S exceeds signed32, but this hidden unit is inactive.
    let mut i = input(1, 16, 36, 1);
    i.initial_coarse.fill(0);
    i.initial_coarse[4 * H] = -1;
    i.initial_coarse[5 * H] = (1 << 20) - 1;
    i.initial_coarse[9 * H] = (1 << 20) - 1;
    let out = run(&i, test_hash, false);
    assert_eq!(out.journal.updates, 5);
}

#[test]
fn forward_floor_and_diagnostic_boundaries() {
    let mut w = [0; P];
    w[4 * H] = 1;
    w[5 * H] = -1;
    assert_eq!(forward(&w, &[0; 4], 16).0[0], -1);
    // A valid coarse vector can still leave the Q diagnostic domain.
    w[4 * H..5 * H].fill((1 << 20) - 1);
    w[5 * H..9 * H].fill((1 << 20) - 1);
    assert!(std::panic::catch_unwind(|| forward(&w, &[0; 4], 16)).is_err());
    let mut m = [0; P];
    m[0] = -1;
    assert_eq!(coarse(&m, 16)[0], -1);
    for v in [-(1i64 << 50), 1i64 << 50] {
        m[0] = v;
        assert!(std::panic::catch_unwind(|| coarse(&m, 16)).is_err());
    }
}

#[test]
fn invalid_schema_slots_and_coarse_are_rejected() {
    let base = input(0, 16, 6, 0);
    for (schema, slots) in [(1, 6), (2, 6), (203, 6), (202, 0), (202, 5), (202, 7), (202, 24006)] {
        let mut i = base.clone(); i.schema = schema; i.slots = slots;
        assert!(std::panic::catch_unwind(|| run(&i, test_hash, false)).is_err());
    }
    for w in [-(1 << 20) - 1, 1 << 20, i64::MIN, i64::MAX] {
        let mut i = base.clone(); i.initial_coarse[0] = w;
        assert!(std::panic::catch_unwind(|| run(&i, test_hash, false)).is_err());
    }
}

#[test]
fn canonical_config_order_and_all_private_bindings() {
    let x = input(1, 256, 12, 1);
    let mut expected = b"zkDRL/ctc/dqn/config/v3\0".to_vec();
    for v in [202, 1, 256, 12, 1, H as u32, CAP as u32, 8, 32, 16, 32, 6, 4, 1, 2, 30] {
        expected.extend_from_slice(&v.to_le_bytes());
    }
    expected.extend_from_slice(&[37; 32]);
    for &v in x.initial_coarse.iter().chain(&x.params) { expected.extend_from_slice(&v.to_le_bytes()); }
    expected.extend_from_slice(&[11; 32]);
    expected.extend_from_slice(&[19; 32]);
    assert_eq!(expected.len(), 216 + 8 * P);
    if H == 16 && CAP == 64 { assert_eq!(expected.len(), 1400); }
    assert_eq!(config_bytes(&x), expected);
    let original = run(&x, test_hash, false);
    for which in 0..7 {
        let mut changed = x.clone();
        match which {
            0 => changed.rng_seed[0] ^= 1,
            1 => changed.initial_coarse[0] += 1,
            2 => changed.seed += 1,
            3 => changed.config_salt[0] ^= 1,
            4 => changed.final_salt[0] ^= 1,
            5 => changed.slots += 6,
            _ => changed.params[0] += 1,
        }
        assert_ne!(config_bytes(&changed), expected);
        assert_ne!(run(&changed, test_hash, false).journal.config_commit, original.journal.config_commit);
    }
    let mut changed = x.clone();
    changed.challenge[0] ^= 1;
    assert_eq!(config_bytes(&changed), expected);
    let other = run(&changed, test_hash, false);
    assert_eq!(original.journal.config_commit, other.journal.config_commit);
    assert_ne!(original.journal.final_commit, other.journal.final_commit);
    assert_eq!(original.final_state.rng, other.final_state.rng);
}

#[test]
fn terminal_padding_eval_cursor_and_challenge_final_binding() {
    let mut x = input(0, 16, 12, 0);
    x.initial_coarse.fill(0);
    x.initial_coarse[9 * H + 3] = 1; // Submit immediately, independent of random words.
    let out = run(&x, test_hash, true);
    assert_eq!(out.final_state.rng, 120);
    assert_eq!((out.journal.padded, out.journal.resets, out.journal.interactions), (10, 2, 0));
    for row in &out.audit {
        assert_eq!(row.action, if row.slot % 6 == 0 { 3 } else { 4 });
        assert_eq!(row.rng, 10 * (row.slot as u64 + 1));
    }
    x.challenge[0] ^= 1;
    let other = run(&x, test_hash, false);
    assert_eq!(out.final_state, other.final_state);
    assert_eq!(out.journal.config_commit, other.journal.config_commit);
    assert_ne!(out.journal.final_commit, other.journal.final_commit);
}

#[test]
fn draws_match_low_bits_and_multiply_high_including_warmup() {
    let x = input(1, 16, 300, 1);
    let out = run(&x, test_hash, true);
    let mut rng = CounterRng::new(DQN, x.rng_seed, x.challenge, test_hash);
    for row in &out.audit {
        let explore = rng.next_u32() & 15;
        let action = rng.next_u32() & 3;
        let draws: [u32; B] = std::array::from_fn(|_| rng.next_u32());
        if explore < 4 { assert_eq!(row.action, action); }
        if row.interactions < WARMUP { assert_eq!(row.replay_indices, [-1; B]); }
        else {
            let count = row.interactions.min(CAP as u32) as u64;
            assert_eq!(row.replay_indices, draws.map(|word| ((word as u64 * count) >> 32) as i64));
        }
        assert_eq!(row.rng, rng.position());
    }
    let mut prefix = x.clone();
    prefix.slots = 36;
    prefix.seed += 1; // Initializer metadata does not enter the random key.
    prefix.config_salt[0] ^= 1;
    prefix.final_salt[0] ^= 1;
    let shorter = run(&prefix, test_hash, true);
    assert_eq!(shorter.audit, out.audit[..36]);
}

#[test]
fn canonical_state_and_unsigned_cursor() {
    let x = input(1, 16, 6, 0);
    let mut state = run(&x, test_hash, false).final_state;
    state.rng = u64::MAX; // Serialization is unsigned even outside the run's bounded cursor range.
    let mut expected = Vec::new();
    for &v in state.master.iter().chain(&state.target).chain(state.buffer.iter().flatten()) {
        expected.extend_from_slice(&v.to_le_bytes());
    }
    expected.extend_from_slice(&state.interactions.to_le_bytes());
    expected.extend_from_slice(&state.updates.to_le_bytes());
    let offset = expected.len();
    expected.extend_from_slice(&u64::MAX.to_le_bytes());
    for v in state.state { expected.extend_from_slice(&v.to_le_bytes()); }
    expected.extend_from_slice(&(state.done as u32).to_le_bytes());
    expected.extend_from_slice(&state.total.to_le_bytes());
    for v in [state.resets, state.padded, state.target_copies, state.replay_wraps] {
        expected.extend_from_slice(&v.to_le_bytes());
    }
    assert_eq!(offset, 16 * P + 88 * CAP + 8);
    assert_eq!(state_bytes(&state), expected);
    assert_eq!(expected.len(), STATE_BYTES);
    if H == 16 && CAP == 64 { assert_eq!(STATE_BYTES, 8076); }
    state.buffer.pop();
    assert!(std::panic::catch_unwind(|| state_bytes(&state)).is_err());
}

#[test]
fn native_audit_encodings_and_hash_count() {
    let x = input(0, 16, 12, 1);
    HASH_INPUTS.with(|v| v.borrow_mut().clear());
    let out = run(&x, recording_hash, true);
    HASH_INPUTS.with(|v| {
        let inputs = v.borrow();
        assert_eq!(inputs.len(), 3 + (10 * x.slots as usize + 7) / 8 + 2 * x.slots as usize);
        let states: Vec<_> = inputs.iter().filter(|b| b.starts_with(b"zkRL/risc0/state-chain/v1\0")).collect();
        let trajectories: Vec<_> = inputs.iter().filter(|b| b.starts_with(b"zkRL/risc0/trajectory/v1\0")).collect();
        let mut state_chain = [0; 32];
        let mut trajectory_chain = [0; 32];
        for (i, row) in out.audit.iter().enumerate() {
            let mut trace = b"zkRL/risc0/trajectory/v1\0".to_vec();
            trace.extend_from_slice(&trajectory_chain);
            let mut signed = vec![row.slot as i64, row.slot as i64 / 6, row.slot as i64 % 6];
            signed.extend(row.state);
            signed.extend([row.start_done as i64, row.action as i64, row.reward]);
            signed.extend(row.successor);
            signed.push(row.terminal as i64);
            signed.extend(row.replay_indices);
            signed.extend([row.interactions as i64, row.updates as i64]);
            for v in signed { trace.extend_from_slice(&v.to_le_bytes()); }
            trace.extend_from_slice(&row.rng.to_le_bytes());
            assert_eq!(*trajectories[i], trace);
            trajectory_chain = test_hash(&trace);
            assert_eq!(row.trajectory_chain, trajectory_chain);
            let start = b"zkRL/risc0/state-chain/v1\0".len();
            assert_eq!(&states[i][start..start + 32], &state_chain);
            let state = &states[i][start + 32..];
            assert_eq!(state.len(), STATE_BYTES);
            let offset = 16 * P + 88 * CAP + 8;
            assert_eq!(&state[offset..offset + 8], &row.rng.to_le_bytes());
            if i + 1 == out.audit.len() { assert_eq!(state, state_bytes(&out.final_state)); }
            state_chain = test_hash(states[i]);
            assert_eq!(row.state_chain, state_chain);
        }
        assert_eq!(out.state_chain, Some(state_chain));
        assert_eq!(out.trajectory_chain, Some(trajectory_chain));
    });
}

#[test]
fn journal_abi_and_remaining_input_guards() {
    let x = input(0, 16, 6, 0);
    let j = run(&x, test_hash, false).journal;
    let mut expected = 202u32.to_le_bytes().to_vec();
    for bytes in [j.config_commit, j.final_commit, j.challenge] { expected.extend_from_slice(&bytes); }
    for v in [j.kind, j.scale, j.slots, j.train] { expected.extend_from_slice(&v.to_le_bytes()); }
    expected.extend_from_slice(&j.return_units.to_le_bytes());
    for v in [j.interactions, j.updates, j.resets, j.padded, j.target_copies, j.replay_wraps] {
        expected.extend_from_slice(&v.to_le_bytes());
    }
    assert_eq!(expected.len(), 148);
    assert_eq!(bincode::serialize(&j).unwrap(), expected);
    for which in 0..8 {
        let mut bad = x.clone();
        match which {
            0 => bad.kind = 2,
            1 => bad.train = 2,
            2 => bad.scale = 17,
            3 => { bad.initial_coarse.pop(); },
            4 => bad.initial_coarse.push(0),
            5 => bad.params[0] = 1,
            6 => { bad.kind = 1; bad.params[0] = 16 * 16 + 1; },
            _ => { bad.kind = 1; bad.params[0] = -16 * 16 - 1; },
        }
        assert!(std::panic::catch_unwind(|| run(&bad, test_hash, false)).is_err());
    }
    let d = serde::de::value::MapDeserializer::<_, serde::de::value::Error>::new(
        [("external_tape", 0u32)].into_iter(),
    );
    assert!(PrivateInput::deserialize(d).unwrap_err().to_string().contains("unknown field"));
}
