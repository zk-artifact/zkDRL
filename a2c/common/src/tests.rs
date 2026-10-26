use super::*;
use sha2::{Digest, Sha256};
use std::cell::Cell;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn sha(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn input(kind: u32, episodes: u32) -> PrivateInput {
    PrivateInput {
        schema: SCHEMA, kind, episodes, seed: 0, rng_seed: [0x44; 32],
        initial_actor: vec![0; ACTOR], initial_critic: vec![0; CRITIC],
        config_salt: [0x11; 32], final_salt: [0x22; 32], challenge: [0x33; 32],
    }
}

fn rejects(f: impl FnOnce()) {
    assert!(catch_unwind(AssertUnwindSafe(f)).is_err(), "expected rejection");
}

#[test]
fn negative_floor_and_step_floor_placement() {
    assert_eq!(floor_div(-1, 192), -1);
    assert_eq!(floor_div(-192, 192), -1);
    assert_eq!(floor_div(-193, 192), -2);
    assert_eq!(floor_div(191, 192), 0);
    assert_eq!(floor_div(i128::MIN, 1), i128::MIN);
    rejects(|| { floor_div(1, 0); });
    rejects(|| { floor_div(1, -1); });
    let mut masters = [0; ACTOR];
    masters[0] = -1;
    masters[1] = -RATIO - 1;
    assert_eq!(&quantize(&masters)[..2], &[-1, -2]);

    let mut a = [0; ACTOR];
    let mut c = [0; CRITIC];
    let mut ga = [0; ACTOR];
    let mut gc = [0; CRITIC];
    ga[..4].copy_from_slice(&[1, -1, 191, -193]);
    gc[..4].copy_from_slice(&[1, -1, 383, -385]);
    apply_update(&mut a, &mut c, &ga, &gc);
    assert_eq!(&a[..4], &[0, 1, 0, 2]);
    assert_eq!(&c[..4], &[0, 1, 0, 2]);
    // Updating masters retains changes too small to move the forward grid.
    assert_eq!(quantize(&a), [0; ACTOR]);
    apply_update(&mut a, &mut c, &ga, &gc);
    assert_eq!(&a[..4], &[0, 2, 0, 4]);
    assert_eq!(quantize(&a), [0; ACTOR]);
}

#[test]
fn lut_endpoints_interpolation_cutoff_and_ties() {
    assert_eq!(exp_weight(0), 1 << 48);
    assert_eq!(exp_weight(-256), 280377609773053);
    assert_eq!(exp_weight(-1), (255 * (1u64 << 48) + 280377609773053) / 256);
    assert_eq!(exp_weight(-255), ((1u64 << 48) + 255 * 280377609773053) / 256);
    assert_eq!(exp_weight(-32 * S), 3);
    assert_eq!(exp_weight(-32 * S - 1), 0);
    assert_eq!(exp_weight(i64::MIN), 0);
    rejects(|| { exp_weight(1); });
    assert_eq!(masses(&[0; 4]), [P / 4; 4]);
    // P mod 3 = 1: equal remainders award the extra unit to action zero.
    assert_eq!(masses(&[0, 0, 0, -33 * S]), [1431655766, 1431655765, 1431655765, 0]);
    assert_eq!(masses(&[-33 * S, 0, 0, 0]), [0, 1431655766, 1431655765, 1431655765]);
    assert_eq!(masses(&[0, -33 * S, -33 * S, -33 * S]), [P, 0, 0, 0]);
    assert_eq!(masses(&[12, 12 - 33 * S, 12 - 33 * S, 12 - 33 * S]), [P, 0, 0, 0]);
}

#[test]
fn cdf_zero_full_mass_and_equality_boundaries() {
    let m = [0, 1, P - 2, 1];
    assert_eq!(categorical(&m, 0), 1);
    assert_eq!(categorical(&m, 1), 2);
    assert_eq!(categorical(&m, u32::MAX - 1), 2);
    assert_eq!(categorical(&m, u32::MAX), 3);
    for a in 0..4 {
        let mut deterministic = [0; 4];
        deterministic[a] = P;
        assert_eq!(categorical(&deterministic, 0), a);
        assert_eq!(categorical(&deterministic, u32::MAX), a);
    }
    rejects(|| { categorical(&[0; 4], 0); });
    rejects(|| { categorical(&[P + 1, 0, 0, 0], 0); });
    rejects(|| { categorical(&[u64::MAX; 4], 0); });
}

#[test]
fn forward_layout_negative_floor_relu_and_guards() {
    let mut p = [0; CRITIC];
    p[0] = 2;
    p[64] = -1;
    p[80] = -1;
    let f: Forward<1> = forward(&p, [1, 0, 0, 0]);
    assert_eq!(f.pre[0], 1);
    assert_eq!(f.hidden[0], 1);
    assert_eq!(f.output, [-1]); // -1/S floors to -1.
    let f: Forward<1> = forward(&p, [0; 4]);
    assert_eq!(f.pre[0], -1);
    assert_eq!(f.hidden[0], 0);
    assert_eq!(f.output, [0]);
    for bad in [-1, 7, i64::MIN, i64::MAX] {
        rejects(|| { forward::<1, CRITIC>(&p, [bad, 0, 0, 0]); });
    }
    for bad in [-PARAM_LIMIT, PARAM_LIMIT, i64::MIN, i64::MAX] {
        let mut p = [0; CRITIC]; p[0] = bad;
        rejects(|| { forward::<1, CRITIC>(&p, [0; 4]); });
    }
    for sign in [-1, 1] {
        let mut p = [0; CRITIC];
        p[0] = sign * (PARAM_LIMIT - 1);
        rejects(|| { forward::<1, CRITIC>(&p, [6, 0, 0, 0]); });
        // Exactly +/-1024*S preactivation is also rejected, before ReLU.
        p[0] = sign * 200 * S;
        p[64] = sign * 24 * S;
        rejects(|| { forward::<1, CRITIC>(&p, [5, 0, 0, 0]); });
        let mut p = [0; CRITIC];
        p[64] = 128 * S;
        p[80] = sign * 8 * S;
        rejects(|| { forward::<1, CRITIC>(&p, [0; 4]); });
    }
    rejects(|| { forward::<2, CRITIC>(&[0; CRITIC], [0; 4]); });
    rejects(|| { forward::<1, ACTOR>(&[0; ACTOR], [0; 4]); });
}

#[test]
fn relu_derivative_zero_and_mask_before_hidden_bound() {
    let mut p = [0; CRITIC];
    p[80] = PARAM_LIMIT - 1;
    for pre in [0, -1] {
        p[64] = pre;
        let f = forward::<1, CRITIC>(&p, [1, 0, 0, 0]);
        // Raw dh exceeds GRAD_LIMIT; the zero mask must precede its bound.
        let grad = backward(&p, &[f], &[[GRAD_LIMIT - 1]]);
        assert_eq!(&grad[..96], &[0; 96]);
        assert_eq!(grad[96], GRAD_LIMIT - 1);
    }
    p[64] = 1;
    let f = forward::<1, CRITIC>(&p, [1, 0, 0, 0]);
    rejects(|| { backward(&p, &[f], &[[GRAD_LIMIT - 1]]); });
}

#[test]
fn vjp_sums_before_floor_in_both_dots() {
    let mut p = [0; ACTOR];
    p[64] = 1;
    p[80] = S / 2;
    p[96] = S / 2;
    let f = forward::<4, ACTOR>(&p, [1, 0, 0, 0]);
    let grad = backward(&p, &[f], &[[1, 1, 0, 0]]);
    assert_eq!(grad[0], 1);
    assert_eq!(grad[64], 1);
    let mut p = [0; CRITIC];
    p[64] = S / 2;
    let f = forward::<1, CRITIC>(&p, [0; 4]);
    assert_eq!(backward(&p, &[f, f], &[[1], [1]])[80], 1);
    assert_eq!(backward(&p, &[f, f], &[[-1], [-1]])[80], -1);
}

#[test]
fn backward_rejects_upstream_and_each_gradient_bound() {
    let p = [0; CRITIC];
    let f = forward::<1, CRITIC>(&p, [0; 4]);
    for bad in [-GRAD_LIMIT, GRAD_LIMIT, i64::MIN, i64::MAX] {
        rejects(|| { backward(&p, &[f], &[[bad]]); });
    }
    rejects(|| { backward(&p, &[f, f], &[[GRAD_LIMIT / 2]; 2]); }); // b2
    let mut p = [0; CRITIC];
    p[64] = 3 * S;
    let f = forward::<1, CRITIC>(&p, [0; 4]);
    rejects(|| { backward(&p, &[f], &[[GRAD_LIMIT / 2]]); }); // W2
    p[64] = 1;
    p[80] = S / 2;
    let f = forward::<1, CRITIC>(&p, [6, 0, 0, 0]);
    rejects(|| { backward(&p, &[f], &[[GRAD_LIMIT / 2]]); }); // W1
    p[80] = 2 * S;
    let f = forward::<1, CRITIC>(&p, [0; 4]);
    rejects(|| { backward(&p, &[f, f], &[[GRAD_LIMIT / 4]; 2]); }); // b1
    rejects(|| { backward::<1, CRITIC>(&p, &[], &[]); });
    rejects(|| { backward(&p, &[f; 7], &[[0]; 7]); });
    rejects(|| { backward(&p, &[f], &[]); });
    let mut forged = f;
    forged.hidden[0] = i64::MAX;
    rejects(|| { backward(&p, &[forged], &[[0]]); });
}

#[test]
fn upstream_separate_policy_entropy_floors_and_bounds() {
    let uniform = [P / 4; 4];
    assert_eq!(actor_upstream(&[0; 4], &uniform, 0, 1), [-1, 0, 0, 0]);
    assert_eq!(actor_upstream(&[0; 4], &uniform, 0, -1), [0, -1, -1, -1]);
    assert_eq!(actor_upstream(&[0, -1, 0, 0], &uniform, 0, 1), [8191, -24576, 8192, 8192]);
    // Compare the canceled expression against the original Q40 numerator.
    let logits = [S, -S / 3, 7, -17];
    let mass = masses(&logits);
    let advantage = -G / 7;
    let actual = actor_upstream(&logits, &mass, 2, advantage);
    let shifted = logits.map(|v| v - S);
    let moment: i128 = (0..4).map(|i| mass[i] as i128 * shifted[i] as i128).sum();
    for i in 0..4 {
        let policy = (advantage as i128 * (mass[i] as i128 - if i == 2 { P as i128 } else { 0 })).div_euclid(P as i128);
        let entropy = (mass[i] as i128 * (shifted[i] as i128 * P as i128 - moment) * RATIO as i128)
            .div_euclid(128 * P as i128 * P as i128);
        assert_eq!(actual[i] as i128, policy + entropy);
    }
    rejects(|| { actor_upstream(&[0; 4], &uniform, 4, 0); });
    rejects(|| { actor_upstream(&[0; 4], &uniform, 0, GRAD_LIMIT); });
    rejects(|| { actor_upstream(&[FORWARD_LIMIT, 0, 0, 0], &uniform, 0, 0); });
    rejects(|| { masses(&[-FORWARD_LIMIT, 0, 0, 0]); });
    rejects(|| { actor_upstream(&[0; 4], &[u64::MAX; 4], 0, 0); });
    // Near-limit operands exercise the wide numerator without overflowing.
    let extreme = actor_upstream(&[FORWARD_LIMIT - 1, -FORWARD_LIMIT + 1, 0, 0], &uniform, 1, GRAD_LIMIT - 1);
    assert!(extreme.iter().all(|&v| v > -GRAD_LIMIT && v < GRAD_LIMIT));
}

#[test]
fn finite_returns_direct_order_and_trace_cuts() {
    let result = finite_returns(&[0, -1], &[false, false], &[true, false]);
    assert_eq!(&result[..2], &[-7 * RATIO / 8, -RATIO]);
    let result = finite_returns(&[S, -S, S], &[false, true, false], &[true, false, false]);
    assert_eq!(&result[..3], &[G / 8, -G, G]);
    let rewards = [-S / 16; 6];
    let result = finite_returns(&rewards, &[false; 6], &[true, true, true, true, true, false]);
    for t in 0..6 {
        let expected: i64 = (0..6-t).map(|k| -(G / 16) * 7i64.pow(k as u32) / 8i64.pow(k as u32)).sum();
        assert_eq!(result[t], expected);
    }
    rejects(|| { finite_returns(&[], &[], &[]); });
    rejects(|| { finite_returns(&[0; 7], &[false; 7], &[false; 7]); });
    rejects(|| { finite_returns(&[0], &[], &[false]); });
    rejects(|| { finite_returns(&[0], &[false], &[true]); });
    rejects(|| { finite_returns(&[0, 0], &[true, false], &[true, false]); });
    for bad in [-4 * S, 4 * S, i64::MIN, i64::MAX] {
        rejects(|| { finite_returns(&[bad], &[false], &[false]); });
    }
    // The private common bound is used for targets, advantages and all dy/grad.
    for bad in [-(GRAD_LIMIT as i128), GRAD_LIMIT as i128, i128::MIN, i128::MAX] {
        rejects(|| { bounded(bad, GRAD_LIMIT, "test bound"); });
    }
}

#[test]
fn candidates_reject_atomically_for_either_network_and_boundary() {
    for bad_actor in [true, false] {
        for negative_quantized_edge in [true, false] {
            let mut a = [0; ACTOR];
            let mut c = [0; CRITIC];
            let mut ga = [-1; ACTOR];
            let mut gc = [-1; CRITIC];
            let old = if negative_quantized_edge { -MASTER_LIMIT + RATIO } else { MASTER_LIMIT - 1 };
            if bad_actor {
                a[17] = old;
                ga[17] = if negative_quantized_edge { 192 } else { -192 };
            } else {
                c[17] = old;
                gc[17] = if negative_quantized_edge { 384 } else { -384 };
            }
            validate_masters(&a);
            validate_masters(&c);
            let before_a = a;
            let before_c = c;
            rejects(|| { apply_update(&mut a, &mut c, &ga, &gc); });
            assert_eq!(a, before_a);
            assert_eq!(c, before_c);
        }
    }
    for bad_actor in [true, false] {
        let mut a = [0; ACTOR];
        let mut c = [0; CRITIC];
        let mut ga = [-1; ACTOR];
        let mut gc = [-1; CRITIC];
        if bad_actor { ga[0] = GRAD_LIMIT; } else { gc[0] = -GRAD_LIMIT; }
        rejects(|| { apply_update(&mut a, &mut c, &ga, &gc); });
        assert_eq!(a, [0; ACTOR]);
        assert_eq!(c, [0; CRITIC]);
    }
}

#[test]
fn input_schema_domain_lengths_and_initial_bounds() {
    for schema in [0, 101, 202, 204, u32::MAX] {
        let mut x = input(0, 1); x.schema = schema;
        rejects(|| { run(&x, sha, false); });
    }
    for kind in [2, u32::MAX] {
        rejects(|| { run(&input(kind, 1), sha, false); });
    }
    for episodes in [0, 4001, u32::MAX] {
        rejects(|| { run(&input(0, episodes), sha, false); });
    }
    for actor in [true, false] {
        for len_delta in [-1isize, 1] {
            let mut x = input(0, 1);
            let values = if actor { &mut x.initial_actor } else { &mut x.initial_critic };
            values.resize((values.len() as isize + len_delta) as usize, 0);
            rejects(|| { run(&x, sha, false); });
        }
        for bad in [-MASTER_LIMIT, MASTER_LIMIT, -MASTER_LIMIT + 1, -MASTER_LIMIT + RATIO - 1, i64::MIN, i64::MAX] {
            let mut x = input(0, 1);
            if actor { x.initial_actor[0] = bad; } else { x.initial_critic[0] = bad; }
            rejects(|| { run(&x, sha, false); });
        }
    }
    let mut x = input(1, MAX_EPISODES as u32);
    x.seed = u64::MAX;
    x.initial_actor[0] = -MASTER_LIMIT + RATIO;
    x.initial_critic[0] = MASTER_LIMIT - 1;
    validate_input(&x); // Validate the accepted edges without training 4000 episodes.
    let deserializer = serde::de::value::MapDeserializer::<_, serde::de::value::Error>::new(
        [("unexpected_hint", 0u32)].into_iter(),
    );
    let error = PrivateInput::deserialize(deserializer).unwrap_err().to_string();
    assert!(error.contains("unknown field"), "{error}");
}

#[test]
fn exact_login_mdp_and_successful_terminal() {
    let initial = [0, 0, 0, 1];
    let (credential, r, done) = transition(0, initial, 0, false);
    assert_eq!((credential, r, done), ([1, 0, 0, 1], -S / 16, false));
    assert_eq!(transition(0, credential, 1, false), (initial, -S / 16, false));
    let (both, _, _) = transition(0, credential, 2, false);
    let (terminal, r, done) = transition(0, both, 3, false);
    assert_eq!((terminal, r, done), ([1, 1, 1, 1], S, true));
    assert_eq!(transition(0, terminal, 4, true), (terminal, 0, true));
    assert_eq!(transition(0, initial, 3, false), ([0, 0, 1, 1], -S, true));
    for a in 0..4 {
        let (state, reward, done) = transition(1, [0; 4], a, false);
        assert_eq!(reward, [6, 23, 11, 7][a] * (S / 16));
        assert_eq!(state[a], 1);
        assert!(!done);
        assert_eq!(transition(1, state, a, false).1, reward - S / 4);
    }
}

#[test]
fn one_active_row_keeps_six_denominator_and_exact_capture_encoding() {
    let mut x = input(0, 1);
    x.initial_actor[147] = 40 * G; // Deterministic submit regardless of seed.
    let out = run(&x, sha, true);
    assert_eq!(out.final_state.actor, x.initial_actor);
    let mut expected_critic = [0i64; CRITIC];
    expected_critic[96] = -(G / 384);
    assert_eq!(out.final_state.critic, expected_critic);
    assert_eq!(out.critic_export[96], -171);
    assert_eq!(out.final_state.state, [0, 0, 1, 1]);
    assert!(out.final_state.done);
    assert_eq!(out.journal.return_units, -S);
    assert_eq!((out.journal.slots, out.journal.interactions, out.journal.padded), (6, 1, 5));
    assert_eq!((out.journal.episodes, out.journal.updates, out.journal.resets, out.journal.successes), (1, 1, 1, 0));
    assert_eq!(out.final_state.rng, 6);
    let audit = &out.audit[0];
    assert_eq!(audit.episode, 0);
    assert_eq!(audit.trace.len(), 6);
    let mut tape = CounterRng::new(A2C, x.rng_seed, x.challenge, sha);
    assert_eq!(audit.trace[0], [0, 0, 0, 0, 0, 1, 3, -S, 0, 0, 1, 1, 0, 1, 0, tape.next_u32() as i64]);
    for t in 1..6 {
        assert_eq!(audit.trace[t][15], tape.next_u32() as i64);
        assert_eq!(&audit.trace[t][..15], &[0, t as i64, 0, 0, 1, 1, 4, 0, 0, 0, 1, 1, 1, 1, 0]);
    }
    // Independent, literal serialization of this analytically solved episode.
    let mut stages = b"zkRL/A2C-SP1/stages/v1\0".to_vec();
    stages.extend_from_slice(&1u32.to_le_bytes());
    for v in [0i64, 0, 0, 40 * S, 0] { stages.extend_from_slice(&v.to_le_bytes()); }
    for m in [0u64, 0, 0, P] { stages.extend_from_slice(&m.to_le_bytes()); }
    for v in [-G, -G, 0, 0, 0, 0, G] { stages.extend_from_slice(&v.to_le_bytes()); }
    stages.extend_from_slice(&[0; 64 * 8]); // Four hidden/pre vectors.
    stages.extend_from_slice(&[0; ACTOR * 8]);
    stages.extend_from_slice(&[0; 96 * 8]);
    stages.extend_from_slice(&G.to_le_bytes());
    assert_eq!(audit.stages_sha256, sha(&stages));
    let mut model = b"zkRL/A2C-fixed/state/v1\0".to_vec();
    model.extend_from_slice(&16u32.to_le_bytes());
    model.extend_from_slice(&1u64.to_le_bytes());
    for &v in &x.initial_actor { model.extend_from_slice(&v.to_le_bytes()); }
    for &v in &expected_critic { model.extend_from_slice(&v.to_le_bytes()); }
    assert_eq!(audit.model_sha256, sha(&model));
}

#[test]
fn recommendation_administrative_cut_is_not_terminal() {
    let mut x = input(1, 1);
    x.initial_actor[145] = 40 * G; // Always action 1.
    let out = run(&x, sha, true);
    assert_eq!(out.final_state.state, [0, 6, 0, 0]);
    assert!(!out.final_state.done);
    assert_eq!(out.journal.return_units, 78 * S / 16);
    assert_eq!((out.journal.interactions, out.journal.padded, out.journal.successes), (6, 0, 0));
    assert_eq!(out.audit[0].trace[5][13], 0);
    assert_eq!(out.final_state.actor, x.initial_actor);
    // Closed finite-return sum, independent of the engine's backward recursion.
    let mut sum = 0i64;
    for t in 0..6 {
        for next in t..6 {
            let k = (next - t) as u32;
            sum += (23 - 4 * next as i64) * (G / 16) * 7i64.pow(k) / 8i64.pow(k);
        }
    }
    assert_eq!(out.final_state.critic[96], -(-sum).div_euclid(384));
    assert!(out.final_state.critic[..96].iter().all(|&v| v == 0));
}

#[test]
fn successful_login_counts_once_and_environment_resets_only() {
    let mut x = input(0, 2);
    // logits=[40, 0, 80*credential0, 120*credential1] in real units:
    // action 0, then 2, then 3 are deterministic for every possible draw.
    x.initial_actor[0] = G;
    x.initial_actor[5] = G;
    x.initial_actor[80 + 2 * 16] = 80 * G;
    x.initial_actor[80 + 3 * 16 + 1] = 120 * G;
    x.initial_actor[144] = 40 * G;
    let out = run(&x, sha, true);
    assert_eq!(out.final_state.actor, x.initial_actor);
    assert_eq!(out.final_state.state, [1, 1, 1, 1]);
    assert!(out.final_state.done);
    assert_eq!(out.journal.successes, 2);
    assert_eq!(out.journal.return_units, 2 * (S - S / 8));
    assert_eq!((out.journal.interactions, out.journal.padded, out.journal.resets), (6, 6, 2));
    for audit in &out.audit {
        assert_eq!(audit.trace.iter().map(|r| r[6]).collect::<Vec<_>>(), [0, 2, 3, 4, 4, 4]);
        assert_eq!(audit.trace[0][12], 0);
        assert_eq!(&audit.trace[0][2..6], &[0, 0, 0, 1]);
        assert_eq!(audit.trace[2][7], S);
    }
    assert_ne!(out.audit[0].model_sha256, out.audit[1].model_sha256);
    assert_ne!(out.final_state.critic[96], 0);
}

thread_local! { static HASH_CALLS: Cell<usize> = const { Cell::new(0) }; }

fn counted_sha(bytes: &[u8]) -> [u8; 32] {
    HASH_CALLS.with(|count| count.set(count.get() + 1));
    sha(bytes)
}

#[test]
fn capture_quiet_equality_no_per_episode_quiet_hashes_and_resets() {
    for kind in [0, 1] {
        let mut x = input(kind, 3);
        x.seed = u64::MAX;
        HASH_CALLS.with(|c| c.set(0));
        let quiet = run(&x, counted_sha, false);
        assert_eq!(HASH_CALLS.with(Cell::get), 4 + (6 * x.episodes as usize + 7) / 8);
        HASH_CALLS.with(|c| c.set(0));
        let captured = run(&x, counted_sha, true);
        assert_eq!(HASH_CALLS.with(Cell::get), 4 + (6 * x.episodes as usize + 7) / 8 + 2 * x.episodes as usize);
        assert_eq!(quiet.journal, captured.journal);
        assert_eq!(quiet.final_state, captured.final_state);
        assert_eq!(quiet.actor_export, captured.actor_export);
        assert_eq!(quiet.critic_export, captured.critic_export);
        assert!(quiet.audit.is_empty());
        assert_eq!(quiet.audit.capacity(), 0);
        assert_eq!(captured.audit.len(), 3);
        assert_eq!(captured.final_state.rng, 6 * x.episodes as u64);
        assert_eq!(captured.final_state.updates, 3);
        assert_eq!(captured.final_state.resets, 3);
        assert_eq!(captured.final_state.interactions + captured.final_state.padded, 18);
        for (ep, audit) in captured.audit.iter().enumerate() {
            assert_eq!(audit.episode, ep as u32);
            assert_eq!(&audit.trace[0][2..6], if kind == 0 { &[0, 0, 0, 1] } else { &[0, 0, 0, 0] });
            assert_eq!(audit.trace[0][12], 0);
            for (t, row) in audit.trace.iter().enumerate() {
                assert_eq!((row[0], row[1], row[14]), (ep as i64, t as i64, ep as i64));
            }
        }
    }
}

#[test]
fn canonical_config_state_and_commitment_domains() {
    let mut x = input(0, 1);
    x.seed = 0x0807060504030201;
    x.initial_actor[0] = -123;
    x.initial_actor[147] = 40 * G;
    let mut expected = b"zkDRL/ctc/a2c/config/v2\0".to_vec();
    for v in [203u32, 0, 1, 16, 40, 32, 48, 16, 6, 7, 8, 1, 32, 64, 128, 256, 32] {
        expected.extend_from_slice(&v.to_le_bytes());
    }
    expected.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
    expected.extend_from_slice(&[0x44; 32]);
    for &v in x.initial_actor.iter().chain(&x.initial_critic) { expected.extend_from_slice(&v.to_le_bytes()); }
    expected.extend_from_slice(&[0x11; 32]);
    expected.extend_from_slice(&[0x22; 32]);
    expected.extend_from_slice(&[
        0x39,0x78,0x6b,0x3a,0x4a,0x9f,0xfc,0x46,0x84,0x2a,0xe9,0x0d,0x55,0xbe,0xed,0xab,
        0xa8,0x4f,0x11,0x56,0xcf,0xd3,0x65,0xe7,0xee,0xb8,0xb3,0xf4,0x3b,0x0f,0x34,0x8d,
    ]);
    assert_eq!(expected.len(), 2188);
    assert_eq!(config_bytes(&x), expected);
    let out = run(&x, sha, false);
    assert_eq!(out.journal.config_commit, sha(&expected));
    let state = &out.final_state;
    let encoded = state_bytes(state);
    assert_eq!(encoded.len(), 2036);
    assert_eq!(&encoded[..8], &(-123i64).to_le_bytes());
    assert_eq!(&encoded[1960..1968], &state.rng.to_le_bytes());
    for i in 0..4 { assert_eq!(&encoded[1968 + i * 8..1976 + i * 8], &state.state[i].to_le_bytes()); }
    assert_eq!(&encoded[2000..2004], &1u32.to_le_bytes());
    assert_eq!(&encoded[2004..2012], &(-S).to_le_bytes());
    for (i, v) in [1u32, 1, 1, 1, 5, 0].iter().enumerate() {
        assert_eq!(&encoded[2012 + i * 4..2016 + i * 4], &v.to_le_bytes());
    }
    let mut initial = b"zkDRL/ctc/a2c/initial/v2\0".to_vec();
    initial.extend_from_slice(&out.journal.config_commit);
    initial.extend_from_slice(&x.config_salt);
    for &v in x.initial_actor.iter().chain(&x.initial_critic) { initial.extend_from_slice(&v.to_le_bytes()); }
    assert_eq!(out.journal.initial_commit, sha(&initial));
    let mut final_bytes = b"zkDRL/ctc/a2c/final/v2\0".to_vec();
    final_bytes.extend_from_slice(&out.journal.config_commit);
    final_bytes.extend_from_slice(&x.challenge);
    final_bytes.extend_from_slice(&x.final_salt);
    final_bytes.extend_from_slice(&encoded);
    assert_eq!(out.journal.final_commit, sha(&final_bytes));
    for mutate in 0..3 {
        let mut changed = x.clone();
        match mutate { 0 => changed.config_salt[0] ^= 1, 1 => changed.final_salt[0] ^= 1, _ => changed.challenge[0] ^= 1 }
        let other = run(&changed, sha, false);
        assert_eq!(other.final_state, out.final_state);
        if mutate == 2 {
            assert_eq!(other.journal.config_commit, out.journal.config_commit);
            assert_eq!(other.journal.initial_commit, out.journal.initial_commit);
        } else {
            assert_ne!(other.journal.config_commit, out.journal.config_commit);
            assert_ne!(other.journal.initial_commit, out.journal.initial_commit);
        }
        assert_ne!(other.journal.final_commit, out.journal.final_commit);
    }
    let mut malformed = state.clone();
    malformed.actor.pop();
    rejects(|| { state_bytes(&malformed); });
}

#[test]
fn config_binds_private_fields_but_not_challenge() {
    let x = input(1, 2);
    let config = config_bytes(&x);
    let original = run(&x, sha, false);
    for which in 0..7 {
        let mut changed = x.clone();
        match which {
            0 => changed.rng_seed[0] ^= 1,
            1 => changed.initial_actor[0] += 1,
            2 => changed.initial_critic[0] += 1,
            3 => changed.seed += 1,
            4 => changed.config_salt[0] ^= 1,
            5 => changed.final_salt[0] ^= 1,
            _ => changed.episodes += 1,
        }
        assert_ne!(config_bytes(&changed), config);
        let other = run(&changed, sha, false);
        assert_ne!(other.journal.config_commit, original.journal.config_commit);
        assert_ne!(other.journal.initial_commit, original.journal.initial_commit);
    }
    let mut changed = x.clone();
    changed.challenge[0] ^= 1;
    assert_eq!(config_bytes(&changed), config);
    let other = run(&changed, sha, false);
    assert_eq!(other.journal.config_commit, original.journal.config_commit);
    assert_eq!(other.journal.initial_commit, original.journal.initial_commit);
    assert_ne!(other.journal.final_commit, original.journal.final_commit);
    assert_eq!(other.final_state.rng, original.final_state.rng);
}

#[test]
fn tape_repeatability_prefixes_and_padding_cursor() {
    for kind in [0, 1] {
        let mut x = input(kind, 3);
        if kind == 0 { x.initial_actor[147] = 40 * G; } // Guaranteed terminal then five pads.
        let out = run(&x, sha, true);
        assert_eq!(out, run(&x, sha, true));
        assert_eq!(out.final_state.rng, 18);
        if kind == 0 { assert_eq!(out.journal.padded, 15); }
        let mut tape = CounterRng::new(A2C, x.rng_seed, x.challenge, sha);
        for audit in &out.audit {
            for row in &audit.trace { assert_eq!(row[15], tape.next_u32() as i64); }
        }
        assert_eq!(tape.position(), out.final_state.rng);
        x.episodes = 1;
        x.seed = u64::MAX;
        x.config_salt[0] ^= 1;
        x.final_salt[0] ^= 1;
        let prefix = run(&x, sha, true);
        assert_eq!(prefix.final_state.rng, 6);
        assert_eq!(prefix.audit[0], out.audit[0]);
    }
}

#[test]
fn journal_fixed_width_order() {
    let j = run(&input(1, 1), sha, false).journal;
    let mut expected = 203u32.to_le_bytes().to_vec();
    for bytes in [j.config_commit, j.initial_commit, j.final_commit, j.challenge] {
        expected.extend_from_slice(&bytes);
    }
    for v in [j.kind, j.episodes, j.slots, j.interactions, j.updates, j.resets, j.padded, j.successes] {
        expected.extend_from_slice(&v.to_le_bytes());
    }
    expected.extend_from_slice(&j.return_units.to_le_bytes());
    assert_eq!(expected.len(), 172);
    assert_eq!(bincode::serialize(&j).unwrap(), expected);
}
