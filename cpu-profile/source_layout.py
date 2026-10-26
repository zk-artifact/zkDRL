"""Auditable marker-only insertion recipe for the frozen schema-202 source."""
LEAVES = [
    'init_config_rngkey', 'slot_reset_counters', 'random_tape_draws',
    'policy_forward_action', 'transition_guards', 'replay_insert_select',
    'batch_forward_target_backprop', 'master_refresh_targetcopy',
    'final_state_export_encoding_sha_commit',
]
HEADER = 'include!("markers.rs");\n'

def instrument(source):
    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError(f'anchor must occur exactly once: {old!r}')
        source = source.replace(old, new)
    def before(anchor, *markers):
        indent = anchor[:len(anchor)-len(anchor.lstrip())]
        replace(anchor, ''.join(indent + m + '\n' for m in markers) + anchor)
    def start(label): return f'profile_start!("{label}");'
    def end(label): return f'profile_end!("{label}");'
    replace('use serde::{Deserialize, Serialize};\n', 'use serde::{Deserialize, Serialize};\n' + HEADER)
    before('    // All rows originate here, AFTER a checked in-guest environment transition.', start(LEAVES[5]))
    replace('if st.interactions < WARMUP { return [-1; B]; }',
            'if st.interactions < WARMUP { ' + end(LEAVES[5]) + ' return [-1; B]; }')
    before('    let qw = &st.online;', end(LEAVES[5]), start(LEAVES[6]))
    before('    for (m, g) in st.master.iter_mut().zip(grad) {', end(LEAVES[6]), start(LEAVES[7]))
    before('    indices\n}', end(LEAVES[7]))
    before('    assert_eq!(input.schema, SCHEMA);', 'profile_calibrate!();', start('total_run'), start(LEAVES[0]))
    before('    // Counters increment at most once per slot: all remain <= 24000.', end(LEAVES[0]))
    before('        if slot % HORIZON == 0 {', start(LEAVES[1]))
    before('        // A fixed ten draws per slot, even in padding/evaluation. No host RNG.', end(LEAVES[1]), start(LEAVES[2]))
    before('        let (q, _) = forward_bounded(&st.online, &st.state, bits);', end(LEAVES[2]), start(LEAVES[3]))
    before('        let (successor, reward, terminal) = env_step(input.kind, before, action, &input.params, scale, done_before);', end(LEAVES[3]), start(LEAVES[4]))
    before('        let mut indices = [-1; B];', end(LEAVES[4]))
    before('        if done_before { st.padded += 1; }', start(LEAVES[1]))
    before('        // Native comparison only. The guest skips all per-slot serialization', end(LEAVES[1]))
    before('    let coarse_export = st.online.to_vec();', start(LEAVES[8]))
    replace('    Output { journal, final_state: st.into_state(), coarse_export,',
            '    let output = Output { journal, final_state: st.into_state(), coarse_export,')
    replace('        trajectory_chain: capture.then_some(trajectory_chain), audit }\n}',
            '        trajectory_chain: capture.then_some(trajectory_chain), audit };\n'
            + '    ' + end(LEAVES[8]) + '\n    ' + end('total_run') + '\n    output\n}')
    return source

def strip_markers(source):
    import re
    source = source.replace(HEADER, '')
    source = re.sub(r'^ *profile_(?:start|end)!\("[a-z_]+"\);\n', '', source, flags=re.M)
    source = source.replace('    profile_calibrate!();\n', '')
    source = source.replace('profile_end!("replay_insert_select"); return', 'return')
    source = source.replace('    let output = Output { journal,', '    Output { journal,')
    source = source.replace('trajectory_chain: capture.then_some(trajectory_chain), audit };\n    output\n}',
                            'trajectory_chain: capture.then_some(trajectory_chain), audit }\n}')
    return source

MARKERS = '''// Only marker emission is feature-gated; native marker calls are no-ops.
macro_rules! profile_start {
    ($label:literal) => {
        #[cfg(all(feature = "markers", target_os = "zkvm"))]
        println!(concat!("cycle-tracker-report-start: ", $label));
    };
}
macro_rules! profile_end {
    ($label:literal) => {
        #[cfg(all(feature = "markers", target_os = "zkvm"))]
        println!(concat!("cycle-tracker-report-end: ", $label));
    };
}
macro_rules! profile_calibrate {
    () => {
        #[cfg(all(feature = "markers", target_os = "zkvm"))]
        for _ in 0..32 {
            profile_start!("empty_marker");
            profile_end!("empty_marker");
        }
    };
}
'''
