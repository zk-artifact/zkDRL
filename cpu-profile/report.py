#!/usr/bin/env python3
"""Parse retained captures; publish CPU_PROFILE.json only after every gate passes."""
import datetime
import json
from pathlib import Path
import struct
import sys
sys.dont_write_bytecode = True
from checksource import ROOT, check, require, sha
from source_layout import LEAVES
RESULTS = ROOT / 'results'
VARIANTS = ('original', 'control', 'instrumented')
ELVES = {
    'original': ROOT.parent/'dqn/elf/zkdrl-ctc-dqn-guest',
    'control': ROOT/'elf-control/zkdrl-profile-dqn-guest',
    'instrumented': ROOT/'elf-instrumented/zkdrl-profile-dqn-guest',
}
HOSTS = {v: ROOT/'bin'/('host-' + v) for v in ('control', 'instrumented')}

def load(name):
    return json.loads((RESULTS/name).read_text())

def save(name, value):
    with (RESULTS/name).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')

def journal_bytes(j):
    # Exact fixed-integer bincode ABI of the current 148-byte Journal.
    data = struct.pack('<I', j['schema'])
    for k in ('config_commit', 'final_commit', 'challenge'):
        require(isinstance(j[k], list) and len(j[k]) == 32, f'journal {k} byte array')
        data += bytes(j[k])
    data += struct.pack('<IIIIqIIIIII', *[j[k] for k in (
        'kind','scale','slots','train','return_units','interactions','updates',
        'resets','padded','target_copies','replay_wraps')])
    require(len(data) == 148, 'journal ABI')
    return data

def capture(name, operation, variant, input_path):
    raw = (RESULTS/(name+'.stdout.log')).read_text()
    # Preserve all stdout (including guest marker output). Select our single host envelope.
    found = []
    for line in raw.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get('operation') == operation:
            found.append(value)
    require(len(found) == 1, f'{name}: require exactly one host envelope')
    d = found[0]
    command = load(name+'.command.json')
    expected_args = [operation, variant, str(input_path)]
    require(command['returncode'] == 0 and command['command'][-3:] == expected_args, f'{name}: failed/wrong command')
    require(d['cryptographic_proof'] is False and d['variant'] == variant, f'{name}: identity')
    require(d['input_sha256'] == sha(input_path), f'{name}: input binding')
    require(d['journal_bytes'] == 148, f'{name}: journal length')
    if operation == 'native':
        require(d['capture'] is False and d['markers_feature'] == (variant == 'instrumented'), f'{name}: native feature/capture')
        require(d['output']['audit'] == [] and d['output']['state_chain'] is None and d['output']['trajectory_chain'] is None, f'{name}: capture(false)')
        j = d['output']['journal']
    else:
        require(d['elf_sha256'] == sha(ELVES[variant]), f'{name}: ELF binding')
        require(d['sdk'] == '6.9.0' and d['sdk_default_features'] is False and d['sdk_features'] == ['profiling'], f'{name}: SDK')
        r = d['report']
        require(r['exit_code'] == 0, f'{name}: guest exit')
        require(sum(r['opcode_counts'].values()) == d['instruction_cycles'] > 0, f'{name}: opcode sum')
        require(sum(r['syscall_counts'].values()) == d['syscalls'], f'{name}: syscall sum')
        j = d['journal']
    require(journal_bytes(j).hex() == d['journal_hex'], f'{name}: actual canonical journal bytes')
    return d

def group(prefix, input_path, reference=None):
    input_data = json.loads(input_path.read_text())
    slots = input_data['slots']
    require(input_data['schema'] == 202 and input_data['kind'] == 1 and input_data['train'] == 1, 'exposure training schema-202 only')
    require(len(input_data['initial_coarse']) == 148 and input_data['scale'] in (16,256), 'H16 default profile')
    natives = {v: capture(f'{prefix}-native-{v}', 'native', v, input_path) for v in HOSTS}
    output = natives['control']['output']
    require(output == natives['instrumented']['output'], 'complete native outputs differ')
    if reference is not None:
        for field in ('journal','final_state','coarse_export'):
            require(field in reference and output[field] == reference[field], f'raw Python reference {field} mismatch')
    j = output['journal']
    require(j['slots'] == slots and j['interactions'] == slots and j['updates'] == slots-31 and j['resets'] == slots//6 and j['padded'] == 0 and j['target_copies'] == (slots-31)//32 and j['replay_wraps'] == max(0,slots-64), 'learner counters')
    require(output['final_state']['rng'] == slots*10, 'all ten SHA-counter draws per slot')
    executions = {v: capture(f'{prefix}-{v}', 'execute', v, input_path) for v in VARIANTS}
    for v,d in executions.items():
        require(d['journal'] == j and d['journal_hex'] == natives['control']['journal_hex'], f'{prefix}-{v}: actual journal differs from native/reference')
        if v != 'instrumented':
            require(not d['report']['cycle_tracker'] and not d['report']['invocation_tracker'], f'{v}: unexpected markers')
    marked = executions['instrumented']
    c, inv = marked['report']['cycle_tracker'], marked['report']['invocation_tracker']
    expected = {label: slots for label in LEAVES}
    expected.update(init_config_rngkey=1, slot_reset_counters=2*slots,
                    batch_forward_target_backprop=slots-31, master_refresh_targetcopy=slots-31,
                    final_state_export_encoding_sha_commit=1, total_run=1, empty_marker=32)
    require(inv == expected and set(c) == set(expected), 'marker labels/invocations')
    require(all(type(x) is int and x > 0 for x in c.values()), 'positive integral tracker deltas')
    leaf_sum = sum(c[k] for k in LEAVES)
    total = marked['instruction_cycles']
    # Calibration is OUTSIDE total_run; total_run is a nested diagnostic only.
    require(leaf_sum <= c['total_run'] and c['total_run'] + c['empty_marker'] <= total, 'nonoverlap/accounting')
    residual = total - leaf_sum - c['empty_marker']
    require(leaf_sum + c['empty_marker'] + residual == total, 'leaf+calibration+residual reconciliation')
    ecalls = {v: d['report']['opcode_counts']['ECALL'] for v,d in executions.items()}
    require(ecalls['original'] == ecalls['control'], 'control ECALL drift')
    require(ecalls['instrumented'] - ecalls['control'] == 2*sum(inv.values()), 'marker pair ECALL delta')
    sha_calls = {v: {k: d['report']['syscall_counts'][k] for k in ('SHA_EXTEND','SHA_COMPRESS')} for v,d in executions.items()}
    require(sha_calls['original'] == sha_calls['control'] == sha_calls['instrumented'], 'SHA syscall drift')
    # Default H16/C64 schema-202 messages: config 1400B, key 89B, RNG block 63B,
    # final 10563B. SHA padding adds 9 bytes; each RNG block needs TWO SHA blocks.
    # SP1 additionally hashes the148 public journal bytes on FD_PUBLIC_VALUES
    # and finalizes that hasher at HALT (3more compressions), outside total_run.
    blocks = lambda n: (n+9+63)//64
    expected_sha = blocks(1400) + blocks(89) + ((10*slots+7)//8)*blocks(63) + blocks(10563) + blocks(148)
    require(all(n == expected_sha for n in sha_calls['original'].values()), 'SHA compression count including all RNG blocks')
    return {'executions': executions, 'native': natives, 'leaf_sum': leaf_sum,
            'calibration': c['empty_marker'], 'residual': residual,
            'ecalls': ecalls, 'sha256_syscalls': sha_calls}

def report():
    require(not (RESULTS/'CPU_PROFILE.json').exists(), 'refuse report overwrite')
    source = check()
    require(source == load('source-check.json'), 'source changed since capture')
    provenance = load('provenance.json')
    for path, expected in provenance['sha256'].items():
        require(sha(path) == expected, f'artifact changed: {path}')
    full_input = load('input-full.json')
    short_input = load('input-short.json')
    require(full_input['slots'] == 24000, 'full 4000 episodes required')
    require(short_input == dict(full_input, slots=48), 'short must derive only slots from full; challenge unchanged')
    short = group('short', RESULTS/'input-short.json')
    full = group('full', RESULTS/'input-full.json', load('reference.json'))
    ex = full['executions']
    counts = {v:d['instruction_cycles'] for v,d in ex.items()}
    c = ex['instrumented']['report']['cycle_tracker']
    inv = ex['instrumented']['report']['invocation_tracker']
    total = counts['instrumented']
    def inflation(a,b):
        return {'extra_instructions':counts[a]-counts[b], 'ratio':counts[a]/counts[b], 'percent':100*(counts[a]/counts[b]-1)}
    result = {
        'status':'passed', 'cryptographic_proof':False, 'formal_approved_key':False,
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source_relation':source, 'provenance':provenance,
        'gates':{'short_8_episode_native_and_three_actual_journals':True,
                 'full_4000_episode_native_control_instrumented_equal_python_state_export_journal':True,
                 'full_three_actual_guest_journals_equal_python_148_bytes':True,
                 'marker_invocations_ecall_delta_sha_counts':True,
                 'source_relation_is_not_formal_refinement':True},
        'instruction_counts':counts,
        'marker_inflation':{a+'_vs_'+b:inflation(a,b) for a,b in [('instrumented','control'),('control','original'),('instrumented','original')]},
        'share_denominator':{'description':'whole instrumented execution; nested total_run excluded from sum','instructions':total},
        'leaves':{k:{'instructions':c[k], 'invocations':inv[k], 'percent':100*c[k]/total} for k in LEAVES},
        'accounting':{'leaf_sum':full['leaf_sum'], 'calibration_outside_total_run':full['calibration'], 'residual':full['residual'], 'total':total},
        'nested_total_run':{'instructions':c['total_run'], 'invocations':1,
                            'residual_inside':c['total_run']-full['leaf_sum'],
                            'residual_outside_excluding_calibration':total-c['total_run']-full['calibration'], 'added_to_leaf_sum':False},
        'calibration':{'intervals':32,'instructions':full['calibration'],'mean_interval':full['calibration']/32,'subtracted':False},
        'sha256_syscalls':full['sha256_syscalls'], 'ecall_instructions':full['ecalls'],
        'syscall_counts':{v:d['report']['syscall_counts'] for v,d in ex.items()},
        'short_instruction_counts':{v:d['instruction_cycles'] for v,d in short['executions'].items()},
        'limitations':[
            'RISC instruction counts and tracker global-clock deltas describe CPU guest execution, not GPU proving time or precompile AIR rows.',
            'SHA256 syscalls are separate counts; their host/precompile internal work is not expanded into RISC instructions.',
            'SDK syscall_counts is AIR-mapped, not a complete runtime syscall census; use ECALL for aggregate instruction counts.',
            'Markers affect code layout, inlining and registers. Inflation is measured, not subtracted. Empty intervals are not universal pair-overhead corrections.',
            'Residual includes loop/control/row construction, marker paths, runtime and journal I/O. total_run is nested and never double-counted.',
            'Guest exports journal bytes only; full state/export comparisons use native capture(false). Short fixture is nonformal and checked against native, not an independent Python short reference.',
            'No proof generated and no approved program/key registered. Existing S16/S256 core-gate cost points and GPU results are untouched.'
        ],
        'raw_capture_sha256':{str(p):sha(p) for p in sorted(RESULTS.iterdir()) if p.is_file()},
    }
    save('CPU_PROFILE.json', result)
    print(json.dumps({'status':'passed','report':str(RESULTS/'CPU_PROFILE.json'),'instruction_counts':counts}))

if __name__ == '__main__':
    report()
