import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / 'numerics'
sys.path.insert(0, str(ROOT))
import runner as r


def fixture(algo, kind=0, episodes=12):
    i = (r.dr if algo == 'dqn' else r.ar).fixture(kind=kind, seed=5000, episodes=episodes)
    i['challenge'] = list(range(1, 33))  # issued test value, never created by harness
    return i


def job(algo, condition, i):
    n = i['slots']//6 if algo == 'dqn' else i['episodes']
    return dict(id=f'{algo}-{condition}', algorithm=algo, condition=condition,
                kind=i['kind'], seed=i['seed'], episodes=n, checkpoints=[0, n])


class Harness(unittest.TestCase):
    def test_all_conditions_pairing_and_grid(self):
        for algo, conditions in r.CONDITIONS.items():
            i = fixture(algo)
            records, banks = [], []
            for condition in conditions:
                with self.subTest(algo=algo, condition=condition):
                    # The harness must use no legacy random initializer or tape.
                    with patch.object(np.random, 'default_rng', side_effect=AssertionError('old RNG')):
                        record, arrays = r.run(job(algo, condition, i), i)
                    self.assertEqual(record['status'], 'completed', record.get('error'))
                    self.assertEqual(record['completed_episodes'], 12)
                    self.assertEqual([v['episode'] for v in record['checkpoints']], [0, 12])
                    self.assertEqual(record['evaluation_counts'], dict(available=2, unavailable=0))
                    self.assertEqual(record['evaluation_failure_count'], 0)
                    self.assertEqual(record['evaluation_failures'], [])
                    for cp in record['checkpoints']:
                        self.assertEqual(cp['evaluation_status'], 'available')
                        self.assertTrue({'return', 'success', 'actions'} <= cp.keys() if algo == 'dqn'
                                        else {'greedy', 'stochastic'} <= cp.keys())
                    self.assertEqual(record['consumed_words'], 12*6*(10 if algo == 'dqn' else 1))
                    self.assertFalse(record['cryptographic_proof'])
                    self.assertFalse(record['guest_execution'])
                    json.dumps(record, allow_nan=False)
                    self.assertTrue(all(a.dtype.kind != 'O' for a in arrays.values()))
                    if algo == 'dqn':
                        initial = arrays['checkpoint_params'][0]
                        divisor = (1 if condition == 'main_F' else 16 if condition == 'main_N'
                                   else 128*16 if condition == 'residual'
                                   else 128*(256 if condition == 'precision_S256' else 16)**2)
                        np.testing.assert_array_equal(initial/divisor, arrays['initial_real'])
                    else:
                        divisor = 1 if condition == 'main_F' else r.fm.G
                        np.testing.assert_array_equal(arrays['checkpoint_actor'][0]/divisor, arrays['initial_actor_real'])
                        np.testing.assert_array_equal(arrays['checkpoint_critic'][0]/divisor, arrays['initial_critic_real'])
                    if condition == 'main_N' and algo == 'a2c':
                        self.assertEqual(record['diagnostics']['master_residual_max'], 0)
                        self.assertTrue(np.all(arrays['final_actor'] % (1 << 24) == 0))
                        self.assertTrue(np.all(arrays['final_critic'] % (1 << 24) == 0))
                    records.append(record)
                    banks.append(arrays)
            for key in ('tape_sha256', 'initial_grid_sha256', 'baseline_input_sha256'):
                self.assertEqual(len({rec[key] for rec in records}), 1)
            for bank in banks[1:]:
                np.testing.assert_array_equal(bank['raw_tape'], banks[0]['raw_tape'])
                np.testing.assert_array_equal(bank['initial_grid'], banks[0]['initial_grid'])
            self.assertEqual(banks[0]['raw_tape'].dtype.str, '<u4')

    def test_dqn_reference(self):
        for kind in (0, 1):
            i = fixture('dqn', kind, 24)
            for condition in ('main_Q', 'precision_S256'):
                with self.subTest(kind=kind, condition=condition):
                    rec, arrays = r.run(job('dqn', condition, i), i)
                    effective = copy.deepcopy(i)
                    if condition == 'precision_S256':
                        effective['scale'] = 256
                        effective['initial_coarse'] = [v*16 for v in i['initial_coarse']]
                        effective['params'] = [v*16 for v in i['params']]
                    expected = r.dr.run(effective, capture=False)
                    self.assertEqual(rec['status'], 'completed', rec.get('error'))
                    self.assertEqual(rec['causal_state'], expected['final_state'])
                    self.assertEqual(rec['coarse_export'], expected['coarse_export'])
                    np.testing.assert_array_equal(arrays['checkpoint_params'][0],
                                                  np.array(effective['initial_coarse'])*128*effective['scale'])

    def test_a2c_reference_and_binding_restoration(self):
        for kind in (0, 1):
            i = fixture('a2c', kind, 12)
            expected = r.ar.run(i, capture=False)
            binding = (r.fm.masses, r.trainer.masses, r.fixed_eval.masses)
            for condition in ('main_Q', 'operator_O1', 'operator_O2', 'main_Q'):
                rec, arrays = r.run(job('a2c', condition, i), i)
                self.assertEqual(binding, (r.fm.masses, r.trainer.masses, r.fixed_eval.masses))
                self.assertEqual(rec['status'], 'completed', rec.get('error'))
                np.testing.assert_array_equal(arrays['traces'][:, -1], arrays['raw_tape'])
                if condition == 'main_Q':
                    self.assertEqual(rec['causal_state'], expected['final_state'])
                    self.assertEqual(rec['actor_export'], expected['actor_export'])
                    self.assertEqual(rec['critic_export'], expected['critic_export'])

    def test_padding_and_challenge(self):
        for algo in ('dqn', 'a2c'):
            i = fixture(algo, episodes=2)
            # Always submit from initial browser state: terminal failure, then padding.
            if algo == 'dqn':
                i['initial_coarse'] = [0]*148
                i['initial_coarse'][-1] = 16*20
            else:
                i['initial_actor'] = [0]*148
                i['initial_actor'][-1] = 40*r.fm.G
                i['initial_critic'] = [0]*97
            rec, bank = r.run(job(algo, 'main_Q', i), i)
            self.assertEqual(rec['status'], 'completed', rec.get('error'))
            self.assertGreater(rec['causal_state']['padded'], 0)
            self.assertEqual(rec['consumed_words'], 120 if algo == 'dqn' else 12)
            changed = copy.deepcopy(i)
            changed['challenge'][0] ^= 1
            other, _ = r.run(job(algo, 'main_Q', changed), changed)
            self.assertNotEqual(rec['tape_sha256'], other['tape_sha256'])
            self.assertEqual(rec['initial_grid_sha256'], other['initial_grid_sha256'])

    def test_operator_contract(self):
        op = r.operators
        self.assertEqual(len(op._table('O1')), 2049)
        self.assertEqual(len(op._table('O2')), 256)
        for variant in ('O1', 'O2'):
            self.assertEqual(op.exp_weight(-32*op.Q-1, variant), 0)
            self.assertEqual(op.exp_weight(0, variant), op.E)
            for x in range(-32*op.Q, 1, 1024):
                self.assertTrue(0 <= op.exp_weight(x, variant) <= op.E)
            self.assertEqual(op.softmax_masses([0]*4, [1]*4, variant), [op.P//4]*4)
            m = op.softmax_masses([0, -33*op.Q, -op.Q, -2*op.Q], [1]*4, variant)
            self.assertEqual(sum(m), op.P)
            self.assertEqual(m[1], 0)
            self.assertEqual(op.categorical(m, m[0]-1), 0)
            self.assertEqual(op.categorical(m, m[0]), 2)
            with self.assertRaises(ValueError):
                op.softmax_masses([op.LOGIT_LIMIT, 0, 0, 0], [1]*4, variant)

    def test_abort_and_unexpected_error(self):
        i = fixture('a2c', episodes=2)
        j = job('a2c', 'operator_O1', i)
        original = r.fm.masses
        with patch.object(r.trainer, 'update_fixed', side_effect=OverflowError('test guard')):
            rec, arrays = r.run(j, i)
        self.assertIs(r.fm.masses, original)
        self.assertEqual(rec['status'], 'arithmetic_guard_abort')
        self.assertEqual(len(arrays['traces']), 6)
        self.assertEqual(rec['failed_attempt']['phase'], 'update')
        with patch.object(r.trainer, 'update_fixed', side_effect=RuntimeError('programming error')):
            with self.assertRaisesRegex(RuntimeError, 'programming error'):
                r.run(j, i)
        self.assertIs(r.fm.masses, original)

    def test_validation(self):
        i = fixture('dqn', episodes=1)
        for field, value in [('seed', 5001), ('kind', 1), ('episodes', 2), ('checkpoints', [1])]:
            j = job('dqn', 'main_Q', i)
            j[field] = value
            with self.assertRaises(ValueError):
                r.run(j, i)
        i['challenge'] = [0]*32
        with self.assertRaises(ValueError):
            r.run(job('dqn', 'main_Q', i), i)

    def test_checkpoint_guards_preserve_training_and_reference(self):
        # Final episode 12 exercises the same checkpoint path as production 4000.
        for algo in ('dqn', 'a2c'):
            i = fixture(algo, episodes=12)
            j = job(algo, 'main_Q', i)
            j['checkpoints'] = [0, 6, 12]
            expected = (r.dr if algo == 'dqn' else r.ar).run(i, capture=False)
            baseline, bank = r.run(j, i)
            owner, name = (r.q, 'evaluation') if algo == 'dqn' else (r.fixed_eval.FixedEvaluator, 'model')
            original = getattr(owner, name)
            for failed_indices in ((0,), (1,), (2,), (0, 1, 2)):
                with self.subTest(algo=algo, failed_indices=failed_indices):
                    calls = []
                    def evaluate(*args, **kwargs):
                        index = len(calls)
                        calls.append(index)
                        if index in failed_indices:
                            raise (OverflowError('checkpoint guard'), FloatingPointError('checkpoint guard'),
                                   ValueError('checkpoint must be finite'))[index]
                        return original(*args, **kwargs)
                    with patch.object(owner, name, new=evaluate):
                        rec, arrays = r.run(j, i)
                    self.assertEqual(rec['status'], 'completed')
                    self.assertIsNone(rec['error'])
                    self.assertIsNone(rec['failed_attempt'])
                    self.assertIsNone(rec['failed_guard'])
                    self.assertEqual(rec['completed_episodes'], 12)
                    self.assertEqual(rec['consumed_words'], 12*6*(10 if algo == 'dqn' else 1))
                    self.assertEqual(rec['causal_state'], expected['final_state'])
                    self.assertEqual(rec['final_state']['updates'], expected['final_state']['updates'])
                    self.assertEqual(rec['trajectory_sha256'], baseline['trajectory_sha256'])
                    self.assertEqual(rec['evaluation_failure_count'], len(failed_indices))
                    self.assertEqual(rec['evaluation_counts'],
                                     dict(available=3-len(failed_indices), unavailable=len(failed_indices)))
                    self.assertEqual(rec['evaluation_failures'],
                                     [rec['checkpoints'][k] for k in failed_indices])
                    for index, cp in enumerate(rec['checkpoints']):
                        if index in failed_indices:
                            self.assertEqual(cp['episode'], j['checkpoints'][index])
                            self.assertEqual(cp['evaluation_status'], 'unavailable')
                            self.assertIn('checkpoint', cp['error'])
                            self.assertIsNone(cp['failed_guard'])
                            self.assertTrue({'return', 'success', 'greedy', 'stochastic'}.isdisjoint(cp))
                        else:
                            self.assertEqual(cp, baseline['checkpoints'][index])
                    # Includes all checkpoint snapshots, tape, updates, and final arrays.
                    for key in bank:
                        np.testing.assert_array_equal(arrays[key], bank[key], err_msg=key)

    def test_checkpoint_programming_errors_propagate_and_restore(self):
        for algo in ('dqn', 'a2c'):
            i = fixture(algo, episodes=2)
            j = job(algo, 'main_Q' if algo == 'dqn' else 'operator_O1', i)
            owner, name = (r.q, 'evaluation') if algo == 'dqn' else (r.fixed_eval.FixedEvaluator, 'model')
            original = getattr(owner, name)
            bindings = (r.fm.bound, r.trainer.bound, r.fm.masses, r.trainer.masses, r.fixed_eval.masses)
            for failure_index in (0, 1):
                for error in (RuntimeError('evaluator bug'), ValueError('invariant'), AssertionError('invariant')):
                    with self.subTest(algo=algo, failure_index=failure_index, error=type(error)):
                        calls = []
                        def evaluate(*args, **kwargs):
                            index = len(calls)
                            calls.append(index)
                            if index == failure_index:
                                raise error
                            return original(*args, **kwargs)
                        with patch.object(owner, name, new=evaluate):
                            with self.assertRaises(type(error)) as caught:
                                r.run(j, i)
                        self.assertIs(caught.exception, error)
                        self.assertEqual(bindings, (r.fm.bound, r.trainer.bound, r.fm.masses,
                                                    r.trainer.masses, r.fixed_eval.masses))

    def test_failed_bound_peak_training_and_evaluation(self):
        i = fixture('a2c', episodes=2)
        j = job('a2c', 'operator_O1', i)
        bindings = (r.fm.bound, r.trainer.bound, r.fm.masses, r.trainer.masses, r.fixed_eval.masses)
        cases = ((np.array([np.iinfo(np.int64).min]), (1 << 62)+1, 1 << 63),
                 (np.array([-(1 << 130)], dtype=object), 1 << 127, 1 << 130),
                 (np.array([-17]), 17, 17))  # strict equality must still fail
        for values, limit, peak in cases:
            legacy_audit = {'injected': 7}
            with self.assertRaises(OverflowError):
                r.fm.bound(values, limit, 'injected', legacy_audit)
            self.assertEqual(legacy_audit, {'injected': 7})
            for phase in ('update', 'checkpoint'):
                for owner in (r.fm, r.trainer):
                    with self.subTest(peak=peak, phase=phase, owner=owner.__name__):
                        def fail(*args):
                            audit = args[-1] if phase == 'update' else None
                            owner.bound([7], limit, 'injected', audit)
                            owner.bound(values, limit, 'injected', audit)
                        target, name = ((r.trainer, 'update_fixed') if phase == 'update'
                                        else (r.fixed_eval.FixedEvaluator, 'model'))
                        with patch.object(target, name, new=fail):
                            rec, arrays = r.run(j, i)
                        detail = dict(name='injected', max_abs=peak, strict_limit=limit, phase=phase)
                        error = f'OverflowError: injected: abs {peak} >= {limit}'
                        if phase == 'update':
                            self.assertEqual(rec['status'], 'arithmetic_guard_abort')
                            self.assertEqual(rec['failed_guard'], detail)
                            self.assertEqual(rec['error'], error)
                            self.assertEqual(rec['failed_attempt']['phase'], 'update')
                            self.assertEqual(rec['numerical_maxima']['injected'], 7)
                            self.assertEqual(rec['completed_episodes'], 0)
                            self.assertEqual(rec['evaluation_failure_count'], 0)
                        else:
                            self.assertEqual(rec['status'], 'completed')
                            self.assertIsNone(rec['failed_guard'])
                            self.assertEqual(rec['evaluation_failure_count'], 2)
                            self.assertEqual(len(arrays['checkpoint_actor']), 2)
                            self.assertEqual(len(arrays['checkpoint_critic']), 2)
                            for failure in rec['evaluation_failures']:
                                self.assertEqual(failure['failed_guard'], detail)
                                self.assertEqual(failure['error'], error)
                            self.assertNotIn('injected', rec['numerical_maxima'])
                        roundtrip = json.loads(r.encoded(rec))
                        saved = (roundtrip['failed_guard'] if phase == 'update' else
                                 roundtrip['evaluation_failures'][0]['failed_guard'])
                        self.assertIs(type(saved['max_abs']), int)
                        self.assertIs(type(saved['strict_limit']), int)
                        self.assertEqual(bindings, (r.fm.bound, r.trainer.bound, r.fm.masses,
                                                    r.trainer.masses, r.fixed_eval.masses))
        # No stale failure detail or patched binding survives into a serial run.
        rec, _ = r.run(job('a2c', 'main_Q', i), i)
        self.assertIsNone(rec['failed_guard'])
        self.assertEqual(rec['evaluation_failures'], [])
        self.assertEqual(rec['causal_state'], r.ar.run(i, capture=False)['final_state'])

    def test_overflow_without_peak_keeps_original_error(self):
        i = fixture('a2c', episodes=2)
        error = OverflowError('no measured values')
        bindings = (r.fm.bound, r.trainer.bound)
        with patch.object(r.trainer, 'update_fixed', side_effect=error):
            rec, _ = r.run(job('a2c', 'main_Q', i), i)
        self.assertEqual(rec['status'], 'arithmetic_guard_abort')
        self.assertEqual(rec['error'], 'OverflowError: no measured values')
        self.assertIsNone(rec['failed_guard'])
        self.assertEqual(bindings, (r.fm.bound, r.trainer.bound))
        # Even a conversion overflow inside bound must retain its own exception.
        with r.guard_context():
            with self.assertRaisesRegex(OverflowError, 'infinity') as caught:
                r.fm.bound([float('inf')], 16, 'conversion')
        self.assertFalse(hasattr(caught.exception, 'failed_guard'))
        self.assertEqual(bindings, (r.fm.bound, r.trainer.bound))

    def test_initialization_guard(self):
        i = fixture('dqn', episodes=1)
        i['initial_coarse'][0] = (1 << 20)-1
        rec, arrays = r.run(job('dqn', 'precision_S256', i), i)
        self.assertEqual(rec['status'], 'arithmetic_guard_abort')
        self.assertEqual(rec['failed_attempt']['phase'], 'initialization')
        self.assertEqual(rec['consumed_words'], 0)
        self.assertEqual(arrays['raw_tape'].size, 60)

    def test_cli(self):
        i = fixture('dqn', episodes=2)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root = Path(temp)
            (root/'job.json').write_text(json.dumps(job('dqn', 'main_Q', i)))
            (root/'input.json').write_text(json.dumps(i))
            proc = subprocess.run([sys.executable, '-B', str(ROOT/'runner.py'), '--job', str(root/'job.json'),
                                   '--input', str(root/'input.json'), '--out', str(root/'out')], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual({p.name for p in (root/'out').iterdir()}, {'record.json', 'arrays.npz', 'RESULT.json'})
            result = json.loads((root/'out/RESULT.json').read_text())
            for name, digest in result['files'].items():
                self.assertEqual(hashlib.sha256((root/'out'/name).read_bytes()).hexdigest(), digest)
            with np.load(root/'out/arrays.npz', allow_pickle=False) as arrays:
                self.assertEqual(arrays['raw_tape'].size, 120)
            i['challenge'] = [0]*32
            (root/'input.json').write_text(json.dumps(i))
            proc = subprocess.run([sys.executable, '-B', str(ROOT/'runner.py'), '--job', str(root/'job.json'),
                                   '--input', str(root/'input.json'), '--out', str(root/'invalid')], capture_output=True, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn('Traceback', proc.stderr)
            self.assertFalse((root/'invalid').exists())

    def test_cli_final_checkpoint_availability_and_nonzero_bug(self):
        script = '''
import sys
import runner as r
algo, error, job_path, input_path, output_path = sys.argv[1:]
owner, name = (r.q, 'evaluation') if algo == 'dqn' else (r.fixed_eval.FixedEvaluator, 'model')
original = getattr(owner, name)
calls = 0
def evaluate(*args, **kwargs):
    global calls
    calls += 1
    if calls == 2:
        raise (OverflowError if error == 'guard' else RuntimeError)('final checkpoint injection')
    return original(*args, **kwargs)
setattr(owner, name, evaluate)
sys.argv = ['runner.py', '--job', job_path, '--input', input_path, '--out', output_path]
r.main()
'''
        for algo in ('dqn', 'a2c'):
            i = fixture(algo, episodes=2)
            expected = (r.dr if algo == 'dqn' else r.ar).run(i, capture=False)
            with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
                root = Path(temp)
                (root/'job.json').write_text(json.dumps(job(algo, 'main_Q', i)))
                (root/'input.json').write_text(json.dumps(i))
                for error in ('guard', 'bug'):
                    with self.subTest(algo=algo, error=error):
                        out = root/error
                        proc = subprocess.run(
                            [sys.executable, '-B', '-c', script, algo, error, str(root/'job.json'),
                             str(root/'input.json'), str(out)], cwd=ROOT, capture_output=True, text=True)
                        if error == 'bug':
                            self.assertNotEqual(proc.returncode, 0)
                            self.assertIn('RuntimeError: final checkpoint injection', proc.stderr)
                            self.assertFalse(out.exists())
                            continue
                        self.assertEqual(proc.returncode, 0, proc.stderr)
                        rec = json.loads((out/'record.json').read_text())
                        self.assertEqual(rec['status'], 'completed')
                        self.assertEqual(rec['causal_state'], expected['final_state'])
                        self.assertEqual(rec['evaluation_counts'], dict(available=1, unavailable=1))
                        self.assertEqual(rec['checkpoints'][-1]['evaluation_status'], 'unavailable')
                        result = json.loads((out/'RESULT.json').read_text())
                        self.assertEqual(result['status'], 'completed')
                        for name, digest in result['files'].items():
                            self.assertEqual(hashlib.sha256((out/name).read_bytes()).hexdigest(), digest)
                        with np.load(out/'arrays.npz', allow_pickle=False) as bank:
                            keys = (('checkpoint_params', 'final_params'),) if algo == 'dqn' else (
                                ('checkpoint_actor', 'final_actor'), ('checkpoint_critic', 'final_critic'))
                            for checkpoint_key, final_key in keys:
                                self.assertEqual(len(bank[checkpoint_key]), 2)
                                np.testing.assert_array_equal(bank[checkpoint_key][-1], bank[final_key])


if __name__ == '__main__':
    unittest.main()
