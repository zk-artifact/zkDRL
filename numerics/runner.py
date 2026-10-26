"""Single-job numerical harness. No challenge creation, proof, or run launcher."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
PYTHON = Path(__file__).resolve().parent.parent / 'python'
sys.path.insert(0, str(PYTHON))
import numpy as np
import dqn_reference as dr
import a2c_reference as ar
from random_tape import CounterRng, loads
from legacy import qat_model as q
from legacy.model import env_init, env_step
from legacy.a2c_fixed import fixed_math as fm, trainer, fixed_eval
from legacy.a2c_fixed.fixtures import real_params, reward_params
from legacy.a2c_fixed.float_bridge import FloatModel, Evaluator, float_update
from legacy.a2c_float.core import a2c as floats
try:
    from .operators import core as operators
except ImportError:
    from operators import core as operators

CONDITIONS = {
    'dqn': ('main_F', 'main_N', 'main_Q', 'residual', 'precision_S256'),
    'a2c': ('main_F', 'main_N', 'main_Q', 'precision_Q20', 'operator_O1', 'operator_O2'),
}


def plain(x):
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, np.generic):
        return plain(x.item())
    if isinstance(x, float) and not math.isfinite(x):
        # Preserve failed FP candidates in NPZ; JSON must not contain NaN tokens.
        return str(x)
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [plain(v) for v in x]
    return x


def encoded(x):
    return (json.dumps(plain(x), sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def validate(job, i):
    for key in ('id', 'algorithm', 'kind', 'seed', 'condition', 'episodes', 'checkpoints'):
        if key not in job:
            raise ValueError(f'missing job {key}')
    algo = job['algorithm']
    if algo not in CONDITIONS or job['condition'] not in CONDITIONS[algo]:
        raise ValueError('unregistered algorithm/condition')
    if not isinstance(job['id'], str) or not job['id']:
        raise ValueError('nonempty job id required')
    for key in ('kind', 'seed', 'episodes'):
        if type(job[key]) is not int:
            raise ValueError(f'job {key} must be integer')
    if job['kind'] not in (0, 1) or not 1 <= job['episodes'] <= 4000:
        raise ValueError('job kind/episodes')
    cp = job['checkpoints']
    if (not isinstance(cp, (list, tuple)) or not cp or cp[0] != 0
            or any(type(v) is not int or not 0 <= v <= job['episodes'] for v in cp)
            or list(cp) != sorted(set(cp))):
        raise ValueError('sorted unique checkpoints including zero required')
    (dr.validate if algo == 'dqn' else ar.validate)(i)
    if not any(i['challenge']):
        raise ValueError('issued nonzero challenge required')
    episodes = i['slots'] // 6 if algo == 'dqn' else i['episodes']
    if (job['seed'], job['kind'], job['episodes']) != (i['seed'], i['kind'], episodes):
        raise ValueError('job/input seed, kind, episodes mismatch')
    if algo == 'dqn':
        if i['scale'] != 16 or i['train'] != 1:
            raise ValueError('baseline S16 training input required')
        if i['params'] != ([6, 23, 11, 7] if i['kind'] else [0]*4):
            raise ValueError('registered S16 reward vector required')
    elif any(v % (1 << 24) for k in ('initial_actor', 'initial_critic') for v in i[k]):
        raise ValueError('A2C initial masters must lie on common Q16 grid')


@dataclass(frozen=True)
class DqnSpec:
    kind: str
    seed: int
    episodes: int
    mode: str
    scale: int

    def cfg(self):
        return q.ref.Config('full110', self.kind, self.seed,
                            None if self.mode == 'fp' else self.scale,
                            hidden=16, capacity=64, batch=8, warmup=32,
                            gamma=(1, 2), lr_div=16, eps_start=4,
                            eps_end=None, target_every=32, episodes=self.episodes)


@contextmanager
def operator_context(variant):
    """Patch every imported mass binding, including checkpoint evaluation."""
    if variant is None:
        yield
        return
    def masses(logits, bits, audit=None):
        if bits != 16:
            raise ValueError('operators require Q16')
        z = fm.ints(logits)
        fm.bound(z, 1024*(1 << bits), 'logits', audit)
        return np.array([operators.softmax_masses([int(v) for v in row], [1]*4, variant)
                         for row in z], dtype=object).reshape((-1, 4))
    modules = (fm, trainer, fixed_eval)
    originals = [m.masses for m in modules]
    try:
        for module in modules:
            module.masses = masses
        yield
    finally:
        for module, original in zip(modules, originals):
            module.masses = original


def guard_error(exc):
    # Legacy FP kernels encode only these explicit arithmetic guards as ValueError.
    return (isinstance(exc, (OverflowError, FloatingPointError)) or
            (type(exc) is ValueError and
             (str(exc).endswith('must be finite') or str(exc).endswith('must have abs < 2**31'))))


@contextmanager
def guard_context():
    """Observe failed fixed bounds without changing the legacy audit or decision."""
    modules = (fm, trainer)
    originals = [m.bound for m in modules]
    def wrap(original):
        def bound(a, limit, name, audit=None):
            try:
                return original(a, limit, name, audit)
            except OverflowError as exc:
                # Use Python ints, including for the signed-int64 minimum and
                # raw intermediates larger than int64. Never re-run a forward.
                try:
                    peak = max((abs(int(v)) for v in np.asarray(a, dtype=object).flat), default=0)
                    if peak >= limit:
                        exc.failed_guard = dict(name=name, max_abs=peak, strict_limit=int(limit))
                except (TypeError, ValueError, OverflowError):
                    # An overflow in conversion has no measured integer peak.
                    pass
                raise
        return bound
    try:
        for module, original in zip(modules, originals):
            module.bound = wrap(original)
        yield
    finally:
        for module, original in zip(modules, originals):
            module.bound = original


def failed_guard(exc, pending):
    detail = getattr(exc, 'failed_guard', None)
    return dict(detail, phase=(pending or {}).get('phase', 'unknown')) if detail is not None else None


def evaluate_checkpoint(record, ep, evaluate):
    """Evaluation availability is independent of training completion."""
    row = dict(episode=ep)
    try:
        row.update(evaluate())
    except Exception as exc:
        if not guard_error(exc):
            raise
        row.update(evaluation_status='unavailable', error=f'{type(exc).__name__}: {exc}',
                   failed_guard=failed_guard(exc, dict(phase='checkpoint')))
        record['evaluation_failures'].append(row.copy())
    else:
        row['evaluation_status'] = 'available'
    record['checkpoints'].append(row)


def abort(record, exc, pending):
    if not guard_error(exc):
        raise exc
    record.update(status='arithmetic_guard_abort', error=f'{type(exc).__name__}: {exc}',
                  failed_attempt=pending, failed_guard=failed_guard(exc, pending))


def _dqn(job, i, record, arrays):
    mode = {'main_F': 'fp', 'main_N': 'integer', 'main_Q': 'residual_high',
            'residual': 'residual_low', 'precision_S256': 'residual_high'}[job['condition']]
    scale = 256 if job['condition'] == 'precision_S256' else 16
    kind = 'recommendation' if i['kind'] else 'browser'
    spec = DqnSpec(kind, i['seed'], job['episodes'], mode, scale)
    initial = np.asarray(i['initial_coarse'], dtype=np.int64) / 16
    # Constructors expect an initializer. Supply the issued grid, never a new Gaussian.
    original = q.ref.initialization
    try:
        q.ref.initialization = lambda cfg: (initial.copy() if cfg.scale is None
                                            else (initial*cfg.scale).astype(np.int64))
        learner = q.create(spec, initial)
    finally:
        q.ref.initialization = original
    inner = learner.inner if mode == 'fp' else learner
    unit = 1 if mode == 'fp' else scale
    params = np.asarray(i['params'])/16 if mode == 'fp' else np.asarray(i['params'])*(scale//16)
    state, done = env_init(kind), False
    total = resets = padded = copies = wraps = cursor = 0
    traces, returns, updates, cp_masters = [], [], [], []
    pending = None
    def master():
        return inner.weights if mode == 'fp' else inner.master
    def snapshot():
        return dict(master=master().tolist(), target=inner.target.tolist(), buffer=inner.buffer.tolist(),
                    interactions=inner.interactions, updates=inner.updates, rng=cursor,
                    state=state.tolist(), done=bool(done), total=total, resets=resets,
                    padded=padded, target_copies=copies, replay_wraps=wraps)
    def checkpoint(ep):
        cp_masters.append(master().copy())
        evaluate_checkpoint(record, ep, lambda: q.evaluation(learner, spec, real_params(kind)))
    try:
        checkpoint(0)
        for ep in range(job['episodes']):
            state, done = env_init(kind), False
            resets += 1
            ep_return = 0.
            for t in range(6):
                slot = ep*6+t
                draws = arrays['raw_tape'][slot*10:(slot+1)*10].astype(np.uint64)
                cursor += 10
                pending = dict(episode=ep, slot=t, phase='forward', state=state.tolist(),
                               done=bool(done), draws=draws.tolist(), rng=cursor)
                values = learner.qvalues(state)
                action = 4 if done else int(draws[1] & 3) if (draws[0] & 15) < 4 else int(np.argmax(values))
                successor, reward, terminal = env_step(kind, state, action, params,
                                                       None if mode == 'fp' else scale, done)
                if np.any(successor < 0) or np.any(successor > 6):
                    raise OverflowError('state domain')
                indices = [-1]*8
                pending.update(phase='observe', action=action, reward=plain(reward), successor=successor.tolist())
                if not done:
                    n = min(inner.interactions+1, 64)
                    if inner.interactions >= 64:
                        wraps += 1
                    if inner.interactions+1 >= 32:
                        indices = [(int(x)*n) >> 32 for x in draws[2:]]
                    old_updates = inner.updates
                    before = master().copy()
                    learner.observe(state, action, reward, successor, terminal, draws[2:].astype(np.float64)/2**32)
                    if inner.updates != old_updates:
                        copies += int(inner.updates % 32 == 0)
                        updates.append([slot, inner.updates, float(np.max(np.abs(learner.real_master()))),
                                        float(np.max(np.abs(master().astype(np.float64)-before)))])
                else:
                    padded += 1
                traces.append([slot, ep, t, *state, int(done), action, reward, *successor,
                               int(terminal), *indices, inner.interactions, inner.updates, cursor])
                total += float(reward) if mode == 'fp' else int(reward)
                ep_return += float(reward)/unit
                state, done = successor, terminal
                pending = None
            returns.append(ep_return)
            record['completed_episodes'] = ep+1
            if ep+1 in job['checkpoints']:
                checkpoint(ep+1)
    except Exception as exc:
        abort(record, exc, pending)
    record.update(final_state=snapshot(), diagnostics=learner.diagnostics(),
                  effective_scale=scale, mode=mode, consumed_words=cursor)
    record['diagnostics']['max_abs_master_real_all_updates'] = max((row[2] for row in updates), default=0.)
    record['diagnostics']['max_abs_raw_update_all_updates'] = max((row[3] for row in updates), default=0.)
    if record['status'] == 'completed' and job['condition'] in ('main_Q', 'precision_S256'):
        record['causal_state'] = snapshot()
        record['coarse_export'] = learner.qweights().tolist()
        record['target_coarse_export'] = learner.qweights(True).tolist()
    arrays.update(final_params=master().copy(), final_target=inner.target.copy(),
                  physical_buffer=inner.buffer.copy(), initial_real=initial,
                  checkpoint_params=np.asarray(cp_masters), traces=np.asarray(traces, dtype=np.float64 if mode == 'fp' else np.int64).reshape((-1, 26)),
                  episode_returns=np.asarray(returns), update_diagnostics=np.asarray(updates).reshape((-1, 4)))
    record['update_diagnostic_columns'] = ['slot', 'updates', 'max_abs_master_real', 'max_abs_raw_update']


class GridModel(fm.FixedModel):
    """N: one atomic commit including both post-floor guards."""
    def commit(self, ga, gc, denominator, expected_hash, audit=None):
        if expected_hash != self.fingerprint():
            raise ValueError('stale/off-policy model snapshot')
        ac, cc = fm.candidates(self.actor_master, self.critic_master, ga, gc, denominator, audit)
        ratio = 1 << 24
        for v in (ac, cc):
            fm.bound(v//ratio, 1 << 24, 'forward_candidate_params', audit)
        ac, cc = ac//ratio*ratio, cc//ratio*ratio
        for v in (ac, cc):
            fm.bound(v, 256*fm.G, 'floored_master', audit)
            fm.bound(v//ratio, 1 << 24, 'floored_forward', audit)
        self.actor_master, self.critic_master = ac, cc
        self.update_count += 1


def a2c_model(i, condition):
    fixed = condition != 'main_F'
    cls = GridModel if condition == 'main_N' else fm.FixedModel if fixed else FloatModel
    model = cls.__new__(cls)
    model.update_count = 0
    if fixed:
        model.bits = 20 if condition == 'precision_Q20' else 16
        model.actor_master, model.critic_master = fm.ints(i['initial_actor']), fm.ints(i['initial_critic'])
    else:
        model.actor, model.critic = floats.MLP(16, 4), floats.MLP(16, 1)
        model.actor_params = np.asarray(i['initial_actor'], dtype=np.float64)/fm.G
        model.critic_params = np.asarray(i['initial_critic'], dtype=np.float64)/fm.G
        model.actor_lr, model.critic_lr = 1/32, 1/64
        model.entropy_coef, model.value_coef = 1/128, 1.
    return model


def _a2c(job, i, record, arrays):
    fixed = job['condition'] != 'main_F'
    model = a2c_model(i, job['condition'])
    kind = 'recommendation' if i['kind'] else 'browser'
    evaluator = (fixed_eval.FixedEvaluator if fixed else Evaluator)(kind, real_params(kind), 6, 7/8)
    unit = (1 << model.bits) if fixed else 1
    params = reward_params(kind, model.bits) if fixed else real_params(kind)
    traces, returns, diagnostics, am, cm = [], [], [], [], []
    audit = {}
    state, done = env_init(kind), False
    total = cursor = interactions = padded = successes = resets = 0
    pending = None
    metric_names = None
    def masters():
        return (model.actor_master, model.critic_master) if fixed else (model.actor_params, model.critic_params)
    def checkpoint(ep):
        a, c = masters()
        am.append(np.asarray(a, dtype=np.int64 if fixed else np.float64).copy())
        cm.append(np.asarray(c, dtype=np.int64 if fixed else np.float64).copy())
        evaluate_checkpoint(record, ep, lambda: evaluator.model(model))
    try:
        checkpoint(0)
        for ep in range(job['episodes']):
            state, done = env_init(kind), False
            resets += 1
            rows, ep_trace = [], []
            ep_total = 0
            if fixed:
                aq, cq = model.snapshot()
                fingerprint = model.fingerprint()
            for t in range(6):
                draw = int(arrays['raw_tape'][ep*6+t])
                cursor += 1
                pending = dict(episode=ep, slot=t, phase='collect', state=state.tolist(),
                               done=bool(done), draw=draw, rng=cursor, trace=ep_trace)
                before, was_done = state.copy(), done
                if done:
                    action = 4
                elif fixed:
                    z, _ = fm.forward(aq, state[None, :], 4, model.bits, audit)
                    fm.forward(cq, state[None, :], 1, model.bits, audit)
                    mass = fm.masses(z, model.bits, audit)[0]
                    action = fm.categorical(mass, draw)
                else:
                    z, _ = model.actor.forward(model.actor_params, state[None, :])
                    model.critic.forward(model.critic_params, state[None, :])
                    mass = floats.softmax_logsoftmax(z)[0][0]
                    action = floats.categorical(mass, draw/2**32)
                successor, reward, terminal = env_step(kind, state, action, params, unit if fixed else None, done)
                reward = int(reward) if fixed else float(reward)
                terminal = bool(terminal)
                if np.any(successor < 0) or np.any(successor > 6):
                    raise OverflowError('state domain')
                if fixed:
                    fm.bound([reward], 4*unit, 'reward', audit)
                if not was_done:
                    rows.append((before, action, reward, successor.copy(), int(terminal), int(not terminal and t < 5), mass.copy()))
                    interactions += 1
                    successes += int(i['kind'] == 0 and terminal and reward > 0)
                else:
                    padded += 1
                row = [ep, t, *before, action, reward, *successor, int(was_done), int(terminal), model.update_count, draw]
                ep_trace.append(row)
                traces.append(row)
                ep_total += reward
                total += reward
                state, done = successor, terminal
            dtype = object if fixed else np.float64
            rollout = {key: np.array([r[j] for r in rows], dtype=np.int64 if key == 'actions' else dtype)
                       for j, key in enumerate(('states', 'actions', 'rewards', 'successors', 'terminals', 'trace_continue', 'mass'))}
            rollout.update(planned_slots=6, policy_version=model.update_count)
            pending = dict(episode=ep, phase='update', trace=ep_trace, rollout=rollout)
            old_a, old_c = [v.copy() for v in masters()]
            if fixed:
                rollout.update(fingerprint=fingerprint, actor_snapshot=aq, critic_snapshot=cq)
                metrics = trainer.update_fixed(model, rollout, audit)
            else:
                metrics = float_update(model, rollout, 7/8, 1, False)
                metrics.update(zero_probability_cells=int(np.count_nonzero(rollout['mass'] == 0)))
            new_a, new_c = masters()
            metrics.update(actor_update_max_abs=float(np.max(np.abs(np.asarray(new_a-old_a, dtype=np.float64))))/(fm.G if fixed else 1),
                           critic_update_max_abs=float(np.max(np.abs(np.asarray(new_c-old_c, dtype=np.float64))))/(fm.G if fixed else 1))
            if metric_names is None:
                metric_names = sorted(metrics)
            diagnostics.append([metrics[k] for k in metric_names])
            returns.append(ep_total/unit)
            record['completed_episodes'] = ep+1
            pending = None
            if ep+1 in job['checkpoints']:
                checkpoint(ep+1)
    except Exception as exc:
        abort(record, exc, pending)
    a, c = masters()
    fs = dict(actor=a.tolist(), critic=c.tolist(), rng=cursor, state=state.tolist(), done=bool(done),
              total=total, episodes=record['completed_episodes'], updates=model.update_count,
              interactions=interactions, resets=resets, padded=padded, successes=successes)
    record.update(final_state=fs, consumed_words=cursor, numerical_maxima=audit,
                  update_diagnostic_columns=metric_names or [], forward_bits=model.bits if fixed else None)
    record['diagnostics'] = {k: max(row[j] for row in diagnostics) for j, k in enumerate(metric_names or [])}
    record['diagnostics']['zero_probability_cells_sum'] = sum(
        row[metric_names.index('zero_probability_cells')] for row in diagnostics) if diagnostics else 0
    if job['condition'] == 'main_Q' and record['status'] == 'completed':
        record.update(causal_state=fs, actor_export=(a >> 24).tolist(), critic_export=(c >> 24).tolist())
    arrays.update(final_actor=np.asarray(a, dtype=np.int64 if fixed else np.float64),
                  final_critic=np.asarray(c, dtype=np.int64 if fixed else np.float64),
                  initial_actor_real=np.asarray(i['initial_actor'], dtype=np.float64)/fm.G,
                  initial_critic_real=np.asarray(i['initial_critic'], dtype=np.float64)/fm.G,
                  checkpoint_actor=np.asarray(am), checkpoint_critic=np.asarray(cm),
                  traces=np.asarray(traces, dtype=np.int64 if fixed else np.float64).reshape((-1, 16)),
                  episode_returns=np.asarray(returns),
                  update_diagnostics=np.asarray(diagnostics, dtype=np.float64).reshape((len(diagnostics), len(metric_names or []))))


def run(job, base_input):
    """Return JSON-compatible record and non-object NPZ arrays; no output writes.

    One process per job in production. Guard/operator/initializer bindings are restored
    for serial API callers, but calls in concurrent threads are unsupported.
    """
    validate(job, base_input)
    job, i = plain(job), plain(base_input)
    algo = 1 if job['algorithm'] == 'dqn' else 2
    count = job['episodes']*6*(10 if algo == 1 else 1)
    rng = CounterRng(algo, i['rng_seed'], i['challenge'])
    tape = np.fromiter((rng.next_u32() for _ in range(count)), dtype='<u4', count=count)
    initial = (i['initial_coarse'] if algo == 1 else
               [v//(1 << 24) for k in ('initial_actor', 'initial_critic') for v in i[k]])
    record = dict(schema='full110-numerics-v1', job=job, status='completed', error=None,
                  cryptographic_proof=False, guest_execution=False, completed_episodes=0,
                  checkpoints=[], evaluation_failures=[], baseline_input_sha256=sha(encoded(i)),
                  initial_grid_sha256=sha(np.asarray(initial, dtype='<i8').tobytes()),
                  initial_grid_denominator=16 if algo == 1 else 1 << 16,
                  initial_grid_encoding='concatenated signed little-endian i64; actor then critic for A2C',
                  tape_sha256=sha(tape.tobytes()), planned_words=count,
                  rng_algorithm=algo, rng_seed=i['rng_seed'], challenge=i['challenge'],
                  rng_key_sha256=sha(rng.key), numpy_version=np.__version__,
                  failed_attempt=None, failed_guard=None)
    record['source_sha256'] = {
        'runner.py': sha(Path(__file__).read_bytes()),
        **{'python/'+str(p.relative_to(PYTHON)): sha(p.read_bytes())
           for p in sorted(PYTHON.rglob('*')) if p.suffix in ('.py', '.json')},
    }
    arrays = dict(raw_tape=tape, initial_grid=np.asarray(initial, dtype='<i8'))
    variant = job['condition'].removeprefix('operator_') if job['condition'].startswith('operator_') else None
    if variant:
        record['operator'] = dict(variant=variant, table_sha256=operators.table_hashes()[variant],
                                  contract_sha256=operators.contract_hash())
    try:
        with guard_context(), operator_context(variant):
            (_dqn if algo == 1 else _a2c)(job, i, record, arrays)
    except Exception as exc:
        abort(record, exc, dict(phase='initialization'))
        # A constructor guard can fire before any slot or checkpoint exists.
        record['consumed_words'] = 0
        arrays['traces'] = np.empty((0, 26 if algo == 1 else 16), dtype='<i8')
    record['evaluation_counts'] = {
        status: sum(row['evaluation_status'] == status for row in record['checkpoints'])
        for status in ('available', 'unavailable')}
    record['evaluation_failure_count'] = len(record['evaluation_failures'])
    record['trajectory_sha256'] = sha(arrays['traces'].astype('<f8' if arrays['traces'].dtype.kind == 'f' else '<i8').tobytes())
    record['arrays'] = {k: dict(shape=list(v.shape), dtype=v.dtype.str,
                              sha256=sha(v.tobytes(order='C'))) for k, v in arrays.items()}
    return plain(record), arrays


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    source = args.input.read_bytes()
    record, arrays = run(loads(args.job.read_text()), loads(source))
    record['baseline_source_file_sha256'] = sha(source)
    record['job_source_file_sha256'] = sha(args.job.read_bytes())
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise FileExistsError('output directory must be empty')
    with (args.out/'arrays.npz').open('xb') as f:
        np.savez_compressed(f, **arrays)
    (args.out/'record.json').write_bytes(encoded(record))
    result = dict(schema='full110-result-v1', job_id=record['job']['id'], status=record['status'],
                  cryptographic_proof=False, guest_execution=False,
                  files={name: sha((args.out/name).read_bytes()) for name in ('record.json', 'arrays.npz')})
    (args.out/'RESULT.json').write_bytes(encoded(result))
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
