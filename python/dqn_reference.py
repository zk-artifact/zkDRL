"""Schema-202 commit-then-challenge reference using the original learner/MDP.

Only input validation, scheduling, serialization, and commitments live here.
Learning is qat_model.MixedLearner(residual_high/fixed_replay_fast); transitions
are model.env_step. Fixtures use public reproducible salts and public data,
not confidential inputs or a claim about initialization provenance or fairness.

Add this python directory to sys.path, then import dqn_reference. Public API:
fixture(kind=1, seed=30, episodes=4000, *, scale=16, train=1, slots=None,
        hidden=16, capacity=64, random_seed=0)
run(i, capture=True, *, hidden=16, capacity=64)
config_bytes(i, *, hidden=16, capacity=64), journal_bytes(j)
load_input(path, *, hidden=16, capacity=64) rejects duplicate JSON keys.
Profile metadata is passed explicitly and never added to PrivateInput.
The auditor must replace fixture challenge zeros AFTER config registration.
"""
from pathlib import Path
from dataclasses import dataclass
import hashlib
import struct

import numpy as np
from legacy import qat_model as q
from legacy.model import env_init, env_step
from random_tape import CounterRng, public_rng_seed, loads, u64s

H, P, CAP, B, L = 16, 148, 64, 8, 16
SCHEMA, MAX_SLOTS, STATE_BYTES = 202, 24000, 8076


def profile(hidden=16, capacity=64):
    if type(hidden) is not int or hidden not in (16, 32, 64):
        raise ValueError('hidden must be 16, 32, or 64')
    if type(capacity) is not int or capacity not in (64, 128, 256):
        raise ValueError('capacity must be 64, 128, or 256')
    return hidden, 9 * hidden + 4, capacity

INPUT_FIELDS = frozenset((
    'schema', 'kind', 'scale', 'slots', 'train', 'seed', 'rng_seed', 'initial_coarse',
    'params', 'config_salt', 'final_salt', 'challenge',
))


def sha(b):
    return hashlib.sha256(b).digest()


def u32s(values):
    return b''.join(struct.pack('<I', int(x)) for x in values)


def i64s(values):
    return b''.join(struct.pack('<q', int(v)) for v in np.asarray(values, dtype=object).flat)


def validate(i, *, hidden=16, capacity=64):
    """Match serde's integer types, exact fields, array lengths, and run bounds."""
    _, P, _ = profile(hidden, capacity)
    if type(i) is not dict or set(i) != INPUT_FIELDS:
        raise ValueError('PrivateInput requires exactly the schema-202 fields')
    for key in ('schema', 'kind', 'scale', 'slots', 'train', 'seed'):
        if type(i[key]) is not int or not 0 <= i[key] < 2**32:
            raise ValueError(f'{key} must be a u32')
    if i['schema'] != SCHEMA or i['kind'] not in (0, 1) or i['train'] not in (0, 1):
        raise ValueError('unsupported schema/kind/train')
    if i['scale'] not in (16, 256) or not 6 <= i['slots'] <= MAX_SLOTS or i['slots'] % 6:
        raise ValueError('unsupported scale/slots')
    for key, n, low, high in (
        ('initial_coarse', P, -2**20, 2**20 - 1),
        ('params', 4, -16 * i['scale'], 16 * i['scale']),
        ('config_salt', 32, 0, 255), ('final_salt', 32, 0, 255),
        ('challenge', 32, 0, 255), ('rng_seed', 32, 0, 255),
    ):
        a = i[key]
        if (type(a) is not list or len(a) != n
                or any(type(x) is not int or not low <= x <= high for x in a)):
            raise ValueError(f'invalid {key}')
    if i['kind'] == 0 and i['params'] != [0] * 4:
        raise ValueError('login parameters must be zero')


def config_bytes(i, *, hidden=16, capacity=64):
    H, _, CAP = profile(hidden, capacity)
    validate(i, hidden=hidden, capacity=capacity)
    return (b'zkDRL/ctc/dqn/config/v3\0'
            + u32s([i['schema'], i['kind'], i['scale'], i['slots'], i['train'],
                    H, CAP, B, 32, L, 32, 6, 4, 1, 2, i['seed']])
            + bytes(i['rng_seed']) + i64s(i['initial_coarse']) + i64s(i['params'])
            + bytes(i['config_salt']) + bytes(i['final_salt']))


def state_bytes(s, *, hidden=16, capacity=64):
    _, P, CAP = profile(hidden, capacity)
    STATE_BYTES = 16*P + 88*CAP + 76
    if len(s['state']) != 4 or len(s['master']) != P or len(s['target']) != P or len(s['buffer']) != CAP or any(len(r) != 11 for r in s['buffer']):
        raise ValueError('noncanonical state dimensions')
    b = (i64s(s['master']) + i64s(s['target']) + i64s(s['buffer'])
         + u32s([s['interactions'], s['updates']]) + u64s([s['rng']]) + i64s(s['state'])
         + u32s([s['done']]) + i64s([s['total']])
         + u32s([s['resets'], s['padded'], s['target_copies'], s['replay_wraps']]))
    if len(b) != STATE_BYTES:
        raise ValueError('noncanonical state dimensions')
    return b


@dataclass(frozen=True)
class LongSpec:
    """Exact ExtensionSpec fixed_replay_fast configuration, without runner imports.

    Importing run_reference_extension would also load unrelated study runners.
    This supplies only its configuration interface to the EXISTING MixedLearner.
    It implements no forward, backward, or update arithmetic.
    """
    kind: str
    seed: int
    scale: int
    episodes: int
    hidden: int = 16
    capacity: int = 64
    name: str = 'ctc-reference'
    profile: str = 'fixed_replay_fast'
    mode: str = 'residual_high'
    checkpoints: tuple = ()

    def cfg(self):
        return q.ref.Config(
            self.name, self.kind, self.seed, self.scale, episodes=self.episodes,
            hidden=self.hidden, capacity=self.capacity, batch=B, warmup=32, gamma=(1, 2),
            lr_div=L, eps_start=4, eps_end=None, target_every=32,
        )


def spec(i, hidden=16, capacity=64):
    return LongSpec('browser' if i['kind'] == 0 else 'recommendation',
                    i['seed'], i['scale'], i['slots'] // 6, hidden, capacity)


def fixture(kind=1, seed=30, episodes=4000, *, scale=16, train=1,
            slots=None, hidden=16, capacity=64, random_seed=0):
    """PUBLIC reproducible fixture; challenge zeros are an auditor placeholder."""
    profile(hidden, capacity)
    if type(episodes) is not int or not 1 <= episodes <= 4000:
        raise ValueError('episodes must be 1..4000')
    slots = episodes * 6 if slots is None else slots
    initial = q.ref.initialization(spec(dict(kind=kind, scale=scale, slots=slots, seed=seed), hidden, capacity).cfg())
    params = [v * (scale // 16) for v in (6,23,11,7)] if kind == 1 else [0]*4
    label = f'PUBLIC-CTC202-{kind}-{scale}-{slots}-{train}-{seed}-{hidden}-{capacity}'.encode()
    i = dict(schema=SCHEMA, kind=kind, scale=scale, slots=slots, train=train, seed=seed,
             rng_seed=public_rng_seed(1, random_seed), initial_coarse=initial.tolist(), params=params,
             config_salt=list(sha(label+b'/config')), final_salt=list(sha(label+b'/final')),
             challenge=[0]*32)
    validate(i, hidden=hidden, capacity=capacity)
    return i


def run(i, capture=True, *, hidden=16, capacity=64):
    _, _, CAP = profile(hidden, capacity)
    validate(i, hidden=hidden, capacity=capacity)
    if type(capture) is not bool:
        raise ValueError('capture must be bool')
    learner = q.MixedLearner(spec(i, hidden, capacity))
    learner.master = np.array(i['initial_coarse'], dtype=np.int64) * learner.factor
    learner.target = learner.master.copy()
    config = sha(config_bytes(i, hidden=hidden, capacity=capacity))
    tape = CounterRng(1, i['rng_seed'], i['challenge'])
    rng = 0
    kind = 'browser' if i['kind'] == 0 else 'recommendation'
    params = np.array(i['params'], dtype=np.int64)
    state, done = env_init(kind), False
    total = resets = padded = copies = wraps = 0
    state_chain = trajectory_chain = bytes(32)
    audit = []

    def snapshot(successor, terminal):
        return dict(master=learner.master.tolist(), target=learner.target.tolist(),
                    buffer=learner.buffer.tolist(), interactions=learner.interactions,
                    updates=learner.updates, rng=rng, state=successor.tolist(),
                    done=bool(terminal), total=total, resets=resets, padded=padded,
                    target_copies=copies, replay_wraps=wraps)

    for slot in range(i['slots']):
        if slot % 6 == 0:
            state, done = env_init(kind), False
            resets += 1
        draws = [tape.next_u32() for _ in range(10)]
        rng = tape.position()
        # Preserve forward diagnostics even on exploration and padding slots.
        values = learner.qvalues(state)
        action = (4 if done else (draws[1] & 3)
                  if i['train'] and (draws[0] & 15) < 4 else int(np.argmax(values)))
        successor, reward, terminal = env_step(kind, state, action, params, i['scale'], done)
        if np.any(successor < 0) or np.any(successor > 6):
            raise OverflowError('state domain')
        indices = [-1] * B
        if not done and i['train']:
            n = min(learner.interactions + 1, CAP)
            if learner.interactions >= CAP:
                wraps += 1
            if learner.interactions + 1 >= 32:
                indices = [(x * n) >> 32 for x in draws[2:]]
            old_updates = learner.updates
            # The dyadic quotient and multiplication by n<=256 are exact in FP64.
            # Existing learner floor(quantile*n) therefore equals (draw*n)>>32.
            learner.observe(state, action, reward, successor, terminal,
                            np.array(draws[2:], dtype=np.float64) / 2**32)
            if learner.updates != old_updates and learner.updates % 32 == 0:
                copies += 1
        if done:
            padded += 1
        total += int(reward)
        if capture:
            fs = snapshot(successor, terminal)
            trace = [slot, slot // 6, slot % 6, *state, int(done), action,
                     int(reward), *successor, int(terminal), *indices,
                     learner.interactions, learner.updates]
            trajectory_chain = sha(b'zkRL/risc0/trajectory/v1\0' + trajectory_chain + i64s(trace) + u64s([rng]))
            state_chain = sha(b'zkRL/risc0/state-chain/v1\0' + state_chain + state_bytes(fs, hidden=hidden, capacity=capacity))
            audit.append(dict(slot=slot, action=action, reward=int(reward), start_done=bool(done),
                              terminal=bool(terminal), state=state.tolist(), successor=successor.tolist(),
                              replay_indices=indices, interactions=learner.interactions,
                              updates=learner.updates, rng=rng, state_chain=list(state_chain),
                              trajectory_chain=list(trajectory_chain)))
        state, done = successor, terminal
    fs = snapshot(state, done)
    export = learner.qweights().tolist()
    final = sha(b'zkDRL/ctc/dqn/final/v3\0' + config + bytes(i['challenge'])
                + bytes(i['final_salt']) + state_bytes(fs, hidden=hidden, capacity=capacity) + i64s(export)
                + i64s(learner.qweights(True)))
    journal = dict(schema=SCHEMA, config_commit=list(config), final_commit=list(final),
                   challenge=i['challenge'], kind=i['kind'], scale=i['scale'], slots=i['slots'],
                   train=i['train'], return_units=total, interactions=learner.interactions,
                   updates=learner.updates, resets=resets, padded=padded, target_copies=copies,
                   replay_wraps=wraps)
    return dict(journal=journal, final_state=fs, coarse_export=export,
                state_chain=list(state_chain) if capture else None,
                trajectory_chain=list(trajectory_chain) if capture else None, audit=audit)


def loads_input(data, *, hidden=16, capacity=64):
    i = loads(data)
    validate(i, hidden=hidden, capacity=capacity)
    return i


def load_input(path, *, hidden=16, capacity=64):
    return loads_input(Path(path).read_text(), hidden=hidden, capacity=capacity)


def journal_bytes(j):
    raw = u32s([j['schema']]) + b''.join(bytes(j[k]) for k in ('config_commit','final_commit','challenge'))
    raw += u32s([j[k] for k in ('kind','scale','slots','train')]) + i64s([j['return_units']])
    raw += u32s([j[k] for k in ('interactions','updates','resets','padded','target_copies','replay_wraps')])
    if len(raw) != 148:
        raise ValueError('journal length')
    return raw
