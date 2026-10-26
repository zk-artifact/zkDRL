"""Independent NumPy/reference semantics for a bounded DQN variant.

Integer arithmetic uses floor rounding at the documented operation boundaries.
The proof circuit is a separate implementation; Python checking is NOT a proof.
"""
from dataclasses import dataclass
import numpy as np

N = HIDDEN = ACTIONS = CAP = 4
P = 40
ENTRY = 11
LIMIT = 2**31


def checked(x):
    a = np.asarray(x)
    if not np.isfinite(a).all() or np.any(a < -LIMIT) or np.any(a >= LIMIT):
        raise OverflowError('signed-32 numerical profile violated')
    return a


def initial_weights(seed, scale):
    rng = np.random.default_rng(seed)
    # Same real initialization for every precision; positive hidden biases reduce dead units.
    w = np.concatenate([rng.normal(0, .20, 16), np.full(4, .25),
                        rng.normal(0, .20, 16), np.zeros(4)])
    return w if scale is None else np.floor(w * scale).astype(np.int64)


def forward(weights, state, scale):
    checked(weights)
    if scale is not None and (np.any(weights < -2**20) or np.any(weights >= 2**20)):
        raise OverflowError('signed-21 weight profile violated')
    if np.any(np.asarray(state) < 0) or np.any(np.asarray(state) >= 2**16):
        raise OverflowError('unsigned-16 state profile violated')
    w1, b1 = weights[:16].reshape(4, 4), weights[16:20]
    w2, b2 = weights[20:36].reshape(4, 4), weights[36:40]
    raw = w1 @ state + b1
    checked(raw)
    h = np.maximum(raw, 0)
    q = w2 @ h
    if scale is not None:
        q = q // scale
        checked(q)
    q = q + b2
    checked(h); checked(q)
    return q, h


def env_init(kind):
    return np.array([0, 0, 0, 1] if kind == 'browser' else [0, 0, 0, 0], dtype=np.int64)


def env_step(kind, state, action, params, scale, done=False):
    s = np.array(state, copy=True)
    if done:
        return s, 0, True
    if action not in range(4):
        raise ValueError('active action outside catalogue')
    unit = 1.0 if scale is None else scale
    if kind == 'browser':
        if action == 0:
            s[0] = 1
        elif action == 1:
            s[0] = 0
        elif action == 2:
            s[1] = 1
        else:
            s[2] = 1
            return s, unit if s[0] and s[1] else -unit, True
        return s, -unit / 16 if scale is None else -scale // 16, False
    if kind != 'recommendation':
        raise ValueError(kind)
    reward = params[action] - (unit / 4) * s[action]
    if scale is not None:
        reward = int(params[action]) - (scale // 4) * int(s[action])
    s[action] += 1
    return s, reward, False


@dataclass
class Learner:
    weights: np.ndarray
    target: np.ndarray
    buffer: np.ndarray
    interactions: int = 0
    updates: int = 0

    @classmethod
    def create(cls, seed=0, scale=16):
        w = initial_weights(seed, scale)
        return cls(w, w.copy(), np.zeros((CAP, ENTRY), dtype=w.dtype))

    def step(self, state, action, reward, successor, terminal, replay_index, scale):
        entry = np.concatenate([state, [action, reward], successor, [int(terminal)]])
        self.buffer[self.interactions % CAP] = entry
        self.interactions += 1
        if self.interactions < CAP:
            return
        row = self.buffer[replay_index]
        s, a, r, ns, d = row[:4], int(row[4]), row[5], row[6:10], bool(row[10])
        q_next, _ = forward(self.target, ns, scale)
        y = r if d else r + (np.max(q_next) * .5 if scale is None else int(np.max(q_next)) // 2)
        q, h = forward(self.weights, s, scale)
        error = q[a] - y
        checked([error])
        outgoing = self.weights[20:36].reshape(4, 4)[a]
        if scale is None:
            dh = error * outgoing * (h > 0)
            grad1 = np.outer(dh, s)
            grad2 = error * h
            delta1, deltab1 = grad1 / 16, dh / 16
            delta2, deltab2 = grad2 / 16, error / 16
        else:
            dh_raw = error * outgoing // scale
            checked(dh_raw)
            dh = dh_raw * (h > 0)
            checked(dh)
            delta1, deltab1 = np.outer(dh, s) // 16, dh // 16
            delta2, deltab2 = (error * h // scale) // 16, error // 16
        updated = self.weights.copy()
        updated[:16] -= delta1.reshape(-1)
        updated[16:20] -= deltab1
        updated[20 + a*4:24 + a*4] -= delta2
        updated[36+a] -= deltab2
        checked(updated)
        if scale is not None and (np.any(updated < -2**20) or np.any(updated >= 2**20)):
            raise OverflowError('signed-21 updated weight profile violated')
        self.weights = updated
        self.updates += 1
        if self.updates % 4 == 0:
            self.target = updated.copy()


def run(kind, params, seed=0, scale=16, slots=8, horizon=4, train=True,
        draws=None, learner=None, capture=True):
    learner = learner or Learner.create(seed, scale)
    if draws is None:
        draws = np.random.default_rng(10000 + seed).integers(0, [16, 4, 4], size=(slots, 3))
    draws = np.asarray(draws, dtype=np.int64)
    states, successors, actions, rewards, dones, starts, samples = [], [], [], [], [], [], []
    models, targets, buffers = [learner.weights.copy()], [learner.target.copy()], [learner.buffer.copy()]
    done, state, total = False, env_init(kind), 0
    actual_actions = []
    for t in range(slots):
        if t % horizon == 0:
            done, state = False, env_init(kind)
        q, _ = forward(learner.weights, state, scale)
        action = 4 if done else int(draws[t, 1]) if train and draws[t, 0] < 4 else int(np.argmax(q))
        successor, reward, terminal = env_step(kind, state, action, params, scale, done)
        if capture:
            states.append(state.copy()); successors.append(successor.copy())
            starts.append(int(done)); dones.append(int(terminal)); actions.append(action); rewards.append(reward)
            samples.append(int(draws[t, 2]))
        actual_actions.append(action)
        if not done and train:
            learner.step(state, action, reward, successor, terminal, int(draws[t, 2]), scale)
        total += reward
        state, done = successor, terminal
        if capture:
            models.append(learner.weights.copy()); targets.append(learner.target.copy()); buffers.append(learner.buffer.copy())
    trace = dict(states=states, nextStates=successors, actions=actions, rewards=rewards,
                 dones=dones, startDone=starts, samples=samples, models=models, targets=targets, buffers=buffers)
    if capture:
        trace = {k: np.asarray(v).tolist() for k, v in trace.items()}
    return {'trace': trace, 'return': float(total) if scale is None else int(total),
            'actions': actual_actions, 'learner': learner}
