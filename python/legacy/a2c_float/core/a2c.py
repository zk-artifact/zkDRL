"""NumPy float64 A2C mathematics; no rollout collection or training harness.

``ActorCritic(seed).update(states, actions, advantages, returns, denominator)``
performs one simultaneous SGD step. The caller collects on-policy data, computes
detached advantages/returns, and supplies a fixed planned-slot denominator
(e.g. ``6 * rollout_episodes``). Only active rows are passed, with no advantage
normalization. ``policy_value`` returns probabilities, log probabilities, values.
Parameters are public flat arrays, ``actor_params`` and ``critic_params``.
"""

from dataclasses import dataclass

import numpy as np


LIMIT = float(2**31)


def _array(value, name, *, bounded=True):
    raw = np.asarray(value)
    if raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be real numeric data")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    if bounded and np.any(np.abs(result) >= LIMIT):
        raise ValueError(f"{name} must have abs < 2**31")
    return result


def _scalar(value, name, *, minimum=None, maximum=None):
    result = _array(value, name, bounded=False)
    if result.ndim != 0:
        raise ValueError(f"{name} must be scalar")
    result = float(result)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _vector(value, name, n):
    result = _array(value, name)
    if result.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},)")
    return result


@dataclass(frozen=True)
class MLPCache:
    states: np.ndarray
    preactivation: np.ndarray
    hidden: np.ndarray
    w2: np.ndarray


class MLP:
    """4 -> hidden ReLU -> out; flat layout W1, b1, W2, b2 (row major)."""

    def __init__(self, hidden=16, out=4):
        self.hidden = _positive_int(hidden, "hidden")
        self.out = _positive_int(out, "out")
        self.size = self.hidden * 4 + self.hidden + self.out * self.hidden + self.out

    def unpack(self, params):
        p = _vector(params, "params", self.size)
        a, b, c = 4 * self.hidden, 5 * self.hidden, (5 + self.out) * self.hidden
        return p[:a].reshape(self.hidden, 4), p[a:b], p[b:c].reshape(self.out, self.hidden), p[c:]

    def initialize(self, rng, w2_std):
        return _array(np.concatenate((
            rng.normal(0.0, 0.20, (self.hidden, 4)).ravel(),
            np.full(self.hidden, 0.25),
            rng.normal(0.0, w2_std, (self.out, self.hidden)).ravel(),
            np.zeros(self.out),
        )), "initial params")

    def forward(self, params, states):
        """Return (N x out outputs, cache); cache snapshots backward inputs."""
        x = _array(states, "states")
        if x.ndim != 2 or x.shape[1] != 4:
            raise ValueError("states must have shape (N, 4)")
        w1, b1, w2, b2 = self.unpack(params)
        with np.errstate(over="ignore", invalid="ignore"):
            pre = _array(x @ w1.T + b1, "hidden preactivation")
            hidden = np.maximum(pre, 0.0)
            outputs = _array(hidden @ w2.T + b2, "logits/values")
        return outputs, MLPCache(x.copy(), pre, hidden, w2.copy())

    def backward(self, cache, upstream):
        """Exact parameter VJP; ReLU derivative is zero at zero."""
        dy = _array(upstream, "upstream gradient")
        if dy.shape != (cache.states.shape[0], self.out):
            raise ValueError("upstream must have shape (N, out)")
        with np.errstate(over="ignore", invalid="ignore"):
            dh = (dy @ cache.w2) * (cache.preactivation > 0.0)
            grad = np.concatenate((
                (dh.T @ cache.states).ravel(), dh.sum(axis=0),
                (dy.T @ cache.hidden).ravel(), dy.sum(axis=0),
            ))
        return _vector(grad, "parameter gradient", self.size)


def softmax_logsoftmax(logits):
    """Stable float64 row probabilities/log probabilities for N x 4 logits.

    Underflow to zero probabilities is allowed; log probabilities are computed
    from shifted logits, never by taking log of the rounded probabilities.
    """
    z = _array(logits, "logits")
    if z.ndim != 2 or z.shape[1] != 4:
        raise ValueError("logits must have shape (N, 4)")
    shifted = z - np.max(z, axis=1, keepdims=True)
    with np.errstate(under="ignore"):
        exp = np.exp(shifted)
    normalizer = exp.sum(axis=1, keepdims=True)
    p = _array(exp / normalizer, "probabilities", bounded=False)
    logp = _array(shifted - np.log(normalizer), "log probabilities", bounded=False)
    return p, logp


def categorical(probs4, u):
    """Inverse CDF sampling; equality goes right, skipping zero-mass bins.

    Accept sum error of at most eight float64 epsilons and set the last CDF
    entry to one solely to correct floating-point rounding. No renormalization.
    """
    p = _vector(probs4, "probabilities", 4)
    if np.any(p < 0.0) or np.any(p > 1.0) or abs(float(p.sum()) - 1.0) > 8 * np.finfo(np.float64).eps:
        raise ValueError("probabilities must be in [0, 1] and sum to one")
    u = _scalar(u, "u", minimum=0.0)
    if u >= 1.0:
        raise ValueError("u must be < 1")
    cdf = np.cumsum(p)
    cdf[-1] = 1.0
    return int(np.searchsorted(cdf, u, side="right"))


def gae(rewards, values, next_values, terminals, trace_continue, gamma, lam):
    """Return new advantage and detached target arrays.

    next_values are values of the actual successor before any reset. Terminal
    rows suppress bootstrap; nonterminal time limits retain bootstrap. The
    caller sets trace_continue=0 at every reset and rollout edge. Lambda one
    gives n-step A2C advantages. Inputs are never differentiated or mutated.
    """
    rewards = _array(rewards, "rewards")
    if rewards.ndim != 1:
        raise ValueError("rewards must be a vector")
    n = len(rewards)
    values = _vector(values, "values", n)
    next_values = _vector(next_values, "next_values", n)
    terminals = _vector(terminals, "terminals", n)
    trace_continue = _vector(trace_continue, "trace_continue", n)
    if np.any((terminals != 0) & (terminals != 1)) or np.any((trace_continue != 0) & (trace_continue != 1)):
        raise ValueError("terminal and trace masks must be binary")
    if np.any((terminals == 1) & (trace_continue != 0)) or (n and trace_continue[-1] != 0):
        raise ValueError("terminals and final row must cut the trace")
    gamma = _scalar(gamma, "gamma", minimum=0.0, maximum=1.0)
    lam = _scalar(lam, "lam", minimum=0.0, maximum=1.0)
    with np.errstate(over="ignore", invalid="ignore"):
        delta = _array(rewards + gamma * (1.0 - terminals) * next_values - values, "delta")
        advantage = np.empty(n, dtype=np.float64)
        following = 0.0
        for i in range(n - 1, -1, -1):
            following = float(_array(delta[i] + gamma * lam * trace_continue[i] * following, "advantage"))
            advantage[i] = following
        targets = _array(values + advantage, "targets")
    return advantage, targets


@dataclass(frozen=True)
class LossGradients:
    actor_gradient: np.ndarray
    critic_gradient: np.ndarray
    diagnostics: dict


class ActorCritic:
    """Separate actor/critic MLPs and simultaneous plain SGD.

    Defaults use power-of-two learning rates. Rates and coefficients must be
    finite and nonnegative. All losses/entropy use the fixed denominator;
    gradient norms describe old-parameter gradients. update() reports the
    committed max_abs_params and update_count. Empty active batches are valid.
    Invalid data or candidates raise ValueError without committing a step.
    """

    def __init__(self, seed=0, *, hidden=16, actor_lr=2**-10,
                 critic_lr=2**-10, entropy_coef=0.0, value_coef=1.0):
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.actor = MLP(hidden, 4)
        self.critic = MLP(hidden, 1)
        self.actor_params = self.actor.initialize(np.random.default_rng(np.random.SeedSequence([1101, int(seed)])), 0.01)
        self.critic_params = self.critic.initialize(np.random.default_rng(np.random.SeedSequence([1102, int(seed)])), 0.20)
        self.actor_lr = _scalar(actor_lr, "actor_lr", minimum=0.0)
        self.critic_lr = _scalar(critic_lr, "critic_lr", minimum=0.0)
        self.entropy_coef = _scalar(entropy_coef, "entropy_coef", minimum=0.0)
        self.value_coef = _scalar(value_coef, "value_coef", minimum=0.0)
        self.update_count = 0

    def policy_value(self, states):
        logits, _ = self.actor.forward(self.actor_params, states)
        values, _ = self.critic.forward(self.critic_params, states)
        p, logp = softmax_logsoftmax(logits)
        return p, logp, values[:, 0]

    def losses_and_gradients(self, states, actions, advantages, returns, denominator):
        logits, actor_cache = self.actor.forward(self.actor_params, states)
        values, critic_cache = self.critic.forward(self.critic_params, states)
        n = logits.shape[0]
        actions = np.asarray(actions)
        if actions.shape != (n,) or actions.dtype.kind not in "iu" or np.any(actions < 0) or np.any(actions >= 4):
            raise ValueError("actions must be an integer vector with entries in [0, 4)")
        advantages = _vector(advantages, "advantages", n)
        returns = _vector(returns, "returns", n)
        d = _scalar(denominator, "denominator", minimum=float(n))
        if d <= 0.0:
            raise ValueError("denominator must be positive and >= active rows")
        entropy_coef = _scalar(self.entropy_coef, "entropy_coef", minimum=0.0)
        value_coef = _scalar(self.value_coef, "value_coef", minimum=0.0)
        p, logp = softmax_logsoftmax(logits)
        with np.errstate(over="ignore", invalid="ignore"):
            residual = values[:, 0] - returns
            policy_loss = -np.sum(advantages * logp[np.arange(n), actions]) / d
            entropy = -np.sum(p * logp) / d
            critic_loss = value_coef * np.sum(residual**2) / (2.0 * d)
            onehot = np.eye(4, dtype=np.float64)[actions]
            # Center logits before the expectation to retain shift invariance.
            z = logits - np.max(logits, axis=1, keepdims=True)
            dz = (advantages[:, None] * (p - onehot)
                  + entropy_coef * p * (z - np.sum(p * z, axis=1, keepdims=True))) / d
            dv = value_coef * residual[:, None] / d
            actor_gradient = self.actor.backward(actor_cache, dz)
            critic_gradient = self.critic.backward(critic_cache, dv)
            diagnostics = {
                "policy_loss": policy_loss,
                "actor_total": policy_loss - entropy_coef * entropy,
                "critic_loss": critic_loss,
                "entropy_fixed_denominator": entropy,
                "actor_gradient_norm": np.linalg.norm(actor_gradient),
                "critic_gradient_norm": np.linalg.norm(critic_gradient),
                "max_abs_params": max(np.max(np.abs(self.actor_params)), np.max(np.abs(self.critic_params))),
            }
        diagnostics = {key: _scalar(value, key) for key, value in diagnostics.items()}
        return LossGradients(actor_gradient, critic_gradient, diagnostics)

    def update(self, states, actions, advantages, returns, denominator):
        result = self.losses_and_gradients(states, actions, advantages, returns, denominator)
        actor_lr = _scalar(self.actor_lr, "actor_lr", minimum=0.0)
        critic_lr = _scalar(self.critic_lr, "critic_lr", minimum=0.0)
        with np.errstate(over="ignore", invalid="ignore"):
            actor_candidate = _array(self.actor_params - actor_lr * result.actor_gradient, "actor candidate params")
            critic_candidate = _array(self.critic_params - critic_lr * result.critic_gradient, "critic candidate params")
        # Both gradients and both candidates are validated before either commit.
        self.actor_params, self.critic_params = actor_candidate, critic_candidate
        self.update_count += 1
        diagnostics = dict(result.diagnostics)
        diagnostics["max_abs_params"] = float(max(np.max(np.abs(actor_candidate)), np.max(np.abs(critic_candidate))))
        diagnostics["update_count"] = self.update_count
        return diagnostics
