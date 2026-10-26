"""Behavior-only controlled DQN ablations. No changes to the proof relation.

Baseline float64/floor runs must reproduce the reference configuration before use.
Different precision/initialization/reward-rounding interventions are explicit.
"""
from dataclasses import dataclass, asdict, replace
import numpy as np
from .model import env_init, env_step, checked


@dataclass(frozen=True)
class Config:
    name: str
    kind: str
    seed: int
    scale: int | None = None
    hidden: int = 4
    capacity: int = 4
    batch: int = 1
    warmup: int = 4
    gamma: tuple = (1, 2)
    lr_div: int = 16
    eps_start: int = 4
    eps_end: int | None = None
    target_every: int = 4
    episodes: int = 300
    horizon: int = 6
    rounding: str = 'floor'
    init_quant_scale: int | None = None
    reward_quant_scale: int | None = None

    def __post_init__(self):
        assert self.scale in (None, 16, 256, 4096)
        assert 1 <= self.batch <= 8 and 1 <= self.warmup <= self.capacity
        assert self.hidden > 0 and self.lr_div > 0 and self.target_every > 0
        assert self.rounding in ('floor', 'nearest_update', 'nearest_math')

    @property
    def unit(self): return 1 if self.scale is None else self.scale

    @property
    def discount(self): return self.gamma[0] / self.gamma[1]


def divide(x, divisor, cfg, update=False):
    """Exact integer round-to-nearest, ties-to-even, or original floor."""
    q = np.floor_divide(x, divisor)
    nearest = cfg.rounding == 'nearest_math' or (update and cfg.rounding == 'nearest_update')
    if not nearest: return q
    remainder = np.asarray(x) - q * divisor
    up = (2 * remainder > divisor) | ((2 * remainder == divisor) & (q % 2 != 0))
    return q + up.astype(np.int64)


def initialization(cfg):
    rng = np.random.default_rng(cfg.seed)
    h = cfg.hidden
    w = np.concatenate([rng.normal(0, .20, 4*h), np.full(h, .25),
                        rng.normal(0, .20, 4*h), np.zeros(4)])
    if cfg.scale is not None:
        return np.floor(w * cfg.scale).astype(np.int64)
    if cfg.init_quant_scale is not None:
        return np.floor(w * cfg.init_quant_scale) / cfg.init_quant_scale
    return w


def forward(weights, state, cfg):
    h = cfg.hidden
    checked(weights)
    if cfg.scale is not None and (np.any(weights < -2**20) or np.any(weights >= 2**20)):
        raise OverflowError('signed-21 weight bounds')
    if np.any(np.asarray(state) < 0) or np.any(np.asarray(state) >= 2**16):
        raise OverflowError('unsigned-16 state bounds')
    w1, b1 = weights[:4*h].reshape(h, 4), weights[4*h:5*h]
    w2, b2 = weights[5*h:9*h].reshape(4, h), weights[9*h:]
    raw = w1 @ state + b1; checked(raw)
    activation = np.maximum(raw, 0)
    q = w2 @ activation
    if cfg.scale is not None:
        q = divide(q, cfg.scale, cfg); checked(q)
    q = q + b2; checked(q)
    return q, activation


def training_params(base, cfg):
    if cfg.scale is not None: return np.floor(base * cfg.scale).astype(np.int64)
    if cfg.reward_quant_scale is not None:
        return np.floor(base * cfg.reward_quant_scale) / cfg.reward_quant_scale
    return np.asarray(base).copy()


class Learner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.weights = initialization(cfg)
        self.target = self.weights.copy()
        self.buffer = np.zeros((cfg.capacity, 11), dtype=self.weights.dtype)
        self.interactions = self.updates = 0
        self.zero_updates = self.successful_replay_samples = self.successful_replay_updates = 0
        self.total_replay_samples = self.delta_coordinates = self.delta_zero_coordinates = 0
        self.positive_subtick = self.negative_subtick = 0
        self.positive_subtick_zero = self.negative_subtick_zero = 0
        self.round_error_sum = self.abs_delta_sum = 0.0

    def observe(self, state, action, reward, successor, terminal, quantiles):
        c = self.cfg; h = c.hidden
        self.buffer[self.interactions % c.capacity] = np.concatenate([state, [action, reward], successor, [int(terminal)]])
        self.interactions += 1
        if self.interactions < c.warmup: return
        indices = np.floor(quantiles[:c.batch] * min(self.interactions, c.capacity)).astype(int)
        g1 = np.zeros((h,4), dtype=self.weights.dtype)
        gb1 = np.zeros(h, dtype=self.weights.dtype)
        g2 = np.zeros((4,h), dtype=self.weights.dtype)
        gb2 = np.zeros(4, dtype=self.weights.dtype)
        successes = 0
        for index in indices:
            row = self.buffer[index]
            s, a, r, ns, done = row[:4], int(row[4]), row[5], row[6:10], bool(row[10])
            qnext, _ = forward(self.target, ns, c)
            if done: y = r
            elif c.scale is None: y = r + c.discount * qnext.max()
            else: y = r + divide(c.gamma[0] * qnext.max(), c.gamma[1], c)
            q, activation = forward(self.weights, s, c)
            error = q[a] - y; checked(error)
            outgoing = self.weights[5*h+a*h:5*h+(a+1)*h]
            if c.scale is None:
                dh = error * outgoing * (activation > 0)
                grad2 = error * activation
            else:
                dhraw = divide(error * outgoing, c.scale, c); checked(dhraw)
                dh = dhraw * (activation > 0)
                grad2 = divide(error * activation, c.scale, c); checked(grad2)
            checked(dh)
            g1 += np.outer(dh, s); gb1 += dh
            g2[a] += grad2; gb2[a] += error
            successes += int(c.kind == 'browser' and done and r > 0)
        gradient = np.concatenate([g1.reshape(-1), gb1, g2.reshape(-1), gb2])
        divisor = c.lr_div * c.batch
        raw_delta = gradient / divisor
        delta = raw_delta if c.scale is None else divide(gradient, divisor, c, update=True)
        new_weights = self.weights - delta; checked(new_weights)
        if c.scale is not None and (np.any(new_weights < -2**20) or np.any(new_weights >= 2**20)):
            raise OverflowError('signed-21 updated weight bounds')
        self.zero_updates += int(np.all(delta == 0))
        self.delta_coordinates += delta.size
        self.delta_zero_coordinates += int(np.count_nonzero(delta == 0))
        self.abs_delta_sum += float(np.abs(delta).sum()) / c.unit
        if c.scale is not None:
            positive = (gradient > 0) & (gradient < divisor)
            negative = (gradient < 0) & (gradient > -divisor)
            self.positive_subtick += int(positive.sum()); self.negative_subtick += int(negative.sum())
            self.positive_subtick_zero += int(np.count_nonzero(positive & (delta == 0)))
            self.negative_subtick_zero += int(np.count_nonzero(negative & (delta == 0)))
            self.round_error_sum += float(np.sum(delta - raw_delta)) / c.unit
        self.successful_replay_samples += successes
        self.successful_replay_updates += int(successes > 0)
        self.total_replay_samples += c.batch
        self.weights = new_weights; self.updates += 1
        if self.updates % c.target_every == 0: self.target = new_weights.copy()

    def diagnostics(self):
        return {'updates':self.updates, 'interactions':self.interactions,
                'zero_delta_updates':self.zero_updates,
                'zero_delta_fraction':self.zero_updates / self.updates if self.updates else None,
                'zero_coordinate_fraction':self.delta_zero_coordinates / self.delta_coordinates if self.delta_coordinates else None,
                'mean_absolute_parameter_step':self.abs_delta_sum / self.delta_coordinates if self.delta_coordinates else None,
                'mean_final_rounding_error':self.round_error_sum / self.delta_coordinates if self.delta_coordinates else None,
                'positive_subtick_count':self.positive_subtick, 'positive_subtick_zero_count':self.positive_subtick_zero,
                'negative_subtick_count':self.negative_subtick, 'negative_subtick_zero_count':self.negative_subtick_zero,
                'successful_replay_samples':self.successful_replay_samples,
                'successful_replay_updates':self.successful_replay_updates,
                'total_replay_samples':self.total_replay_samples}


def evaluate(weights, cfg, native_params, original_params):
    state, done = env_init(cfg.kind), False
    total = common = discounted = 0.0
    actions = []; success = False
    for t in range(cfg.horizon):
        q, _ = forward(weights, state, cfg)
        action = 4 if done else int(np.argmax(q))
        successor, reward, terminal = env_step(cfg.kind,state,action,native_params,cfg.scale,done)
        common_next, common_reward, common_done = env_step(cfg.kind,state,action,original_params,None,done)
        assert np.array_equal(successor,common_next) and terminal==common_done
        total += float(reward)/cfg.unit; common += float(common_reward)
        discounted += cfg.discount**t * float(reward)/cfg.unit
        success |= cfg.kind=='browser' and not done and terminal and reward > 0
        actions.append(action);state,done = successor,terminal
    return {'greedy_return':total,'common_reward_return':common,'greedy_discounted_return':discounted,
            'greedy_success':bool(success) if cfg.kind=='browser' else None,'greedy_actions':actions}


def browser_oracle(gamma=.5):
    states = [np.array([u,p,0,1]) for u,p in ((0,0),(0,1),(1,0),(1,1))]
    lookup = {tuple(s):i for i,s in enumerate(states)}
    values = np.zeros(4)
    for _ in range(10000):
        q = np.empty((4,4))
        for i,state in enumerate(states):
            for a in range(4):
                successor,r,d = env_step('browser',state,a,np.zeros(4),None)
                q[i,a] = r if d else r+gamma*values[lookup[tuple(successor)]]
        updated = q.max(axis=1)
        if np.max(np.abs(updated-values)) < 1e-13: return states,q
        values = updated
    raise RuntimeError('Oracle value iteration failed')


def constructed_browser_weights(gamma=.5):
    """Exact Q* representation in the existing four-unit network, not training."""
    _,qstar = browser_oracle(gamma)
    w = np.zeros(40)
    w[:16] = np.array([[1,0,0,0],[0,1,0,0],[1,1,0,0],[0,0,0,0]]).reshape(-1)
    w[18] = -1  # third hidden unit is the Boolean AND, ReLU(u+p-1)
    w[20+0*4+1] = qstar[1,0]-qstar[0,0]; w[36] = qstar[0,0]
    w[20+1*4+1] = qstar[1,1]-qstar[0,1]; w[37] = qstar[0,1]
    w[20+2*4+0] = qstar[2,2]-qstar[0,2]; w[38] = qstar[0,2]
    w[20+3*4+2] = 2; w[39] = -1
    return w


def train(cfg, original_params):
    params = training_params(original_params,cfg)
    draws = np.random.default_rng(10000+cfg.seed).integers(0,[16,4,4],size=(cfg.episodes,cfg.horizon,3))
    # The first replay quantile reproduces the original four-way index exactly.
    # The same extra uniform bank is shared across configurations of a seed.
    quantiles = np.random.default_rng(30000+cfg.seed).random((cfg.episodes,cfg.horizon,8))
    quantiles[:,:,0] = (draws[:,:,2]+quantiles[:,:,0])/4
    learner = Learner(cfg)
    returns = []; successes=0; first_success=None
    state_visits = np.zeros(4,dtype=int); state_actions = np.zeros((4,4),dtype=int)
    ready_visits = ready_submits = 0
    try:
        for episode in range(cfg.episodes):
            eps = cfg.eps_start if cfg.eps_end is None else int(round(cfg.eps_start+(cfg.eps_end-cfg.eps_start)*episode/max(1,cfg.episodes-1)))
            state,done,total = env_init(cfg.kind),False,0.0
            for t in range(cfg.horizon):
                q,_ = forward(learner.weights,state,cfg)
                action = 4 if done else int(draws[episode,t,1]) if draws[episode,t,0]<eps else int(np.argmax(q))
                successor,reward,terminal = env_step(cfg.kind,state,action,params,cfg.scale,done)
                if not done:
                    if cfg.kind=='browser':
                        index=int(state[0])*2+int(state[1]);state_visits[index]+=1;state_actions[index,action]+=1
                        ready_visits += int(index==3); ready_submits += int(index==3 and action==3)
                        if terminal and reward>0:
                            successes+=1
                            if first_success is None:first_success=episode
                    learner.observe(state,action,reward,successor,terminal,quantiles[episode,t])
                total+=float(reward)/cfg.unit;state,done=successor,terminal
            returns.append(total)
    except (OverflowError,ValueError,FloatingPointError) as exc:
        return {'status':'failed','error':f'{type(exc).__name__}: {exc}',
                'completed_episodes':len(returns),'config':asdict(cfg),**learner.diagnostics()},learner.weights,returns
    record={'status':'completed','config':asdict(cfg),**evaluate(learner.weights,cfg,params,original_params),
            **learner.diagnostics(),'training_successes':successes if cfg.kind=='browser' else None,
            'first_success_episode':first_success,'ready_state_visits':ready_visits,
            'ready_state_submits':ready_submits,'state_visits':state_visits.tolist() if cfg.kind=='browser' else None,
            'state_actions':state_actions.tolist() if cfg.kind=='browser' else None,
            'first50_return':float(np.mean(returns[:50])),'last50_return':float(np.mean(returns[-50:]))}
    if cfg.kind=='browser':
        states,qstar=browser_oracle(cfg.discount)
        learned=np.array([forward(learner.weights,s,cfg)[0]/cfg.unit for s in states])
        record['browser_q_values']=learned.tolist()
        record['browser_q_rmse']=float(np.sqrt(np.mean((learned-qstar)**2)))
        record['browser_greedy_actions_by_state']=learned.argmax(axis=1).tolist()
    return record,learner.weights,returns
