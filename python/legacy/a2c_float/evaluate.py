"""Enumerated finite-horizon evaluation, not Monte Carlo and not a proof.

Uses existing model.env_step. The actor/critic use float64; rewards deliberately
match the S16-grid computational MDP already proved for DQN.
"""
from pathlib import Path
from functools import lru_cache
import sys
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent

from ..model import env_init,env_step
from .core.a2c import softmax_logsoftmax


class Evaluator:
    def __init__(self,kind,params,horizon=6,gamma=.5):
        if kind not in ('browser','recommendation'):raise ValueError(kind)
        self.kind=kind;self.params=np.asarray(params,dtype=np.float64)
        self.horizon=int(horizon);self.gamma=float(gamma)
        if self.params.shape!=(4,) or not np.isfinite(self.params).all():raise ValueError('four finite parameters required')
        self.initial=tuple(env_init(kind).tolist())
        # Enumerate all active decision states under any action within horizon.
        current={self.initial};active=set()
        for _ in range(self.horizon):
            active.update(current);following=set()
            for state in current:
                for action in range(4):
                    successor,_,terminal=self.transition(state,action)
                    if not terminal:following.add(successor)
            current=following
        self.states=sorted(active);self.index={s:j for j,s in enumerate(self.states)}
        self.state_array=np.asarray(self.states,dtype=np.float64)
        self.oracle=self._oracle()
        self.random=self.probabilities(np.full((len(self.states),4),.25))

    @lru_cache(None)
    def transition(self,state,action):
        successor,reward,terminal=env_step(self.kind,np.asarray(state,dtype=np.int64),int(action),self.params,None,False)
        return tuple(successor.tolist()),float(reward),bool(terminal)

    def _oracle(self):
        @lru_cache(None)
        def value(state,remaining):
            if remaining==0:return (0.,0.,0.)
            rows=[]
            for action in range(4):
                ns,reward,terminal=self.transition(state,action)
                future=(0.,0.,0.) if terminal else value(ns,remaining-1)
                success=float(self.kind=='browser' and terminal and reward>0)
                rows.append((reward+future[0],reward+self.gamma*future[1],success+future[2]))
            return tuple(max(r[j] for r in rows) for j in range(3))
        undisc,disc,success=value(self.initial,self.horizon)
        return dict(undiscounted_return=undisc,discounted_return=disc,success_probability=success if self.kind=='browser' else None,
                    dp_states=value.cache_info().currsize)

    def probabilities(self,p):
        p=np.asarray(p,dtype=np.float64)
        if p.shape!=(len(self.states),4) or np.any(p<0) or not np.isfinite(p).all() or not np.allclose(p.sum(axis=1),1,rtol=0,atol=2e-14):
            raise ValueError('invalid policy probabilities')
        @lru_cache(None)
        def value(state,remaining):
            if remaining==0:return (0.,0.,0.)
            out=np.zeros(3,dtype=np.float64)
            for action,probability in enumerate(p[self.index[state]]):
                if probability==0:continue
                ns,reward,terminal=self.transition(state,action)
                f=(0.,0.,0.) if terminal else value(ns,remaining-1)
                success=float(self.kind=='browser' and terminal and reward>0)
                out+=probability*np.array([reward+f[0],reward+self.gamma*f[1],success+f[2]])
            return tuple(map(float,out))
        u,d,s=value(self.initial,self.horizon)
        return dict(undiscounted_return=u,discounted_return=d,success_probability=s if self.kind=='browser' else None)

    def model(self,actor_critic):
        logits,_=actor_critic.actor.forward(actor_critic.actor_params,self.state_array)
        p,logp=softmax_logsoftmax(logits)
        greedy=np.eye(4,dtype=np.float64)[np.argmax(logits,axis=1)]
        result=dict(stochastic=self.probabilities(p),greedy=self.probabilities(greedy))
        result['initial_state_entropy']=float(-np.sum(p[self.index[self.initial]]*logp[self.index[self.initial]]))
        return result


def parameters(kind):
    if kind=='browser':return np.zeros(4,dtype=np.float64)
    if kind!='recommendation':raise ValueError(kind)
    with np.load(ROOT.parent/'data/recommendation.npz') as d:
        return np.floor(np.asarray(d['rewards'][0],dtype=np.float64)*16)/16
