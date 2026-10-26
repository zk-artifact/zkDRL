"""Episodic objective/entropy trainer for the reference A2C ablation.

The environment's terminal flags NEVER change. Episodic mode masks only the
return estimator's time-limit bootstrap. Both policies and critics stay frozen
while collecting; no PPO, clipping or replay is introduced.
"""
from pathlib import Path
import hashlib
import struct
import sys
import time
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent

from ..model import env_init,env_step
from .core.a2c import ActorCritic,softmax_logsoftmax,categorical,gae
from .evaluate import Evaluator


def state_hash(model):
    h=hashlib.sha256(b'zkRL/A2C-float/state/v1\0')
    h.update(np.asarray(model.actor_params,dtype='<f8').tobytes())
    h.update(np.asarray(model.critic_params,dtype='<f8').tobytes())
    h.update(struct.pack('<Q',model.update_count))
    return h.hexdigest()


def collect(model,kind,params,uniforms,first_episode,horizon=6):
    """No parameter changes allowed while collecting this on-policy rollout."""
    version=model.update_count
    rows=[];episode_returns=[];episode_successes=[];trace=[]
    for offset,draws in enumerate(uniforms):
        state=env_init(kind);done=False;total=0.;success=False
        for t in range(horizon):
            before=state.copy();was_done=done
            if done:action=4
            else:
                z,_=model.actor.forward(model.actor_params,state.astype(np.float64)[None,:])
                probs,_=softmax_logsoftmax(z)
                action=categorical(probs[0],float(draws[t]))
            successor,reward,terminal=env_step(kind,state,action,params,None,done)
            reward=float(reward)
            if not np.isfinite(reward):raise ValueError('nonfinite environment reward')
            if not done:
                rows.append((state.copy(),action,reward,successor.copy(),bool(terminal),bool(not terminal and t+1<horizon)))
                success|=bool(kind=='browser' and terminal and reward>0)
            total+=reward;state,done=successor,bool(terminal)
            trace.append([first_episode+offset,t,*before,action,reward,*successor,int(was_done),int(terminal),version])
        episode_returns.append(total);episode_successes.append(success if kind=='browser' else None)
    if model.update_count!=version:raise RuntimeError('policy changed inside rollout')
    # The final active row must end an episode or terminate; no reset leakage.
    if not rows or rows[-1][-1]:raise RuntimeError('rollout must cut its final trace')
    states=np.asarray([x[0] for x in rows],dtype=np.float64)
    actions=np.asarray([x[1] for x in rows],dtype=np.int64)
    rewards=np.asarray([x[2] for x in rows],dtype=np.float64)
    successors=np.asarray([x[3] for x in rows],dtype=np.float64)
    terminals=np.asarray([x[4] for x in rows],dtype=np.float64)
    trace_continue=np.asarray([x[5] for x in rows],dtype=np.float64)
    return dict(states=states,actions=actions,rewards=rewards,successors=successors,terminals=terminals,
                trace_continue=trace_continue,planned_slots=len(uniforms)*horizon,
                episode_returns=episode_returns,episode_successes=episode_successes,trace=trace,policy_version=version)


def update_rollout(model,rollout,gamma,lam,bootstrap_time_limit=True):
    old_version=model.update_count
    if rollout['policy_version']!=old_version:raise ValueError('stale/off-policy rollout')
    values,_=model.critic.forward(model.critic_params,rollout['states'])
    next_values,_=model.critic.forward(model.critic_params,rollout['successors'])
    if not bootstrap_time_limit:
        # Rollouts contain whole six-slot episodes. A nonterminal trace cut is
        # precisely an administrative horizon end, not an MDP terminal event.
        cut=(rollout['terminals']==0)&(rollout['trace_continue']==0)
        next_values[cut,0]=0.
    advantages,targets=gae(rollout['rewards'],values[:,0],next_values[:,0],rollout['terminals'],
                           rollout['trace_continue'],gamma,lam)
    metrics=model.update(rollout['states'],rollout['actions'],advantages,targets,rollout['planned_slots'])
    metrics.update(active_rows=len(rollout['actions']),planned_slots=rollout['planned_slots'],
                   mean_advantage=float(np.mean(advantages)),mean_target=float(np.mean(targets)),
                   entropy_active_mean=float(metrics['entropy_fixed_denominator']*rollout['planned_slots']/len(rollout['actions'])))
    return metrics


def train(config,params,episodes=4000,checkpoints=tuple(range(0,4001,200))):
    """One registered arm; failures return all completed evidence, never a reset."""
    c=dict(config);kind=c['kind'];seed=int(c['seed']);batch_episodes=int(c['rollout_episodes']);horizon=6
    if episodes<=0 or batch_episodes<=0 or any(x<0 or x>episodes for x in checkpoints):raise ValueError('invalid episode budget')
    if any(x%batch_episodes for x in checkpoints if x!=episodes):raise ValueError('checkpoint must coincide with update boundary')
    model=ActorCritic(seed,hidden=int(c.get('hidden',16)),actor_lr=c['actor_lr'],critic_lr=c['critic_lr'],
                      entropy_coef=c['entropy_coef'],value_coef=c.get('value_coef',1.))
    evaluator=Evaluator(kind,params,horizon,c.get('gamma',.5))
    bank=np.random.default_rng(np.random.SeedSequence([1103,seed])).random((episodes,horizon))
    trajectory=hashlib.sha256(b'zkRL/A2C-float/trajectory/v1\0')
    initial=evaluator.model(model)
    curves=[];returns=[];successes=[];updates=[];interactions=0;finished=0;started=time.perf_counter()
    def checkpoint(ep):
        result=evaluator.model(model)
        result.update(episode=ep,updates=model.update_count,interactions=interactions,model_sha256=state_hash(model),
            training_window_return=float(np.mean(returns[-200:])) if returns else None,
            training_window_success=float(np.mean(successes[-200:])) if successes and kind=='browser' else None)
        curves.append(result)
    if 0 in checkpoints:checkpoint(0)
    error=None
    try:
        while finished<episodes:
            stop=min(finished+batch_episodes,episodes)
            schedule=c.get('entropy_schedule')
            if schedule:
                fraction=min(finished/schedule['decay_episodes'],1.)
                model.entropy_coef=schedule['start']+(schedule['end']-schedule['start'])*fraction
            rollout=collect(model,kind,np.asarray(params),bank[finished:stop],finished,horizon)
            if rollout['policy_version']!=model.update_count:raise RuntimeError('on-policy version invariant')
            metrics=update_rollout(model,rollout,c.get('gamma',.5),c['gae_lambda'],c.get('bootstrap_time_limit',True))
            metrics['entropy_coefficient']=float(model.entropy_coef)
            trajectory.update(np.asarray(rollout['trace'],dtype='<f8').tobytes())
            returns.extend(rollout['episode_returns']);successes.extend(rollout['episode_successes'])
            interactions+=len(rollout['actions']);finished=stop
            metrics['episode']=finished;updates.append(metrics)
            if finished in checkpoints:checkpoint(finished)
        final=evaluator.model(model)
    except (ValueError,OverflowError,FloatingPointError,RuntimeError) as exc:
        error=f'{type(exc).__name__}: {exc}';final=None
    record=dict(status='failed' if error else 'completed',error=error,config=c,episodes_requested=episodes,completed_episodes=finished,
        interactions=interactions,updates=model.update_count,params=list(map(float,params)),initial=initial,final=final,
        random_baseline=evaluator.random,oracle=evaluator.oracle,active_evaluation_states=len(evaluator.states),
        checkpoints=curves,episode_returns=returns,episode_successes=successes if kind=='browser' else None,
        update_diagnostics=updates,trajectory_sha256=trajectory.hexdigest(),final_model_sha256=state_hash(model),
        elapsed_seconds=time.perf_counter()-started,cryptographic_proof=False,arithmetic='float64')
    return record,model
