"""Prospective numerical runs. No guest, no unchecked environment trace input."""
from dataclasses import dataclass
import hashlib
import time
import numpy as np
from .fixed_math import (FixedModel,G,P,forward,backward,masses,categorical,finite_returns,
                        actor_upstream,shadow_gradients,bound)
from .fixtures import tape,reward_params,real_params
from .float_bridge import (FloatModel,Evaluator,parameters,env_init,env_step,
                          float_collect,float_update,float_hash)
from .fixed_eval import FixedEvaluator


@dataclass
class RunResult:
    record: dict
    model: object
    traces: np.ndarray
    actor_checkpoints: np.ndarray
    critic_checkpoints: np.ndarray


def collect_fixed(model,kind,draws,episode,audit=None):
    if np.asarray(draws).shape!=(6,):raise ValueError('exactly six planned draws')
    actor,critic=model.snapshot(); fingerprint=model.fingerprint(); version=model.update_count
    state=env_init(kind); done=False; rows=[]; traces=[]; total=0; success=False
    params=reward_params(kind,model.bits); s=1<<model.bits
    for t,draw in enumerate(draws):
        before=state.copy(); previous_done=done
        if done:action=4
        else:
            logits,_=forward(actor,state[None,:],4,model.bits,audit)
            mass=masses(logits,model.bits,audit)[0]
            action=categorical(mass,int(draw))
        successor,reward,terminal=env_step(kind,state,action,params,s,done)
        if not isinstance(reward,(int,np.integer)):raise ValueError('integer reward required')
        reward=int(reward)
        if not done:
            rows.append((state.copy(),action,reward,successor.copy(),int(terminal),int(not terminal and t<5),mass.copy()))
            success|=bool(kind=='browser' and terminal and reward>0)
        total+=reward
        traces.append([episode,t,*map(int,before),action,reward,*map(int,successor),int(previous_done),int(terminal),version,int(draw)])
        state,done=successor,bool(terminal)
    if model.fingerprint()!=fingerprint or model.update_count!=version:raise RuntimeError('model changed during collection')
    return dict(states=np.array([r[0] for r in rows],dtype=object),actions=np.array([r[1] for r in rows],dtype=object),
                rewards=np.array([r[2] for r in rows],dtype=object),successors=np.array([r[3] for r in rows],dtype=object),
                terminals=np.array([r[4] for r in rows],dtype=object),trace_continue=np.array([r[5] for r in rows],dtype=object),
                mass=np.array([r[6] for r in rows],dtype=object),planned_slots=6,trace=traces,
                episode_return=total/s,episode_success=success,policy_version=version,
                fingerprint=fingerprint,actor_snapshot=actor,critic_snapshot=critic)


def gradients_fixed(model,rollout,audit=None,shadow=True):
    if rollout['policy_version']!=model.update_count or rollout['fingerprint']!=model.fingerprint():raise ValueError('stale/off-policy rollout')
    if rollout['planned_slots']!=6 or not 1<=len(rollout['actions'])<=6:raise ValueError('invalid planned slots/active rows')
    current=model.snapshot()
    if not all(np.array_equal(v,w) for v,w in zip(current,(rollout['actor_snapshot'],rollout['critic_snapshot']))):raise ValueError('mutated snapshot')
    z,ac=forward(current[0],rollout['states'],4,model.bits,audit)
    v,cc=forward(current[1],rollout['states'],1,model.bits,audit)
    mass=masses(z,model.bits,audit)
    if not np.array_equal(mass,rollout['mass']):raise ValueError('policy differs from collection')
    target=finite_returns(rollout['rewards'],rollout['terminals'],rollout['trace_continue'],model.bits,audit)
    adv=target-v[:,0]*(G//(1<<model.bits))
    bound(adv,65536*G,'advantages',audit)
    dz=actor_upstream(z,mass,rollout['actions'],adv,model.bits,audit)
    dv=-adv[:,None]  # derivative of 1/2*(V-target)^2, no factor2
    ga=backward(ac,dz,4,model.bits,audit); gc=backward(cc,dv,1,model.bits,audit)
    error_a=error_c=0.
    if shadow:
        sa,sc=shadow_gradients(ac,cc,z,mass,rollout['actions'],adv,target,model.bits)
        error_a=float(np.max(np.abs(np.asarray(ga,dtype=np.float64)/G-sa)))
        error_c=float(np.max(np.abs(np.asarray(gc,dtype=np.float64)/G-sc)))
    metrics=dict(active_rows=len(adv),planned_slots=6,mean_advantage=float(np.mean(np.asarray(adv,dtype=np.float64)/G)),
                 mean_target=float(np.mean(np.asarray(target,dtype=np.float64)/G)),
                 shadow_actor_max_abs_error=error_a,shadow_critic_max_abs_error=error_c,
                 actor_gradient_max_abs=max(abs(int(x)) for x in ga)/G,
                 critic_gradient_max_abs=max(abs(int(x)) for x in gc)/G,
                 zero_probability_cells=int(np.count_nonzero(mass==0)),
                 deterministic_active_rows=int(np.count_nonzero(np.max(mass,axis=1)==P)))
    return ga,gc,metrics,dict(logits=z,values=v,mass=mass,targets=target,advantages=adv,dz=dz,dv=dv)


def update_fixed(model,rollout,audit=None):
    ga,gc,metrics,_=gradients_fixed(model,rollout,audit)
    old_a,old_c=model.actor_master.copy(),model.critic_master.copy(); old=model.snapshot()
    model.commit(ga,gc,rollout['planned_slots'],rollout['fingerprint'],audit)
    after=model.snapshot(); ratio=G//(1<<model.bits)
    metrics['master_cells_changed']=sum(int(np.count_nonzero(a!=b)) for a,b in zip((old_a,old_c),(model.actor_master,model.critic_master)))
    metrics['forward_cells_changed']=sum(int(np.count_nonzero(a!=b)) for a,b in zip(old,after))
    metrics['master_residual_max']=max(int(np.max(model.actor_master%ratio)),int(np.max(model.critic_master%ratio)))
    metrics['max_abs_master_real']=max(max(abs(int(x)) for x in model.actor_master),max(abs(int(x)) for x in model.critic_master))/G
    metrics['update_count']=model.update_count
    return metrics


def train(kind,seed,mode='q16_m40_p32',episodes=4000,checkpoints=None):
    if kind not in ('browser','recommendation'):raise ValueError('unknown environment')
    if mode not in ('float_control','q16_m40_p32','q20_m40_p32'):raise ValueError('unregistered mode')
    if type(episodes)!=int or episodes<=0:raise ValueError('positive episode count')
    checkpoints=tuple(range(0,episodes+1,200)) if checkpoints is None else tuple(checkpoints)
    if any(type(x)!=int or not 0<=x<=episodes for x in checkpoints):raise ValueError('invalid checkpoint')
    fixed=mode!='float_control'; bits=16 if mode=='q16_m40_p32' else 20
    model=FixedModel(seed,bits) if fixed else FloatModel(seed,hidden=16,actor_lr=1/32,critic_lr=1/64,entropy_coef=1/128)
    float_params=real_params(kind)
    evaluator=(FixedEvaluator if fixed else Evaluator)(kind,float_params,6,7/8)
    bank=tape(seed,episodes); uniforms=np.asarray(bank,dtype=np.float64)/P
    fingerprint=model.fingerprint if fixed else lambda:float_hash(model)
    started=time.perf_counter(); initial=evaluator.model(model)
    returns=[]; successes=[]; updates=[]; curves=[]; traces=[]; am=[]; cm=[]; audit={}; interactions=0; finished=0
    trajectory=hashlib.sha256(b'zkRL/A2C-fixed/trajectory/v1\0' if fixed else b'zkRL/A2C-float-new-tape/trajectory/v1\0')
    def checkpoint(ep):
        row=evaluator.model(model)
        row.update(episode=ep,updates=model.update_count,interactions=interactions,model_sha256=fingerprint(),
                   training_window_return=float(np.mean(returns[-200:])) if returns else None,
                   training_window_success=float(np.mean(successes[-200:])) if successes and kind=='browser' else None)
        curves.append(row)
        am.append(np.asarray(model.actor_master if fixed else model.actor_params,dtype=np.int64 if fixed else np.float64).copy())
        cm.append(np.asarray(model.critic_master if fixed else model.critic_params,dtype=np.int64 if fixed else np.float64).copy())
    if 0 in checkpoints:checkpoint(0)
    error=None; pending_trace=None
    try:
        for ep in range(episodes):
            if fixed:
                rollout=collect_fixed(model,kind,bank[ep],ep,audit)
                pending_trace=rollout['trace']
                metrics=update_fixed(model,rollout,audit)
                total=rollout['episode_return']; success=rollout['episode_success']; active=len(rollout['actions'])
                encoded=np.asarray(pending_trace,dtype='<i8')
            else:
                rollout=float_collect(model,kind,float_params,uniforms[ep:ep+1],ep,6)
                pending_trace=rollout['trace']
                metrics=float_update(model,rollout,7/8,1,False)
                total=rollout['episode_returns'][0]; success=rollout['episode_successes'][0]; active=len(rollout['actions'])
                encoded=np.asarray(pending_trace,dtype='<f8')
            trajectory.update(encoded.tobytes()); traces.extend(pending_trace); pending_trace=None
            returns.append(total); successes.append(success); metrics['episode']=ep+1
            metrics['model_sha256']=fingerprint(); updates.append(metrics)
            interactions+=active; finished=ep+1
            if finished in checkpoints:checkpoint(finished)
        final=evaluator.model(model)
    except (ValueError,OverflowError,FloatingPointError,RuntimeError) as exc:
        error=f'{type(exc).__name__}: {exc}'; final=None
    record=dict(schema='zkrl-a2c-fixed-run-v1',status='failed' if error else 'completed',error=error,
                config=dict(kind=kind,seed=seed,mode=mode),episodes_requested=episodes,completed_episodes=finished,
                interactions=interactions,updates=model.update_count,initial=initial,final=final,
                random_baseline=evaluator.random,oracle=evaluator.oracle,active_evaluation_states=len(evaluator.states),
                checkpoints=curves,episode_returns=returns,episode_successes=successes if kind=='browser' else None,
                update_diagnostics=updates,numerical_maxima=audit,trajectory_sha256=trajectory.hexdigest(),
                final_model_sha256=fingerprint(),tape_sha256=hashlib.sha256(bank.astype('<u8').tobytes()).hexdigest(),
                failed_attempt_trace=pending_trace,elapsed_seconds=time.perf_counter()-started,
                arithmetic='Python arbitrary integers in NumPy object arrays' if fixed else 'float64 with new uint32 tape',
                cryptographic_proof=False,guest_execution=False)
    return RunResult(record,model,np.asarray(traces,dtype=np.int64 if fixed else np.float64),np.asarray(am),np.asarray(cm))
