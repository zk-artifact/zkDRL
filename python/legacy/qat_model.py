"""Mixed-precision/QAT diagnostics. Behavioral only, no proof implementation.

Exact-forward QAT uses integer inference and a straight-through surrogate
backward pass. Integer high-resolution masters provide a separately implemented
candidate with the same dyadic update equations for this bounded two-layer MLP.
"""
from dataclasses import dataclass,asdict
import hashlib
import numpy as np
from . import control_model as ref
from .model import checked,env_init,env_step

PROFILES={
 'original':dict(hidden=4,capacity=4,batch=1,warmup=4,gamma=(1,2),lr_div=16,eps_start=4,eps_end=None,target_every=4),
 'stable':dict(hidden=16,capacity=64,batch=8,warmup=32,gamma=(1,2),lr_div=64,eps_start=8,eps_end=1,target_every=32),
}
MODES=('fp','fp_grid','integer','integer_nearest_update','residual_low','fp_master_intgrad',
       'qat_exact','residual_high','qat_nearest','residual_high_nearest','qat_weights')

@dataclass(frozen=True)
class Spec:
    name:str
    kind:str
    seed:int
    profile:str='original'
    mode:str='fp'
    scale:int=16
    episodes:int=2000
    phase:int=0
    pretrained_episodes:int=0
    checkpoints:tuple=tuple(range(200,2001,200))

    def __post_init__(self):
        if self.mode not in MODES or self.profile not in PROFILES:raise ValueError('unsupported profile/mode')
        if self.scale not in (16,256,4096):raise ValueError('unsupported scale')
        if self.kind not in ('browser','recommendation'):raise ValueError('unsupported environment')

    def cfg(self):
        kw=PROFILES[self.profile]
        fp=self.mode in ('fp','fp_grid')
        return ref.Config(self.name,self.kind,self.seed,None if fp else self.scale,
             episodes=self.episodes,**kw,
             init_quant_scale=self.scale if self.mode=='fp_grid' else None,
             reward_quant_scale=self.scale if self.mode=='fp_grid' else None)


def nearest_div(x,d):
    q=np.floor_divide(x,d);r=np.asarray(x)-q*d
    return q+((2*r>d)|((2*r==d)&(q%2!=0))).astype(np.int64)


def qdivide(x,d,nearest=False):return nearest_div(x,d) if nearest else np.floor_divide(x,d)


def quantize(x,scale,nearest=False):
    v=np.asarray(x)*scale
    if not np.isfinite(v).all():raise OverflowError('nonfinite master')
    q=np.rint(v) if nearest else np.floor(v)
    if np.any(q < -2**20) or np.any(q >= 2**20):raise OverflowError('coarse signed-21 parameter bounds')
    return q.astype(np.int64)


def net_batch(qw,states,h,scale,nearest=False,weight_only=False):
    states=np.asarray(states)
    w1=qw[:4*h].reshape(h,4);b1=qw[4*h:5*h]
    w2=qw[5*h:9*h].reshape(4,h);b2=qw[9*h:]
    raw=states@w1.T+b1;checked(raw)
    act=np.maximum(raw,0)
    dot=act@w2.T
    if weight_only:
        return dot/(scale*scale)+b2/scale,act/scale
    output=qdivide(dot,scale,nearest);checked(output)
    output=output+b2;checked(output)
    return output,act


def components(qw,target,rows,cfg,scale,nearest=False):
    """Quantized forward, stopped quantized TD target, low/high integer gradients.

    High gradients have scale S^2; low gradients have scale S. Diagnostics use
    only bounded intermediates; no saturation or overflow is silently accepted.
    """
    h=cfg.hidden;states=rows[:,:4].astype(np.int64);nextstates=rows[:,6:10].astype(np.int64)
    if np.any(states<0) or np.any(states>cfg.horizon) or np.any(nextstates<0) or np.any(nextstates>cfg.horizon):
        raise OverflowError('reference state domain exceeded')
    actions=rows[:,4].astype(int);done=rows[:,10].astype(bool);rewards=rows[:,5].astype(np.int64)
    q,act=net_batch(qw,states,h,scale,nearest)
    future,_=net_batch(target,nextstates,h,scale,nearest)
    y=rewards+np.where(done,0,qdivide(cfg.gamma[0]*future.max(axis=1),cfg.gamma[1],nearest))
    err=q[np.arange(len(rows)),actions]-y;checked(err)
    onehot=np.eye(4,dtype=np.int64)[actions]
    selected=qw[5*h:9*h].reshape(4,h)[actions]
    dh_hi=err[:,None]*selected*(act>0)
    g2_hi=err[:,None]*act
    if np.any(np.abs(dh_hi)>=2**46) or np.any(np.abs(g2_hi)>=2**50):
        raise OverflowError('exact-gradient diagnostic range exceeded')
    dh_lo=qdivide(err[:,None]*selected,scale,nearest)*(act>0);checked(dh_lo)
    g2_lo=qdivide(g2_hi,scale,nearest);checked(g2_lo)
    low=np.concatenate([(dh_lo.T@states).reshape(-1),dh_lo.sum(axis=0),
                        (onehot.T@g2_lo).reshape(-1),onehot.T@err])
    high=np.concatenate([(dh_hi.T@states).reshape(-1),dh_hi.sum(axis=0),
                         (onehot.T@g2_hi).reshape(-1),scale*(onehot.T@err)])
    if np.any(np.abs(high)>=2**50):raise OverflowError('high-gradient accumulator bounds')
    # Independently evaluate the STE backward equations in real units.
    real_error=q[np.arange(len(rows)),actions]/scale-y/scale
    dh_real=real_error[:,None]*(selected/scale)*(act>0)
    g2_real=real_error[:,None]*(act/scale)
    full=np.concatenate([(dh_real.T@states).reshape(-1),dh_real.sum(axis=0),
                         (onehot.T@g2_real).reshape(-1),onehot.T@real_error])
    if not np.array_equal(full*(scale*scale),high.astype(np.float64)):
        raise ArithmeticError('STE and high-integer gradients are not exactly equal')
    return low,high,full,err,act


class MixedLearner:
    def __init__(self,spec,initial=None):
        self.spec=spec;self.cfg=spec.cfg();c=self.cfg;S=spec.scale
        self.nearest=spec.mode in ('qat_nearest','residual_high_nearest')
        self.D=c.lr_div*c.batch
        self.factor=(self.D if spec.mode in ('residual_low','fp_master_intgrad') else self.D*S)
        self.fine_scale=self.factor*S
        self.is_integer=spec.mode in ('integer','integer_nearest_update','residual_low','residual_high','residual_high_nearest')
        self.coarse_only=spec.mode in ('integer','integer_nearest_update')
        start=ref.initialization(c)  # common floored initialization, already scaled by S
        if initial is not None:
            start=quantize(initial,S,self.nearest)
        if self.coarse_only:self.master=start.copy()
        elif initial is not None:
            fine=np.rint(np.asarray(initial)*self.fine_scale).astype(np.int64)
            self.master=fine if self.is_integer else fine/self.fine_scale
        else:self.master=start*self.factor if self.is_integer else start/S
        self.target=self.master.copy()
        self.buffer=np.zeros((c.capacity,11),dtype=np.int64)
        self.interactions=self.updates=0
        self.state_zero=self.coarse_zero=self.low_lost=self.high_nonzero=0
        self.bias_sum=0.0;self.low_error_sum=0.0;self.error_den_sum=0.0
        self.residue_nonzero=self.residue_count=0

    def qweights(self,target=False):
        m=self.target if target else self.master
        if self.coarse_only:return m.copy()
        if self.is_integer:
            q=qdivide(m,self.factor,self.nearest)
            if np.any(q < -2**20) or np.any(q >= 2**20):raise OverflowError('coarse parameter bounds')
            return q
        return quantize(m,self.spec.scale,self.nearest)

    def real_master(self):
        if self.coarse_only:return self.master/self.spec.scale
        return self.master/self.fine_scale if self.is_integer else self.master.copy()

    def canonical_master(self,target=False):
        m=self.target if target else self.master
        if self.coarse_only:return m
        if self.is_integer:return m
        # Exact-forward STE arithmetic stays on this dyadic lattice. Weight-only
        # QAT uses a different lattice and is not included in equality claims.
        units=m*self.fine_scale
        if self.spec.mode!='qat_weights' and not np.array_equal(units,np.rint(units)):
            raise ArithmeticError('floating master left the asserted integer lattice')
        return np.rint(units).astype(np.int64)

    def qvalues(self,state,deployment='native'):
        S=self.spec.scale
        if deployment=='floor':
            qw=quantize(self.real_master(),S,False)
            q,_=net_batch(qw,np.asarray(state)[None,:],self.cfg.hidden,S,False)
            return q[0]/S
        qw=self.qweights()
        weight_only=self.spec.mode=='qat_weights' and deployment=='native'
        q,_=net_batch(qw,np.asarray(state)[None,:],self.cfg.hidden,S,self.nearest,weight_only)
        return q[0] if weight_only else q[0]/S

    def observe(self,state,action,reward,nextstate,done,quantiles):
        c=self.cfg;S=self.spec.scale
        self.buffer[self.interactions%c.capacity]=np.concatenate([state,[action,reward],nextstate,[int(done)]])
        self.interactions+=1
        if self.interactions<c.warmup:return
        idx=np.floor(quantiles[:c.batch]*min(self.interactions,c.capacity)).astype(int)
        rows=self.buffer[idx];qw=self.qweights();target=self.qweights(True)
        old_master=self.master.copy()
        if self.spec.mode=='qat_weights':
            states=rows[:,:4];actions=rows[:,4].astype(int)
            q,act=net_batch(qw,states,c.hidden,S,weight_only=True)
            future,_=net_batch(target,rows[:,6:10],c.hidden,S,weight_only=True)
            y=rows[:,5]/S+np.where(rows[:,10]!=0,0,c.discount*future.max(axis=1))
            error=q[np.arange(len(rows)),actions]-y
            selected=qw[5*c.hidden:9*c.hidden].reshape(4,c.hidden)[actions]/S
            dh=error[:,None]*selected*(act>0);onehot=np.eye(4)[actions]
            grad=np.concatenate([(dh.T@states).reshape(-1),dh.sum(axis=0),
                 (onehot.T@(error[:,None]*act)).reshape(-1),onehot.T@error])
            self.master=self.master-grad/self.D
        else:
            low,high,full,error,act=components(qw,target,rows,c,S,self.nearest)
            self.low_lost+=int(np.count_nonzero((high!=0)&(low==0)))
            self.high_nonzero+=int(np.count_nonzero(high))
            self.low_error_sum+=float(np.linalg.norm(low/S-full))
            self.error_den_sum+=float(np.linalg.norm(full))
            mode=self.spec.mode
            if mode in ('integer','integer_nearest_update'):
                delta=qdivide(low,self.D,mode=='integer_nearest_update')
                self.bias_sum+=float(np.mean(delta-low/self.D))/S
                self.master=self.master-delta
            elif mode=='residual_low':self.master=self.master-low
            elif mode=='fp_master_intgrad':self.master=self.master-low/(self.D*S)
            elif mode in ('residual_high','residual_high_nearest'):self.master=self.master-high
            else:self.master=self.master-full/self.D
        checked(self.real_master())
        after=self.qweights()
        if not self.coarse_only:
            units=self.canonical_master()
            if np.any(np.abs(units)>=2**50):raise OverflowError('fine-master exactness bound')
            self.residue_nonzero+=int(np.count_nonzero(units%self.factor))
            self.residue_count+=len(units)
        self.state_zero+=int(np.array_equal(old_master,self.master))
        self.coarse_zero+=int(np.array_equal(qw,after))
        self.updates+=1
        if self.updates%c.target_every==0:self.target=self.master.copy()

    def diagnostics(self):
        return {'updates':self.updates,'interactions':self.interactions,
            'master_zero_update_fraction':self.state_zero/self.updates if self.updates else None,
            'coarse_zero_update_fraction':self.coarse_zero/self.updates if self.updates else None,
            'low_gradient_zero_given_high_nonzero':self.low_lost/self.high_nonzero if self.high_nonzero else None,
            'relative_low_gradient_error':self.low_error_sum/self.error_den_sum if self.error_den_sum else None,
            'mean_final_update_rounding_bias':self.bias_sum/self.updates if self.updates else None,
            'residue_occupancy':self.residue_nonzero/self.residue_count if self.residue_count else None,
            'master_scale':self.spec.scale if self.coarse_only else self.fine_scale,
            'mode':self.spec.mode}


class FloatLearner:
    def __init__(self,spec,initial=None):
        self.spec=spec;self.cfg=spec.cfg();self.inner=ref.Learner(self.cfg)
        if initial is not None:
            self.inner.weights=np.asarray(initial).copy();self.inner.target=np.asarray(initial).copy()

    def qvalues(self,state,deployment='native'):
        if deployment=='native':return ref.forward(self.inner.weights,state,self.cfg)[0]
        S=self.spec.scale;qw=quantize(self.inner.weights,S,False)
        q,_=net_batch(qw,np.asarray(state)[None,:],self.cfg.hidden,S)
        return q[0]/S

    def observe(self,*args):self.inner.observe(*args)
    def real_master(self):return self.inner.weights.copy()
    def diagnostics(self):return self.inner.diagnostics()


def create(spec,initial=None):
    return FloatLearner(spec,initial) if spec.mode in ('fp','fp_grid') else MixedLearner(spec,initial)


def parameters(base,spec):
    if spec.mode=='fp':return np.array(base),None
    if spec.mode=='fp_grid':return np.floor(base*spec.scale)/spec.scale,None
    return np.floor(base*spec.scale).astype(np.int64),spec.scale


def evaluation(learner,spec,base,deployment='native'):
    state,done=env_init(spec.kind),False
    actions=[];ret=0.;success=False
    for _ in range(6):
        q=learner.qvalues(state,deployment)
        a=4 if done else int(np.argmax(q))
        successor,reward,terminal=env_step(spec.kind,state,a,base,None,done)
        success|=spec.kind=='browser' and not done and terminal and reward>0
        ret+=float(reward);actions.append(a);state,done=successor,terminal
    return {'return':ret,'success':bool(success) if spec.kind=='browser' else None,'actions':actions}


def safe_export(learner,spec,base,deployment):
    try:return evaluation(learner,spec,base,deployment)
    except (OverflowError,ValueError,FloatingPointError) as exc:
        return {'status':'failed','error':f'{type(exc).__name__}: {exc}'}


def _hash_array(hasher,value):
    arr=np.ascontiguousarray(value,dtype='<i8');hasher.update(arr.tobytes())


def train(spec,base,initial=None):
    learner=create(spec,initial);cfg=spec.cfg()
    params,scale=parameters(base,spec);unit=1 if scale is None else scale
    offset=100000*spec.phase
    draws=np.random.default_rng(10000+spec.seed+offset).integers(0,[16,4,4],size=(spec.episodes,6,3))
    quantiles=np.random.default_rng(30000+spec.seed+offset).random((spec.episodes,6,8))
    quantiles[:,:,0]=(draws[:,:,2]+quantiles[:,:,0])/4
    quantiles=np.minimum(quantiles,np.nextafter(1.,0.))
    initial_views={d:safe_export(learner,spec,base,d) for d in ('native','deploy','floor')}
    curves=[];checkpoints=[];successes=0
    trajectory=hashlib.sha256();state_chain=hashlib.sha256()
    try:
        for episode in range(spec.episodes):
            epsilon=cfg.eps_start if cfg.eps_end is None else int(round(cfg.eps_start+(cfg.eps_end-cfg.eps_start)*min(episode/999,1)))
            state,done,ret=env_init(spec.kind),False,0.
            for t in range(6):
                q=learner.qvalues(state)
                action=4 if done else int(draws[episode,t,1]) if draws[episode,t,0]<epsilon else int(np.argmax(q))
                successor,reward,terminal=env_step(spec.kind,state,action,params,scale,done)
                if not done:
                    learner.observe(state,action,reward,successor,terminal,quantiles[episode,t])
                    successes+=int(spec.kind=='browser' and terminal and reward>0)
                _hash_array(trajectory,[episode,t,*state,action,int(round(float(reward)*4096/unit)),*successor,int(terminal)])
                if isinstance(learner,MixedLearner) and spec.mode!='qat_weights':
                    _hash_array(state_chain,learner.canonical_master())
                    _hash_array(state_chain,learner.canonical_master(True))
                    _hash_array(state_chain,learner.buffer)
                    _hash_array(state_chain,[learner.interactions,learner.updates])
                ret+=float(reward)/unit;state,done=successor,terminal
            curves.append(ret)
            if episode+1 in spec.checkpoints:
                native=evaluation(learner,spec,base,'native')
                deployed=safe_export(learner,spec,base,'deploy')
                floor=safe_export(learner,spec,base,'floor')
                checkpoints.append({'episode':episode+1,'native':native,'deployed':deployed,'floor_export':floor})
        record={'status':'completed','spec':asdict(spec),'checkpoints':checkpoints,'initial_views':initial_views,
            'final_native':evaluation(learner,spec,base,'native'),
            'final_deployed':safe_export(learner,spec,base,'deploy'),
            'final_floor_export':safe_export(learner,spec,base,'floor'),
            'training_successes':successes if spec.kind=='browser' else None,
            'diagnostics':learner.diagnostics(),'trajectory_sha256':trajectory.hexdigest(),
            'learning_state_sha256':state_chain.hexdigest() if isinstance(learner,MixedLearner) and spec.mode!='qat_weights' else None,
            'episode_returns':curves}
    except (OverflowError,ValueError,FloatingPointError,ArithmeticError) as exc:
        record={'status':'failed','spec':asdict(spec),'error':f'{type(exc).__name__}: {exc}',
            'completed_episodes':len(curves),'checkpoints':checkpoints,'episode_returns':curves,'diagnostics':learner.diagnostics()}
    return record,learner.real_master()
