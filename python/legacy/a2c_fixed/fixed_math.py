"""Integer-only A2C arithmetic. Python ints in NumPy object arrays, never int64 math.

Forward quantization/LUT/mass rounding use declared surrogate (STE) backward
formulas. This is NOT the exact gradient of the discrete implemented policy.
See SPEC.md. All // are floor, including for negative values.
"""
from dataclasses import dataclass
from functools import lru_cache
from numbers import Integral
from pathlib import Path
import hashlib
import json
import struct
import numpy as np

G=1<<40
P=1<<32
RAW_LIMIT=1<<127
ROOT=Path(__file__).resolve().parent


def ints(value, name='integer array', shape=None):
    a=np.asarray(value,dtype=object)
    if shape is not None and a.shape!=shape: raise ValueError(f'{name}: shape {shape} required')
    if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,Integral) for v in a.flat):
        raise ValueError(f'{name}: integers, not floats/bools, required')
    return np.array([int(v) for v in a.flat],dtype=object).reshape(a.shape)


def bound(a, limit, name, audit=None):
    peak=max((abs(int(v)) for v in np.asarray(a,dtype=object).flat),default=0)
    if peak>=limit: raise OverflowError(f'{name}: abs {peak} >= {limit}')
    if audit is not None: audit[name]=max(audit.get(name,0),peak)
    return a


def raw(a,name,audit=None): return bound(a,RAW_LIMIT,'raw_'+name,audit)


@lru_cache(None)
def table():
    data=(ROOT/'exp_q48.json').read_bytes()
    info=json.loads((ROOT/'EXP_TABLE.json').read_text())
    if hashlib.sha256(data).hexdigest()!=info['sha256']: raise ValueError('exp table checksum')
    values=json.loads(data)
    if len(values)!=8193 or values[0]!=1<<48 or any(type(v)!=int or v<0 for v in values): raise ValueError('invalid table')
    return tuple(values)


def exp_weight(shifted, bits):
    s=1<<bits
    if shifted>0: raise ValueError('exp requires nonpositive shifted logit')
    q=-int(shifted)
    if q>32*s:return 0
    if q==32*s:return table()[-1]
    step=s//256; k,r=divmod(q,step); t=table()
    return ((step-r)*t[k]+r*t[k+1])//step


def masses(logits,bits,audit=None):
    z=ints(logits,'logits')
    if z.ndim!=2 or z.shape[1]!=4:raise ValueError('logits shape (N,4)')
    bound(z,1024*(1<<bits),'logits',audit)
    out=[]
    for row in z:
        peak=max(row); weights=[exp_weight(v-peak,bits) for v in row]; total=sum(weights)
        if total<=0:raise RuntimeError('subtract-max weight lost')
        numer=[P*w for w in weights]; raw(numer,'mass_numerator',audit)
        floor=[v//total for v in numer]; rem=[v%total for v in numer]; left=P-sum(floor)
        if not 0<=left<4:raise RuntimeError('mass allocation')
        for i in sorted(range(4),key=lambda j:(-rem[j],j))[:left]: floor[i]+=1
        if sum(floor)!=P or any(w==0 and m!=0 for w,m in zip(weights,floor)):raise RuntimeError('mass invariant')
        out.append(floor)
    return np.array(out,dtype=object).reshape((-1,4))


def categorical(mass,draw):
    m=ints(mass,'mass',(4,))
    if any(v<0 or v>P for v in m) or sum(m)!=P:raise ValueError('invalid probability masses')
    if isinstance(draw,(bool,np.bool_)) or not isinstance(draw,Integral) or not 0<=draw<P:raise ValueError('draw must be uint32')
    cumulative=0
    for i,v in enumerate(m):
        cumulative+=v
        if draw<cumulative:return i
    raise RuntimeError('unreachable categorical fallthrough')


def unpack(params,out):
    p=ints(params,'parameters',(80+17*out,))
    return p[:64].reshape(16,4),p[64:80],p[80:80+16*out].reshape(out,16),p[80+16*out:]


@dataclass(frozen=True)
class Cache:
    states: np.ndarray
    pre: np.ndarray
    hidden: np.ndarray
    w2: np.ndarray


def forward(params,states,out,bits,audit=None):
    s=1<<bits
    x=ints(states,'states')
    if x.ndim!=2 or x.shape[1]!=4 or any(v<0 or v>6 for v in x.flat): raise ValueError('states must be (N,4) entries0..6')
    w1,b1,w2,b2=unpack(params,out)
    bound(params,256*s,'forward_params',audit)
    pre=raw(x@w1.T+b1,'pre',audit)
    bound(pre,1024*s,'preactivation',audit)
    hidden=np.maximum(pre,0)
    product=raw(hidden@w2.T,'output_dot',audit)
    y=product//s+b2
    bound(y,1024*s,'outputs',audit)
    copies=[x.copy(),pre.copy(),hidden.copy(),w2.copy()]
    for a in copies:a.flags.writeable=False
    return y,Cache(*copies)


def backward(cache,upstream,out,bits,audit=None):
    s=1<<bits
    dy=ints(upstream,'upstream',(len(cache.states),out))
    bound(dy,65536*G,'upstream',audit)
    hidden_raw=raw(dy@cache.w2,'hidden_vjp_dot',audit)
    dh=(hidden_raw//s)*(cache.pre>0)
    bound(dh,65536*G,'hidden_vjp',audit)
    grad1=raw(dh.T@cache.states,'gradient1_dot',audit)
    grad2=raw(dy.T@cache.hidden,'gradient2_dot',audit)//s
    grad=np.concatenate((grad1.ravel(),dh.sum(axis=0),grad2.ravel(),dy.sum(axis=0)))
    return bound(grad,65536*G,'gradient',audit)


def finite_returns(rewards,terminals,trace_continue,bits,audit=None):
    r=ints(rewards,'rewards'); term=ints(terminals,'terminals',r.shape); trace=ints(trace_continue,'trace',r.shape)
    bound(r,4*(1<<bits),'rewards',audit)
    if r.ndim!=1 or not len(r):raise ValueError('nonempty rewards vector')
    if any(v not in (0,1) for v in term) or any(v not in (0,1) for v in trace):raise ValueError('binary masks required')
    if any(t and c for t,c in zip(term,trace)) or trace[-1]!=0:raise ValueError('terminal/final trace must cut')
    following=0; result=[]
    for i in range(len(r)-1,-1,-1):
        following=int(r[i])*(G//(1<<bits))+(7*following//8 if trace[i] else 0)
        raw([following],'return',audit)
        result.append(following)
    return bound(np.array(result[::-1],dtype=object),65536*G,'targets',audit)


def actor_upstream(logits,mass,actions,advantages,bits,audit=None):
    z=ints(logits,'logits')
    if z.ndim!=2 or z.shape[1]!=4:raise ValueError('logits shape (N,4)')
    bound(z,1024*(1<<bits),'logits',audit)
    n=len(z)
    p=ints(mass,'mass',(n,4)); a=ints(actions,'actions',(n,)); adv=ints(advantages,'advantages',(n,))
    if any(v<0 or v>=4 for v in a):raise ValueError('actions0..3 required')
    if any(any(v<0 or v>P for v in row) or sum(row)!=P for row in p):raise ValueError('invalid masses')
    bound(adv,65536*G,'advantages',audit)
    shifted=z-np.max(z,axis=1,keepdims=True)
    moment=raw((p*shifted).sum(axis=1),'logit_moment',audit)
    one=np.eye(4,dtype=object)[np.asarray(a,dtype=np.int64)]
    policy_num=raw(adv[:,None]*(p-one*P),'policy_numerator',audit)
    entropy_num=raw(p*(shifted*P-moment[:,None])*(G//(1<<bits)),'entropy_numerator',audit)
    dy=policy_num//P+entropy_num//(128*P*P)
    return bound(dy,65536*G,'actor_upstream',audit)


def shadow_gradients(actor_cache,critic_cache,logits,mass,actions,advantages,targets,bits):
    """Real-valued shadow on EXACT fixed forward, no new softmax/forward pass.

    Unrounded analytic dy plus real matrix VJP; discrepancies isolate declared
    backward floors, NOT forward/LUT/master quantization error.
    """
    s=1<<bits
    z=np.asarray(logits,dtype=np.float64)/s; z-=z.max(axis=1,keepdims=True)
    p=np.asarray(mass,dtype=np.float64)/P
    adv=np.asarray(advantages,dtype=np.float64)/G
    a=np.asarray(actions,dtype=np.int64)
    dy=adv[:,None]*(p-np.eye(4)[a])+p*(z-(p*z).sum(axis=1,keepdims=True))/128
    # Critic residual is exactly -advantage: targets/advantages detached.
    dv=-adv[:,None]
    def vjp(cache,up):
        h=np.asarray(cache.hidden,dtype=np.float64)/s
        w2=np.asarray(cache.w2,dtype=np.float64)/s
        x=np.asarray(cache.states,dtype=np.float64)
        dh=(up@w2)*(cache.pre>0)
        return np.concatenate(((dh.T@x).ravel(),dh.sum(axis=0),(up.T@h).ravel(),up.sum(axis=0)))
    return vjp(actor_cache,dy),vjp(critic_cache,dv)


def candidates(actor_master,critic_master,actor_gradient,critic_gradient,denominator,audit=None):
    if isinstance(denominator,bool) or not isinstance(denominator,Integral) or denominator<=0:raise ValueError('positive integer denominator')
    am=ints(actor_master,'actor master',(148,)); cm=ints(critic_master,'critic master',(97,))
    ga=ints(actor_gradient,'actor gradient',(148,)); gc=ints(critic_gradient,'critic gradient',(97,))
    bound(am,256*G,'master_params',audit); bound(cm,256*G,'master_params',audit)
    bound(ga,65536*G,'gradient',audit); bound(gc,65536*G,'gradient',audit)
    ac=raw(am-ga//(int(denominator)*32),'actor_candidate',audit)
    cc=raw(cm-gc//(int(denominator)*64),'critic_candidate',audit)
    bound(ac,256*G,'master_params',audit); bound(cc,256*G,'master_params',audit)
    return ac,cc


def initial_masters(seed):
    if type(seed)!=int or not 0<=seed<1<<64:raise ValueError('seed uint64')
    def init(domain,out,std):
        rng=np.random.default_rng(np.random.SeedSequence([domain,seed]))
        p=np.concatenate((rng.normal(0,.20,64),np.full(16,.25),rng.normal(0,std,16*out),np.zeros(out)))
        # Binary scale multiplication is exact for these float64 fixtures.
        return np.array([int(np.floor(v*G)) for v in p],dtype=object)
    return init(1101,4,.01),init(1102,1,.20)


class FixedModel:
    def __init__(self,seed=0,bits=16):
        if bits not in (16,20):raise ValueError('registered forward precision required')
        self.bits=bits; self.actor_master,self.critic_master=initial_masters(seed); self.update_count=0

    def snapshot(self):
        ratio=G//(1<<self.bits)
        return self.actor_master//ratio,self.critic_master//ratio

    def fingerprint(self):
        h=hashlib.sha256(b'zkRL/A2C-fixed/state/v1\0')
        h.update(struct.pack('<IQ',self.bits,self.update_count))
        for vector in (self.actor_master,self.critic_master):
            # guards ensure master representation fits signed64; encoding is explicit.
            bound(vector,256*G,'master serialization')
            for v in vector:h.update(int(v).to_bytes(8,'little',signed=True))
        return h.hexdigest()

    def commit(self,ga,gc,denominator,expected_hash,audit=None):
        if expected_hash!=self.fingerprint():raise ValueError('stale/off-policy model snapshot')
        ac,cc=candidates(self.actor_master,self.critic_master,ga,gc,denominator,audit)
        ratio=G//(1<<self.bits)
        bound(ac//ratio,256*(1<<self.bits),'forward_candidate_params',audit)
        bound(cc//ratio,256*(1<<self.bits),'forward_candidate_params',audit)
        self.actor_master,self.critic_master=ac,cc
        self.update_count+=1
