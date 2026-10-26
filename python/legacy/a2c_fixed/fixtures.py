"""Public deterministic fixtures. No randomness secrecy or IID guarantee."""
import numpy as np

MASK=(1<<64)-1
DOMAIN=0xA2C0130000000000
PARAMS_S16=(6,23,11,7)


def splitmix64(state):
    state=(state+0x9E3779B97F4A7C15)&MASK
    z=state
    z=((z^(z>>30))*0xBF58476D1CE4E5B9)&MASK
    z=((z^(z>>27))*0x94D049BB133111EB)&MASK
    return state,z^(z>>31)


def tape(seed,episodes=4000):
    if type(seed)!=int or not 0<=seed<=MASK:raise ValueError('seed uint64')
    if type(episodes)!=int or episodes<=0:raise ValueError('positive episodes')
    state=(DOMAIN+seed)&MASK; values=[]
    for _ in range(episodes*6):
        state,z=splitmix64(state); values.append(z>>32)
    return np.array(values,dtype=np.uint64).reshape(episodes,6)


def real_params(kind):
    return np.asarray(reward_params(kind,16),dtype=np.float64)/(1<<16)


def reward_params(kind,bits=16):
    if kind=='browser':return np.zeros(4,dtype=object)
    if kind!='recommendation':raise ValueError(kind)
    return np.array([v*((1<<bits)//16) for v in PARAMS_S16],dtype=object)
