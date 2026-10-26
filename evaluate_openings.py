"""External policy quality after checking opening of the authenticated final digest.

This evaluation is not inside the training proof and is not an optimality proof.
"""
import hashlib
import math
import sys
import numpy as np
from config import ROOT
sys.path.insert(0,str(ROOT/'python'))
import dqn_reference as dqn
import a2c_reference as a2c
from legacy.a2c_fixed.fixed_math import FixedModel
from legacy.a2c_fixed.fixed_eval import FixedEvaluator


def evaluate(case, inp, output, actual_journal):
    state=output['final_state']
    prefix=bytes(actual_journal['config_commit'])+bytes(inp['challenge'])+bytes(inp['final_salt'])
    kind='browser' if case['kind']==0 else 'recommendation'
    params=np.zeros(4) if kind=='browser' else np.array([6,23,11,7])/16
    if case['algorithm']=='dqn':
        bits=7+int(math.log2(inp['scale']))
        target=(np.array(state['target'],dtype=np.int64)>>bits).tolist()
        preimage=(b'zkDRL/ctc/dqn/final/v3\0'+prefix+
            dqn.state_bytes(state,hidden=case['width'],capacity=case['capacity'])+
            dqn.i64s(output['coarse_export'])+dqn.i64s(target))
        model=dqn.q.MixedLearner(dqn.spec(inp,case['width'],case['capacity']))
        model.master=np.array(state['master'],dtype=np.int64)
        result=dqn.q.evaluation(model,model.spec,params)
    else:
        preimage=b'zkDRL/ctc/a2c/final/v2\0'+prefix+a2c.state_bytes(state)
        model=FixedModel(0,16)
        model.actor_master=np.array(state['actor'],dtype=object)
        model.critic_master=np.array(state['critic'],dtype=object)
        result=FixedEvaluator(kind,params,gamma=.875).model(model)
    if list(hashlib.sha256(preimage).digest())!=actual_journal['final_commit']:
        raise ValueError('external model does not open authenticated final state commitment')
    return dict(final_commitment_opening_checked=True,evaluation_inside_proof=False,
        one_predeclared_seed_only=True,evaluation=result)
