"""Evaluate the IMPLEMENTED integer policy. Diagnostics only; not training."""
import numpy as np
from .float_bridge import Evaluator
from .fixed_math import forward, masses, P


class FixedEvaluator(Evaluator):
    def model(self,model):
        actor,_=model.snapshot()
        logits,_=forward(actor,np.asarray(self.state_array,dtype=object).astype(np.int64),4,model.bits)
        mass=masses(logits,model.bits)
        p=np.asarray(mass,dtype=np.float64)/P
        # Keep historical lowest-index argmax of implemented logits.
        greedy=np.eye(4)[np.argmax(np.asarray(logits,dtype=np.int64),axis=1)]
        row=p[self.index[self.initial]]; nonzero=row>0
        return dict(stochastic=self.probabilities(p),greedy=self.probabilities(greedy),
                    initial_state_entropy=float(-np.sum(row[nonzero]*np.log(row[nonzero]))),
                    zero_probability_cells=int(np.count_nonzero(p==0)),
                    deterministic_state_rows=int(np.count_nonzero(np.max(p,axis=1)==1)),
                    evaluation_probability_cells=int(p.size))
