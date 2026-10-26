"""Local, package-isolated copies of the legacy float/evaluation dependencies."""
from ..a2c_float.core.a2c import ActorCritic as FloatModel
from ..a2c_float.followup_trainer import collect as float_collect, update_rollout as float_update, state_hash as float_hash
from ..a2c_float.evaluate import Evaluator, parameters
from ..model import env_init, env_step
