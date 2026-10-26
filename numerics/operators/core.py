"""Exact Python-int implementation. All // operations are signed floor."""
from functools import lru_cache
from pathlib import Path
import hashlib
import json

Q = 1 << 16
E = 1 << 48
P = 1 << 32
LOGIT_LIMIT = 1024 * Q
MAX_ACTIONS = 1 << 16
ROOT = Path(__file__).resolve().parent
VARIANTS = ("O0", "O1", "O2")


def _integer(value, name):
    if type(value) is not int:
        raise TypeError(f"{name} must be a Python int (no bool/float/NumPy scalar)")
    return value


def _variant(variant):
    if type(variant) is not str or variant not in VARIANTS:
        raise ValueError("variant must be O0, O1, or O2")


def _vector(values, name):
    if type(values) not in (list, tuple):
        raise TypeError(f"{name} must be a list or tuple")
    if not 1 <= len(values) <= MAX_ACTIONS:
        raise ValueError(f"{name} length must be 1..{MAX_ACTIONS}")
    return [_integer(v, name) for v in values]


@lru_cache(None)
def _table(variant):
    _variant(variant)
    data = (ROOT / f"{variant}.json").read_bytes()
    manifest = json.loads((ROOT / "contract.json").read_text())
    if hashlib.sha256(data).hexdigest() != manifest["table_sha256"][variant]:
        raise ValueError(f"{variant} table hash mismatch")
    values = json.loads(data)
    if variant == "O2":
        if (len(values) != 256 or any(type(row) is not list or len(row) != 4
                or any(type(v) is not int or not 0 <= v <= E for v in row)
                for row in values)):
            raise ValueError("invalid O2 coefficients")
        return tuple(tuple(row) for row in values)
    if (len(values) != {"O0": 8193, "O1": 2049}[variant]
            or values[0] != E or any(type(v) is not int or not 0 <= v <= E for v in values)
            or any(a < b for a, b in zip(values, values[1:]))):
        raise ValueError("invalid LUT")
    return tuple(values)


def table_hashes():
    """SHA256 of exact frozen JSON bytes; validate tables against contract."""
    for v in VARIANTS:
        _table(v)
    return {v: hashlib.sha256((ROOT / f"{v}.json").read_bytes()).hexdigest()
            for v in VARIANTS}


def contract_hash():
    return hashlib.sha256((ROOT / "contract.json").read_bytes()).hexdigest()


def exp_weight(x_q16, variant="O0"):
    """Signed64 nonpositive Q16 -> unclamped nonnegative Q48 weight."""
    x = _integer(x_q16, "x_q16")
    _variant(variant)
    if not -(1 << 63) <= x <= 0:
        raise ValueError("x_q16 must be signed64 and nonpositive")
    if x < -32 * Q:
        return 0
    if x == 0:
        return E
    table = _table(variant)
    if variant != "O2":
        step = 256 if variant == "O0" else 1024
        k, r = divmod(-x, step)
        if k == len(table) - 1:
            return table[k]
        return ((step - r) * table[k] + r * table[k + 1]) // step
    j = (x + 32 * Q) // 8192
    t = x - (-32 * Q + j * 8192 + 4096)
    a0, a1, a2, a3 = table[j]
    h = a3
    for coefficient in (a2, a1, a0):
        h = (h * t) // Q + coefficient
    return h


def softmax_masses(logits, valid_mask, variant="O0"):
    """One variable-A row -> masses summing to P, stable valid-only ties."""
    z = _vector(logits, "logits")
    mask = _vector(valid_mask, "valid_mask")
    _variant(variant)
    if len(z) != len(mask) or any(v not in (0, 1) for v in mask):
        raise ValueError("matching integer 0/1 mask required")
    if any(not -LOGIT_LIMIT < v < LOGIT_LIMIT for v in z):
        raise ValueError("all logits require abs(logit) < 1024 * 2^16")
    valid = [i for i, v in enumerate(mask) if v]
    if not valid:
        raise ValueError("at least one valid action required")
    peak = max(z[i] for i in valid)
    weights = {i: exp_weight(z[i] - peak, variant) for i in valid}
    if any(w < 0 for w in weights.values()):
        raise ArithmeticError("negative operator weight; no clamp permitted")
    total = sum(weights.values())
    if total <= 0:
        raise ArithmeticError("nonpositive total")
    masses = [0] * len(z)
    remainder = {}
    for i in valid:
        masses[i], remainder[i] = divmod(P * weights[i], total)
    left = P - sum(masses)
    if not 0 <= left < len(valid):
        raise ArithmeticError("largest-remainder residual invariant")
    for i in sorted(valid, key=lambda i: (-remainder[i], i))[:left]:
        masses[i] += 1
    return masses


def categorical(masses, draw):
    """Strict draw < CDF; equality goes right, zero-mass entries skipped."""
    m = _vector(masses, "masses")
    u = _integer(draw, "draw")
    if not 0 <= u < P:
        raise ValueError("draw must be uint32")
    if any(not 0 <= v <= P for v in m) or sum(m) != P:
        raise ValueError("masses must be in [0,P] and sum to P")
    cumulative = 0
    for i, value in enumerate(m):
        cumulative += value
        if u < cumulative:
            return i
    raise ArithmeticError("unreachable CDF fallthrough")
