"""Independent SHA-256 counter tape for the commit-then-challenge protocol.

Fixture seeds are PUBLIC test data, independent of initializer metadata. They
are not private entropy and establish no fairness claim. A real auditor replaces
fixture challenge zeros only AFTER registering the configuration commitment.
"""
import hashlib
import json
import struct
from pathlib import Path

DQN, A2C = 1, 2
MASK = (1 << 64) - 1
KEY_DOMAIN = b'zkDRL/ctc/rng-key/v1\0'
BLOCK_DOMAIN = b'zkDRL/ctc/rng-block/v1\0'


def sha(data):
    return hashlib.sha256(data).digest()


def integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name}: integer in [{low}, {high}] required')


def array(value, length, low, high, name):
    if type(value) is not list or len(value) != length:
        raise ValueError(f'{name}: JSON array of length {length} required')
    for v in value:
        integer(v, low, high, name)


def bytes32(value, name):
    if type(value) is bytes and len(value) == 32:
        return value
    array(value, 32, 0, 255, name)
    return bytes(value)


def u32s(values):
    return b''.join(struct.pack('<I', v) for v in values)


def u64s(values):
    return b''.join(struct.pack('<Q', v) for v in values)


def i64s(values):
    return b''.join(struct.pack('<q', int(v)) for v in values)


class CounterRng:
    def __init__(self, algorithm, rng_seed, challenge):
        integer(algorithm, 1, 2, 'algorithm')
        self.key = sha(KEY_DOMAIN + u32s([algorithm])
                       + bytes32(rng_seed, 'rng_seed') + bytes32(challenge, 'challenge'))
        self._position = 0
        self._cached_block = None
        self._block = None

    def position(self):
        return self._position

    def next_u32(self):
        if self._position == MASK:
            raise OverflowError('RNG word cursor overflow')
        j, offset = divmod(self._position, 8)
        if self._cached_block != j:
            self._block = sha(BLOCK_DOMAIN + self.key + u64s([j]))
            self._cached_block = j
        word = struct.unpack_from('<I', self._block, 4 * offset)[0]
        self._position += 1
        return word


def public_rng_seed(algorithm, random_seed=0):
    """Public pairing identifier, deliberately separate from metadata seed."""
    integer(algorithm, 1, 2, 'algorithm')
    integer(random_seed, 0, MASK, 'random_seed')
    return list(sha(b'zkDRL/ctc/PUBLIC-fixture/rng-seed/v1\0'
                    + u32s([algorithm]) + u64s([random_seed])))


def unique_fields(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise ValueError(f'duplicate JSON field: {k}')
        result[k] = v
    return result


def loads(data):
    def invalid(value):
        raise ValueError(f'invalid JSON constant: {value}')
    return json.loads(data, object_pairs_hook=unique_fields, parse_constant=invalid)


def load(path):
    return loads(Path(path).read_text())
