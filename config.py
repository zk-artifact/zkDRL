"""Shared paths and small I/O helpers for the local campaign harness.

Override the output root with ZKDRL_OUT if the default ``runs/full-023-unified``
is not convenient.  The harness never writes outside ROOT and OUT.
"""
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parent
OUT = Path(os.environ.get('ZKDRL_OUT', ROOT / 'runs' / 'full-023-unified')).resolve()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def binary(program):
    return ROOT / 'bin' / ('host-dqn-h16-c64' if program == 'a2c' else 'host-' + program)
