#!/usr/bin/env python3
"""Run the Python reference on a tiny public fixture (no Rust, no proof)."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))

import a2c_reference
import dqn_reference


def main():
    for name, ref in (('dqn', dqn_reference), ('a2c', a2c_reference)):
        inp = ref.fixture(kind=1, seed=5000, episodes=12, random_seed=77)
        inp['challenge'] = list(range(1, 33))  # test-only issued challenge
        out = ref.run(inp, True)
        journal = out['journal']
        print(json.dumps({
            'algorithm': name,
            'episodes': 12,
            'journal': journal,
            'journal_bytes': len(ref.journal_bytes(journal)),
            'final_state_keys': sorted(out['final_state']),
        }, sort_keys=True))


if __name__ == '__main__':
    main()
