#!/usr/bin/env python3
"""Write a sorted SHA-256 manifest for the review package."""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {'.git', '__pycache__', 'dist', 'runs', 'bin', '.build'}
SKIP_FILES = {'MANIFEST.sha256'}
SKIP_PREFIX = ('dqn/elf', 'a2c/elf', 'cpu-profile/build/', 'cpu-profile/bin/',
               'cpu-profile/elf-', 'cpu-profile/results', 'cpu-profile/tmp/')


def include(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
        return False
    if rel in SKIP_FILES or any(rel.startswith(prefix) for prefix in SKIP_PREFIX):
        return False
    return path.is_file()


def main():
    rows = []
    for path in sorted(p for p in ROOT.rglob('*') if include(p)):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f'{digest}  {path.relative_to(ROOT).as_posix()}')
    (ROOT / 'MANIFEST.sha256').write_text('\n'.join(rows) + '\n')
    print(f'wrote MANIFEST.sha256: {len(rows)} files')


if __name__ == '__main__':
    main()
