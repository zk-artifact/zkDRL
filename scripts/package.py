#!/usr/bin/env python3
"""Build a deterministic zip of this artifact with normalized metadata."""
from pathlib import Path
import hashlib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'dist' / 'zkdrl-anonymous.zip'
SKIP_DIRS = {'.git', '__pycache__', 'dist', 'runs', 'bin', '.build'}
SKIP_FILES = set()
SKIP_PREFIX = ('dqn/elf', 'a2c/elf', 'cpu-profile/build/', 'cpu-profile/bin/',
               'cpu-profile/elf-', 'cpu-profile/results', 'cpu-profile/tmp/')
STAMP = (2026, 1, 1, 0, 0, 0)


def include(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts):
        return False
    if rel in SKIP_FILES or any(rel.startswith(prefix) for prefix in SKIP_PREFIX):
        return False
    return path.is_file()


def main():
    if not (ROOT / 'MANIFEST.sha256').exists():
        raise SystemExit('run scripts/make_manifest.py first')
    files = sorted(p for p in ROOT.rglob('*') if include(p))
    OUT.parent.mkdir(exist_ok=True)
    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in files:
            rel = path.relative_to(ROOT).as_posix()
            info = zipfile.ZipInfo(f'zkdrl-anonymous/{rel}', date_time=STAMP)
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if path.suffix in {'.sh', '.py'} or path.name == 'Makefile' else 0o644) << 16
            zf.writestr(info, path.read_bytes())
    raw = OUT.read_bytes()
    print(f'{OUT.relative_to(ROOT)}: {len(raw)} bytes, {len(files)} files')
    print(f'sha256: {hashlib.sha256(raw).hexdigest()}')


if __name__ == '__main__':
    main()
