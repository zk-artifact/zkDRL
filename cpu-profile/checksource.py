#!/usr/bin/env python3
"""Static structural relation only: not formal refinement or compiled equivalence."""
import hashlib
import json
from pathlib import Path
import sys
import tomllib
sys.dont_write_bytecode = True
from source_layout import instrument, strip_markers, MARKERS
ROOT = Path(__file__).resolve().parent

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def check():
    original = ROOT.parent / 'dqn'
    src = (original/'common/src/lib.rs').read_text()
    copied = (ROOT/'common/src/lib.rs').read_text()
    require(copied == instrument(src), 'source differs from exact marker insertion recipe')
    require(strip_markers(copied) == src, 'marker stripping/output unwrapping not byte-identical')
    require((ROOT/'common/src/markers.rs').read_text() == MARKERS, 'marker macro body changed')
    require((ROOT/'common/src/tests.rs').read_bytes() == (original/'common/src/tests.rs').read_bytes(), 'copied tests differ')
    require((ROOT/'guest/src/main.rs').read_text().replace('zkdrl_profile_dqn_common', 'zkdrl_ctc_dqn') == (original/'guest/src/main.rs').read_text(), 'guest differs beyond crate rename')
    base = tomllib.loads((original/'Cargo.toml').read_text())
    workspace = tomllib.loads((ROOT/'Cargo.toml').read_text())
    require(workspace['patch'] == base['patch'] and workspace['profile'] == base['profile'], 'guest patch/release settings differ')
    common = tomllib.loads((ROOT/'common/Cargo.toml').read_text())
    require(common['dependencies']['zkdrl-ctc-protocol']['path'] == '../../protocol', 'protocol must stay frozen/shared')
    for part in ['common', 'guest']:
        current = tomllib.loads((ROOT/part/'Cargo.toml').read_text())
        frozen = tomllib.loads((original/part/'Cargo.toml').read_text())
        normalized = json.loads(json.dumps(current).replace('zkdrl-profile-dqn-guest', 'zkdrl-ctc-dqn-guest').replace('zkdrl-profile-dqn-common', 'zkdrl-ctc-dqn'))
        expected = [] if part == 'common' else ['zkdrl-ctc-dqn/markers']
        require(normalized['features'].pop('markers') == expected, 'guest marker feature wiring')
        require(normalized == frozen, f'{part} manifest differs beyond names/markers')
    host = tomllib.loads((ROOT/'host/Cargo.toml').read_text())
    require(host['dependencies']['sp1-sdk'] == {'version': '=6.9.0', 'default-features': False, 'features': ['profiling']}, 'portable SDK feature/pin required')
    require(host['features']['markers'] == ['zkdrl-profile-dqn-common/markers'], 'host markers separate from SDK profiling')
    versions = {}
    for lock in ['Cargo.lock', 'host/Cargo.lock']:
        packages = tomllib.loads((ROOT/lock).read_text())['package']
        versions[lock] = sorted({p['version'] for p in packages if p['name'].startswith(('sp1-', 'slop-'))})
        if lock == 'host/Cargo.lock':
            require(versions[lock] == ['6.9.0'], 'host SDK transitive versions must be 6.9.0')
    # Cargo may sort renamed package entries differently when building. Compare
    # exact semantic lock data, normalizing only package names and list order.
    actual_lock=tomllib.loads((ROOT/'Cargo.lock').read_text().replace('zkdrl-profile-dqn-guest','zkdrl-ctc-dqn-guest').replace('zkdrl-profile-dqn-common','zkdrl-ctc-dqn'))
    expected_lock=tomllib.loads((original/'Cargo.lock').read_text())
    for document in (actual_lock,expected_lock):
        document['package']=sorted(document['package'],key=lambda p:(p['name'],p['version'],p.get('source','')))
    require(actual_lock==expected_lock,'guest semantic lock differs beyond local package names/order')
    files = [original/'common/src/lib.rs', original/'guest/src/main.rs', original/'Cargo.toml', original/'Cargo.lock', ROOT.parent/'protocol/src/lib.rs', ROOT.parent/'protocol/Cargo.toml']
    files += [ROOT/p for p in ['Cargo.toml','Cargo.lock','common/Cargo.toml','common/src/lib.rs','common/src/markers.rs','guest/Cargo.toml','guest/src/main.rs','host/Cargo.toml','host/Cargo.lock','host/src/main.rs','source_layout.py','checksource.py','report.py','collect.py','build.py']]
    return {'status': 'passed', 'claim': 'byte-exact source after stripping prescribed markers and unwrapping Output; not formal refinement', 'sp1_slop_versions': versions, 'sha256': {str(p): sha(p) for p in files}}

if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
