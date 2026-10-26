#!/usr/bin/env python3
"""Build isolated profile guests/hosts with an already configured toolchain.
No downloads or environment reconfiguration are performed.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True
from checksource import ROOT, check, require

def main():
    check()
    paths = [ROOT/x for x in ('build','bin','elf-control','elf-instrumented')]
    require(not any(p.exists() for p in paths), 'build/output path exists; refuse overwrite')
    for p in paths: p.mkdir()
    (ROOT/'tmp').mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(TMPDIR=str(ROOT/'tmp'), GOCACHE=str(ROOT/'build/go-cache'),
               GOPATH=str(ROOT/'build/go-path'), CARGO_NET_OFFLINE='true')
    commands = []
    def run(name, command, target):
        local = dict(env, CARGO_TARGET_DIR=str(target))
        print(json.dumps({'building':name,'command':command}), flush=True)
        with (ROOT/'build'/(name+'.stdout.log')).open('x') as out, (ROOT/'build'/(name+'.stderr.log')).open('x') as err:
            result = subprocess.run(command, cwd=ROOT, env=local, stdout=out, stderr=err)
        commands.append({'name':name,'cwd':str(ROOT),'command':command,'returncode':result.returncode,'target':str(target)})
        with (ROOT/'build'/(name+'.command.json')).open('x') as f: json.dump(commands[-1],f,indent=2)
        require(result.returncode == 0, f'{name} failed; retained logs, no automatic retry or lock update')
    for variant in ('control','instrumented'):
        features = ['--features','markers'] if variant == 'instrumented' else []
        run('guest-'+variant, ['cargo','prove','build','-p','zkdrl-profile-dqn-guest','--locked',
            '--output-directory',str(ROOT/('elf-'+variant)),*features], ROOT/'build'/('guest-'+variant))
        target = ROOT/'build'/('host-'+variant)
        run('host-'+variant, ['cargo','build','--release','--offline','--locked','--manifest-path',str(ROOT/'host/Cargo.toml'),*features], target)
        with (ROOT/'bin'/('host-'+variant)).open('xb') as f:
            f.write((target/'release/zkdrl-profile-host').read_bytes())
        (ROOT/'bin'/('host-'+variant)).chmod(0o755)
    check()

if __name__ == '__main__':
    main()
