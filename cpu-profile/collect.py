#!/usr/bin/env python3
"""Parent-run CPU-only capture driver; never builds or proves; no overwrites."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.dont_write_bytecode = True
from checksource import ROOT, check, require, sha
from report import RESULTS, ELVES, HOSTS, VARIANTS, group, report, save

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('--launcher', type=Path, help='optional parent-owned CPU launcher, invoked as LAUNCHER HOST ARGS')
    args = parser.parse_args()
    source = check()
    require(not RESULTS.exists(), 'results already exists: preserve it; do not overwrite or resume mixed captures')
    input_path, reference_path = args.input.resolve(), args.reference.resolve()
    input_raw, reference_raw = input_path.read_bytes(), reference_path.read_bytes()
    full = json.loads(input_raw)
    reference = json.loads(reference_raw)
    require(full['schema'] == 202 and full['kind'] == 1 and full['train'] == 1 and full['slots'] == 24000, 'full schema202 exposure 4000 episodes')
    require(all(k in reference for k in ('journal','final_state','coarse_export')), 'expected raw Python reference Output fields')
    artifacts = [input_path, reference_path, *ELVES.values(), *HOSTS.values()]
    if args.launcher:
        args.launcher = args.launcher.resolve()
        artifacts.append(args.launcher)
    provenance = {'input':str(input_path), 'reference':str(reference_path), 'sha256':{str(p):sha(p) for p in artifacts}}
    RESULTS.mkdir()
    save('source-check.json', source)
    save('provenance.json', provenance)
    with (RESULTS/'input-full.json').open('xb') as f: f.write(input_raw)
    with (RESULTS/'reference.json').open('xb') as f: f.write(reference_raw)
    save('input-short.json', dict(full, slots=48))
    env = os.environ.copy()
    # Process-local variables only; no sourced old environment or outside cache writes.
    env['TMPDIR'] = str(ROOT/'tmp')
    env['RUST_LOG'] = 'error'
    (ROOT/'tmp').mkdir(exist_ok=True)
    def run(name, host, operation, variant, inp):
        command = [str(host), operation, variant, str(inp)]
        if args.launcher: command.insert(0, str(args.launcher))
        print(json.dumps({'starting':name,'command':command}), flush=True)
        started = time.monotonic()
        with (RESULTS/(name+'.stdout.log')).open('x') as out, (RESULTS/(name+'.stderr.log')).open('x') as err:
            process = subprocess.run(command, stdout=out, stderr=err, env=env, cwd=ROOT)
        save(name+'.command.json', {'command':command,'returncode':process.returncode,'wall_seconds':time.monotonic()-started,'cryptographic_proof':False})
        require(process.returncode == 0, f'{name} failed; raw logs retained')
    for prefix in ('short','full'):
        inp = RESULTS/('input-'+prefix+'.json')
        for variant,host in HOSTS.items(): run(f'{prefix}-native-{variant}',host,'native',variant,inp)
        for variant in VARIANTS: run(f'{prefix}-{variant}',HOSTS['instrumented'],'execute',variant,inp)
        group(prefix, inp, reference if prefix == 'full' else None)
    report()

if __name__ == '__main__':
    main()
