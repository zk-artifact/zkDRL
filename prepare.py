"""Freeze/register complete formal matrix before issuing any experimental challenge."""
import concurrent.futures
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
sys.dont_write_bytecode=True
from config import ROOT,OUT,binary,save,sha
from design import PROGRAMS,core_cases,numerical_jobs,group_id,SEEDS,proof_requests
from auditor import Ledger,last_json
sys.path.insert(0,str(ROOT/'python'))
import dqn_reference as dqn
import a2c_reference as a2c


def kwargs(case):return dict(hidden=case['width'],capacity=case['capacity']) if case['algorithm']=='dqn' else {}
def reference(case):return dqn if case['algorithm']=='dqn' else a2c


def call(args,stem,timeout=1800):
 stem.parent.mkdir(parents=True,exist_ok=True)
 if stem.with_suffix('.stdout').exists():raise FileExistsError(stem)
 with tempfile.TemporaryDirectory(prefix='zkdrl-full023-cold-') as tmp:
  started=time.monotonic()
  with stem.with_suffix('.stdout').open('x') as out,stem.with_suffix('.stderr').open('x') as err:
   p=subprocess.run(list(map(str,args)),stdout=out,stderr=err,env=dict(os.environ,TMPDIR=tmp),timeout=timeout)
  native_helpers=list(Path(tmp).glob('sp1-native-runner-bin-*'))
  save(stem.with_suffix('.metrics.json'),dict(command=list(map(str,args)),returncode=p.returncode,
   elapsed_seconds=time.monotonic()-started,cold_tmpdir=True,native_helper_files=[p.name for p in native_helpers]))
  if p.returncode:raise RuntimeError('command failed; preserve stderr: '+str(stem))
  if native_helpers:raise RuntimeError('portable SDK must not rely on native helper cache')
 return stem.with_suffix('.stdout').read_text()


def register():
 cases=core_cases();jobs=numerical_jobs()
 identities={}
 def identity(program):
  alg=PROGRAMS[program]['algorithm']
  value=last_json(call([binary(program),'identity',alg],OUT/'preparation'/('identity-'+program),600))
  assert value['profile']=={k:PROGRAMS[program][k] for k in ('width','capacity')}
  return program,value
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  for program,value in pool.map(identity,PROGRAMS):identities[program]=value
 save(OUT/'program-identities.json',identities)
 scientific={}
 for p in sorted(ROOT.rglob('*')):
  if p.is_file() and p.suffix in ('.py','.rs','.json','.toml','.lock','.sh') and not any(x in p.parts for x in ('target','payload','__pycache__','bin','cloud','runs','dist','build','tmp','results','elf-control','elf-instrumented')):
   scientific[str(p.relative_to(ROOT))]=sha(p)
 save(OUT/'SOURCE_FREEZE.json',dict(status='frozen_before_challenge',files=scientific,
  numerical_source_sha256=hashlib.sha256(json.dumps(scientific,sort_keys=True).encode()).hexdigest()))
 ledger=Ledger(OUT/'auditor.sqlite')
 inputs={}
 for case in cases:
  ref=reference(case)
  if case['condition']=='precision_S256':
   inp=dqn.fixture(kind=case['kind'],seed=case['seed'],episodes=case['episodes'],random_seed=case['seed'])
   inp['scale']=256;inp['initial_coarse']=[v*16 for v in inp['initial_coarse']];inp['params']=[v*16 for v in inp['params']]
  else:
   inp=ref.fixture(kind=case['kind'],seed=case['seed'],episodes=case['episodes'],random_seed=case['seed'],**kwargs(case))
  inputs[case['id']]=inp
  save(OUT/'fixtures'/case['id']/'precommit-input.json',inp)
 for alg in ('dqn','a2c'):
  for seed in SEEDS:
   g=group_id(alg,seed)
   ledger.create_group(g,[c['id'] for c in cases if c['group_id']==g],
    expected_numerical=[j['id'] for j in jobs if j['group_id']==g])
 for case in cases:
  ref=reference(case);inp=inputs[case['id']]
  ledger.register(case['group_id'],case['id'],identities[case['program']],inp,
   config_commit=list(hashlib.sha256(ref.config_bytes(inp,**kwargs(case))).digest()),
   width=case['width'],capacity=case['capacity'],timing_repetitions=case['proof_repetitions'])
 for job in jobs:
  inp=inputs[job['base_case']]
  if job['algorithm']=='dqn':grid=inp['initial_coarse']
  else:grid=[v//(1<<24) for k in ('initial_actor','initial_critic') for v in inp[k]]
  spec=dict(job=job,baseline_precommit_input_sha256=sha(OUT/'fixtures'/job['base_case']/'precommit-input.json'),
   initial_grid=grid,rng_seed=inp['rng_seed'],source_freeze_sha256=sha(OUT/'SOURCE_FREEZE.json'),
   numerical_execution_is_cryptographic_proof=False,
   note='Numeric arm declaration, not a guest program approval. Q/S256 have separately registered compatible core inputs.')
  ledger.register_numerical(job['group_id'],job['id'],spec)
  save(OUT/'numerical-specifications'/job['id']/'specification.json',spec)
  save(OUT/'numerical-specifications'/job['id']/'job.json',job)
 save(OUT/'ledger-before-challenges.json',ledger.snapshot())
 # All groups and numerical arms are fixed before issuing even the first chi.
 for alg in ('dqn','a2c'):
  for seed in SEEDS:ledger.issue(group_id(alg,seed))
 for case in cases:
  inp=inputs[case['id']];approval=ledger.approval(case['id'])
  inp['challenge']=approval['challenge']
  assert list(hashlib.sha256(reference(case).config_bytes(inp,**kwargs(case))).digest())==approval['config_commit']
  save(OUT/'fixtures'/case['id']/'input.json',inp)
  ledger.export_approval(case['id'],OUT/'fixtures'/case['id']/'approval.json')
 save(OUT/'ledger-after-challenges.json',ledger.snapshot())
 save(OUT/'REGISTRATION.json',dict(schema='zkdrl-full-unified023-v1',cases=cases,numerical_jobs=jobs,
  requests=proof_requests(),formal_requests=66,numerical_runs=110,unique_proof_configurations=22,
  source_freeze_sha256=sha(OUT/'SOURCE_FREEZE.json'),identities=identities,
  input_sha256={c['id']:sha(OUT/'fixtures'/c['id']/'input.json') for c in cases},
  note='Three proof timing repetitions per fixed initialization/challenge; not extra learning seeds. All final data replace earlier cohorts; no pooling with prior cohorts.'))
 print('Registered41coreinputs+110numericdeclarations in10groups before challenges;22formalfixtures,66proofrequests.',flush=True)


def worker(case):
 directory=OUT/'fixtures'/case['id'];ref=reference(case)
 inp=json.loads((directory/'input.json').read_text())
 started=time.monotonic();py=ref.run(inp,True,**kwargs(case))
 save(directory/'reference.json',py)
 raw=ref.journal_bytes(py['journal'])
 with (directory/'reference.journal.bin').open('xb') as f:f.write(raw)
 captured=json.loads(call([binary(case['program']),'native',case['algorithm'],directory/'input.json','--capture'],directory/'native-capture'))
 if captured!=py:raise AssertionError('full native/reference state or audit mismatch: '+case['id'])
 quiet=json.loads(call([binary(case['program']),'native',case['algorithm'],directory/'input.json'],directory/'native-quiet'))
 expected=copy.deepcopy(py);expected['audit']=[]
 if case['algorithm']=='dqn':expected.update(state_chain=None,trajectory_chain=None)
 if quiet!=expected:raise AssertionError('quiet changed computation: '+case['id'])
 executed=last_json(call([binary(case['program']),'execute',case['algorithm'],directory/'input.json'],directory/'execute',2400))
 assert executed['exit_code']==0 and bytes.fromhex(executed['journal_hex'])==raw
 assert executed['journal']==py['journal']
 assert py['final_state']['rng']==case['episodes']*6*(10 if case['algorithm']=='dqn' else 1)
 row=dict(case=case,status='passed',full_state_and_audit_exact=True,quiet_exact=True,
  actual_guest_public_bytes_exact=True,cold_tmpdir=True,cycles=executed['instruction_cycles'],
  syscalls=executed['syscalls'],execute_seconds=executed['execute_seconds'],elapsed_seconds=time.monotonic()-started,
  input_sha256=sha(directory/'input.json'),journal_sha256=hashlib.sha256(raw).hexdigest())
 save(directory/'validation.json',row)
 print(json.dumps(dict(id=case['id'],status='passed',cycles=row['cycles'])),flush=True)


def validate():
 cases=core_cases()
 def launch(case):
  call([sys.executable,__file__,'--case',case['id']],OUT/'preflight-logs'/case['id'],3000)
  return json.loads((OUT/'fixtures'/case['id']/'validation.json').read_text())
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(launch,cases))
 for alg in ('dqn','a2c'):
  program='dqn-h16-c64' if alg=='dqn' else 'a2c'
  baseline=OUT/'fixtures'/f'{alg}-main_Q-exposure-s5000'/'reference.json'
  base=json.loads(baseline.read_text())
  for ep in (1000,2000,3000):
   sub=json.loads((OUT/'fixtures'/f'{program}-ep{ep}-s5000'/'reference.json').read_text())
   assert sub['audit']==base['audit'][:ep*(6 if alg=='dqn' else 1)]
 save(OUT/'preflight.json',dict(status='passed',cases=rows,actual_guest_cases=len(rows),
  exact_prefixes=True,cold_tmpdir=True,cryptographic_proofs=0))
 print('All41native/reference/actualguest cold-execution gates pass.',flush=True)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('phase',nargs='?',choices=['register','validate']);p.add_argument('--case');a=p.parse_args()
 if a.case:worker(next(c for c in core_cases() if c['id']==a.case))
 elif a.phase=='register':register()
 elif a.phase=='validate':validate()
 else:p.error('phase required')
