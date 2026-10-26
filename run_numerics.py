"""Execute all110 predeclared CPU arms; retain guards and infrastructure failures."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from config import ROOT,OUT,save,sha
from design import numerical_jobs,cell_id


def launch(job):
 directory=OUT/'numerical-runs'/job['id'];logs=OUT/'numerical-logs';logs.mkdir(exist_ok=True)
 spec=OUT/'numerical-specifications'/job['id']/'job.json'
 inp=OUT/'fixtures'/job['base_case']/'input.json'
 start=time.monotonic()
 with (logs/(job['id']+'.stdout')).open('x') as out,(logs/(job['id']+'.stderr')).open('x') as err:
  result=subprocess.run([sys.executable,'-B',str(ROOT/'numerics/runner.py'),'--job',str(spec),'--input',str(inp),'--out',str(directory)],
   stdout=out,stderr=err,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),timeout=1800)
 receipt=dict(id=job['id'],returncode=result.returncode,elapsed_seconds=time.monotonic()-start,
  job_sha256=sha(spec),input_sha256=sha(inp),cryptographic_proof=False)
 if result.returncode==0:
  r=json.loads((directory/'RESULT.json').read_text())
  for file,digest in r['files'].items():
   if sha(directory/file)!=digest:raise RuntimeError('numeric result hash mismatch')
  receipt.update(status=r['status'],result_sha256=sha(directory/'RESULT.json'))
 else:receipt['status']='infrastructure_or_programming_failure'
 save(logs/(job['id']+'.metrics.json'),receipt)
 print(json.dumps({k:receipt[k] for k in ('id','status','elapsed_seconds')}),flush=True)
 return receipt


def run():
 registration=json.loads((OUT/'REGISTRATION.json').read_text())
 assert len(registration['numerical_jobs'])==110 and registration['numerical_jobs']==numerical_jobs()
 freeze=json.loads((OUT/'SOURCE_FREEZE.json').read_text())['files']
 for path,digest in freeze.items():
  if sha(ROOT/path)!=digest:raise RuntimeError('source changed after prospective registration: '+path)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:receipts=list(pool.map(launch,registration['numerical_jobs']))
 save(OUT/'NUMERICAL_RUN_LEDGER.json',dict(planned=110,attempted=len(receipts),rows=receipts,
  infrastructure_failures=[r['id'] for r in receipts if r['status']=='infrastructure_or_programming_failure'],
  declared_guard_aborts=[r['id'] for r in receipts if r['status']=='arithmetic_guard_abort']))
 if any(r['returncode'] for r in receipts):raise RuntimeError('numeric infrastructure failures; inspect preserved logs, do not count as arithmetic guards')
 print('All110numericaljobs attempted and recorded.',flush=True)


def audit():
 import numpy as np
 records={};pairings={};matched=[];unavailable=[];aborted=[]
 reg=json.loads((OUT/'REGISTRATION.json').read_text())
 sys.path.insert(0,str(ROOT/'numerics'));from runner import encoded
 sys.path.insert(0,str(ROOT/'python'));from random_tape import CounterRng
 freeze=json.loads((OUT/'SOURCE_FREEZE.json').read_text())['files']
 source_expected={'runner.py':sha(ROOT/'numerics/runner.py')}
 source_expected.update({'python/'+str(p.relative_to(ROOT/'python')):sha(p) for p in sorted((ROOT/'python').rglob('*')) if p.suffix in ('.py','.json')})
 for job in reg['numerical_jobs']:
  directory=OUT/'numerical-runs'/job['id'];result=json.loads((directory/'RESULT.json').read_text())
  if result['job_id']!=job['id']:raise RuntimeError('numeric job binding')
  for f,h in result['files'].items():assert sha(directory/f)==h
  record=json.loads((directory/'record.json').read_text())
  assert record['job']==job and not record['cryptographic_proof'] and not record['guest_execution']
  inp_path=OUT/'fixtures'/job['base_case']/'input.json';inp=json.loads(inp_path.read_text())
  assert record['baseline_source_file_sha256']==sha(inp_path)
  assert record['job_source_file_sha256']==sha(OUT/'numerical-specifications'/job['id']/'job.json')
  assert record['baseline_input_sha256']==hashlib.sha256(encoded(inp)).hexdigest()
  assert record['source_sha256']==source_expected
  assert record['challenge']==inp['challenge'] and record['rng_seed']==inp['rng_seed']
  with np.load(directory/'arrays.npz',allow_pickle=False) as arrays:
   for name,value in arrays.items():
    expected=record['arrays'][name]
    assert list(value.shape)==expected['shape'] and value.dtype.str==expected['dtype']
    assert hashlib.sha256(value.tobytes(order='C')).hexdigest()==expected['sha256']
   raw=arrays['raw_tape'];grid=arrays['initial_grid']
   assert hashlib.sha256(raw.astype('<u4').tobytes()).hexdigest()==record['tape_sha256']
   assert hashlib.sha256(grid.astype('<i8').tobytes()).hexdigest()==record['initial_grid_sha256']
   expected_grid=inp['initial_coarse'] if job['algorithm']=='dqn' else [v//(1<<24) for k in ('initial_actor','initial_critic') for v in inp[k]]
   assert grid.tolist()==expected_grid
   rng=CounterRng(1 if job['algorithm']=='dqn' else 2,inp['rng_seed'],inp['challenge'])
   expected_tape=np.fromiter((rng.next_u32() for _ in range(record['planned_words'])),dtype='<u4')
   assert np.array_equal(raw,expected_tape)
  pair=(job['algorithm'],job['seed'])
  signature=(record['tape_sha256'],record['initial_grid_sha256'])
  if pair in pairings:assert pairings[pair]==signature
  else:pairings[pair]=signature
  if record['status']=='arithmetic_guard_abort':aborted.append(job['id'])
  for cp in record['checkpoints']:
   if cp.get('evaluation_status')=='unavailable':unavailable.append(dict(id=job['id'],episode=cp['episode'],error=cp.get('error')))
  if job['condition']=='main_Q' or (job['algorithm']=='dqn' and job['condition']=='precision_S256'):
   # These30 compatible numerical arms have independent ref/native/actualguest gates.
   reference_case=job['id']
   ref=json.loads((OUT/'fixtures'/reference_case/'reference.json').read_text())
   assert record['status']=='completed', 'approved Q/S256 must complete before proving'
   assert record['causal_state']==ref['final_state'], 'numeric/reference persistent state differs: '+job['id']
   if job['algorithm']=='dqn':assert record['coarse_export']==ref['coarse_export']
   else:
    assert record['actor_export']==ref['actor_export'] and record['critic_export']==ref['critic_export']
   assert json.loads((OUT/'fixtures'/reference_case/'validation.json').read_text())['status']=='passed'
   matched.append(job['id'])
  records[job['id']]=record
 assert len(records)==110 and len(pairings)==10 and len(matched)==30
 save(OUT/'NUMERICAL_GATE.json',dict(status='passed',attempted_runs=110,paired_seed_groups=10,
  causal_state_reference_native_guest_matches=matched,paired_initial_grid_and_tape_exact=True,
  source_input_job_hashes_verified=True,arithmetic_guard_aborts=aborted,
  unavailable_evaluations=unavailable,unavailable_analysis_rule='NA with available n; never impute0 or stop otherwise-valid training',
  unique_learning_seed_groups_per_algorithm=5,proof_seed_subset=[5000,5001,5002],
  note='Timing requests, task comparisons, checkpoints and episode prefixes are not independent learning replicates.'))
 print('110numericresults audited;30compatiblecausalstates match approvedreference/native/guest.',flush=True)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['run','audit']);a=p.parse_args()
 (run if a.phase=='run' else audit)()
