import concurrent.futures
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from auditor import Ledger


class LedgerTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
  self.ledger=Ledger(self.root/'ledger.sqlite')
  self.identity=dict(operation='identity',algorithm='dqn',elf_sha256='1'*64,vk_bytes32='0x'+'2'*64,profile=dict(width=16,capacity=64))
  self.inp=dict(challenge=[0]*32,kind=1,slots=24,scale=16,train=1)
 def tearDown(self):self.tmp.cleanup()
 def register(self,run='r',repetitions=3):
  self.ledger.register('g',run,self.identity,self.inp,config_commit=[ord(run[0])]*32,width=16,capacity=64,timing_repetitions=repetitions)
 def active(self,repetitions=3):
  self.ledger.create_group('g',['r']);self.register(repetitions=repetitions);self.ledger.issue('g')
 def submit(self,rep=1,raw=b'proof',journal=None,path=None):
  proof=self.root/('proof-'+str(len(list(self.root.glob('proof-*')))))
  proof.write_bytes(raw)
  report=dict(operation='verify',verified=True,run_id='r',journal=journal or {'return_units':1})
  fake=SimpleNamespace(returncode=0,stdout=json.dumps(report),stderr='')
  with patch('auditor.subprocess.run',return_value=fake):
   return self.ledger.verify_submission('r',proof,'/usr/bin/true',self.root/(path or f'sub-{rep}-{len(list(self.root.glob("sub-*")))}'),repetition=rep)
 def test_numerical_group_frozen_before_challenge(self):
  self.ledger.create_group('g',['r'],expected_numerical=['F','N','Q']);self.register()
  self.ledger.register_numerical('g','F',{'mode':'float'})
  with self.assertRaises(ValueError):self.ledger.issue('g')
  for name in ['N','Q']:self.ledger.register_numerical('g',name,{'mode':name})
  c=self.ledger.issue('g');self.assertEqual(c,self.ledger.approval('r')['challenge'])
  with self.assertRaises(ValueError):self.ledger.register_numerical('g','Q',{'mode':'changed'})
  with self.assertRaises(ValueError):self.ledger.issue('g')
 def test_distinct_fresh_proofs_same_statement_are_timing_repeats(self):
  self.active()
  self.assertEqual(self.submit()['ledger_status'],'accepted')
  self.assertEqual(self.submit(2,b'fresh2')['ledger_status'],'timing_repeat_verified')
  self.assertEqual(self.submit(3,b'fresh3')['ledger_status'],'timing_repeat_verified')
  self.assertEqual(self.submit(2,b'fresh2')['ledger_status'],'idempotent_same_submission')
  s=self.ledger.snapshot();self.assertEqual(len(s['runs']),1);self.assertEqual(len(s['proof_repetitions']),3)
 def test_copied_proof_and_changed_statement_rejected(self):
  self.active();self.submit()
  with self.assertRaises(ValueError):self.submit(2,b'proof')
  with self.assertRaises(ValueError):self.submit(2,b'new2',{'return_units':2})
  with self.assertRaises(ValueError):self.submit(1,b'changed')
 def test_repeat_needs_first_and_limits(self):
  self.active(1)
  with self.assertRaises(ValueError):self.submit(2,b'new2')
  self.submit()
  with self.assertRaises(ValueError):self.submit(2,b'new2')
 def test_cpu_only_has_no_accepted_proofs(self):
  self.active(0)
  with self.assertRaises(ValueError):self.submit()
 def test_restart_and_concurrent_challenge_no_reroll(self):
  self.ledger.create_group('g',['r']);self.register()
  def issue(_):
   try:return Ledger(self.ledger.path).issue('g')
   except ValueError:return None
  with concurrent.futures.ThreadPoolExecutor(max_workers=3) as p:rows=list(p.map(issue,range(3)))
  self.assertEqual(sum(r is not None for r in rows),1)
  self.assertEqual(next(r for r in rows if r),Ledger(self.ledger.path).approval('r')['challenge'])
 def test_pending_missing_or_duplicate_configuration_rejected(self):
  self.ledger.create_group('g',['r','s']);self.register()
  with self.assertRaises(ValueError):self.ledger.issue('g')
  with self.assertRaises(ValueError):
   self.ledger.register('g','s',self.identity,self.inp,config_commit=[ord('r')]*32,width=16,capacity=64)
 def test_later_timing_failure_is_terminal_without_losing_scientific_acceptance(self):
  self.active();self.submit()
  self.ledger.record_request_attempt('r',2,'attempt-1',{'request_id':'r2-r'})
  self.ledger.record_request_failure('r',2,'timeout',{'reason':'deadline'})
  snap=self.ledger.snapshot()
  self.assertEqual(snap['runs'][0]['status'],'accepted')
  self.assertEqual(next(r for r in snap['timing_requests'] if r['repetition']==2)['status'],'timeout')
  with self.assertRaises(ValueError):self.submit(2,b'new2')
  with self.assertRaises(ValueError):self.ledger.record_request_failure('r',2,'infrastructure_failure',{})
  self.assertEqual(len(snap['timing_requests']),3)

 def test_real_operational_verifier_failure_not_acceptance(self):
  self.active();proof=self.root/'bad';proof.write_bytes(b'bad')
  with self.assertRaises(ValueError):self.ledger.verify_submission('r',proof,'/usr/bin/false',self.root/'bad-submit')
  self.assertEqual(self.ledger.snapshot()['events'][-1]['action'],'submit_rejected')
  self.assertEqual(self.ledger.snapshot()['runs'][0]['status'],'challenged')

if __name__=='__main__':unittest.main()
