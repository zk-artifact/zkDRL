import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from acceptance import FullChecks


class FullOutcomeTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'db'
  self.jobs={'r1-a':dict(case={'id':'a'},repetition=1),'r2-a':dict(case={'id':'a'},repetition=2),'canary':dict(case={'id':'c'},repetition=1)}
  self.full=FullChecks(self.path,self.jobs)
 def tearDown(self):self.tmp.cleanup()
 def test_light_success_does_not_hide_later_mandatory_failure(self):
  with sqlite3.connect(self.path) as db:
   db.execute('CREATE TABLE crypto_acceptance(id TEXT,status TEXT)');db.execute('INSERT INTO crypto_acceptance VALUES(?,?)',('r1-a','succeeded'))
  self.full.set('r1-a','running',{})
  self.full.set('r1-a','verification_rejection',{'reason':'laterSDKcheckfailed'})
  self.assertFalse(self.full.complete())
  with self.assertRaises(ValueError):self.full.set('r1-a','checks_passed',{'proof_sha256':'x'})
 def test_canary_and_every_registered_slot_count(self):
  for jid in ['r1-a','r2-a']:self.full.set(jid,'checks_passed',{'proof_sha256':jid})
  self.assertFalse(self.full.complete())
  self.full.set('canary','checks_passed',{'proof_sha256':'c'})
  self.assertTrue(self.full.complete())
  self.full.set('r1-a','checks_passed',{'proof_sha256':'r1-a'})
  with self.assertRaises(ValueError):self.full.set('r1-a','checks_passed',{'proof_sha256':'changed'})
 def test_restart_preserves_outcomes_and_scope(self):
  self.full.set('r1-a','checks_passed',{'proof_sha256':'a'})
  restored=FullChecks(self.path,self.jobs)
  self.assertEqual(restored.rows(),self.full.rows())
  with self.assertRaises(RuntimeError):FullChecks(self.path,{**self.jobs,'new':dict(case={'id':'new'},repetition=1)})

if __name__=='__main__':unittest.main()
