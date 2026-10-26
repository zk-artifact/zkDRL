"""Operational full-check outcomes, distinct from cryptographic submission acceptance.

Approved workloads/challenges stay in the frozen auditor ledger. This additional
local table cannot authorize a proof; it records whether ALL experimental checks
finished, so a later SDK/mutation/reference check failure cannot be hidden by an
earlier successful lightweight cryptographic submission.
"""
import json
import sqlite3
from auditor import canonical


class FullChecks:
 def __init__(self,path,jobs):
  self.path=path;self.jobs=jobs
  with sqlite3.connect(path,timeout=30) as db:
   db.execute('PRAGMA synchronous=FULL')
   db.execute('CREATE TABLE IF NOT EXISTS mandatory_request_outcomes(id TEXT PRIMARY KEY,run_id TEXT NOT NULL,repetition INTEGER NOT NULL,status TEXT NOT NULL,detail TEXT NOT NULL)')
   existing={r[0] for r in db.execute('SELECT id FROM mandatory_request_outcomes')}
   if existing and existing!=set(jobs):raise RuntimeError('mandatory-check request scope cannot change after initialization')
   for jid,job in jobs.items():
    row=db.execute('SELECT run_id,repetition FROM mandatory_request_outcomes WHERE id=?',(jid,)).fetchone()
    expected=(job['case']['id'],job.get('repetition',1))
    if row is None:db.execute('INSERT INTO mandatory_request_outcomes VALUES(?,?,?,?,?)',(jid,*expected,'pending','{}'))
    elif row!=expected:raise RuntimeError('mandatory-check request binding changed')
   actual={r[0] for r in db.execute('SELECT id FROM mandatory_request_outcomes')}
   if actual!=set(jobs):raise RuntimeError('mandatory-check request denominator changed')

 def rows(self):
  with sqlite3.connect(self.path,timeout=30) as db:
   db.row_factory=sqlite3.Row
   return [dict(r) for r in db.execute('SELECT * FROM mandatory_request_outcomes ORDER BY id')]

 def set(self,jid,status,detail):
  if jid not in self.jobs or status not in ('running','checks_passed','timeout','infrastructure_failure','verification_rejection','not_started'):
   raise ValueError('mandatory-check outcome')
  raw=canonical(detail)
  with sqlite3.connect(self.path,timeout=30) as db:
   db.execute('BEGIN IMMEDIATE')
   row=db.execute('SELECT status,detail FROM mandatory_request_outcomes WHERE id=?',(jid,)).fetchone()
   if not row:raise ValueError('unregistered request')
   if row[0]=='checks_passed':
    if status=='checks_passed' and json.loads(row[1])['proof_sha256']==detail['proof_sha256']:return
    raise ValueError('completed mandatory outcome cannot be replaced')
   if row[0] not in ('pending','running'):
    if (row[0],row[1])==(status,raw):return
    raise ValueError('failed mandatory outcome is terminal; explicit amendment required')
   db.execute('UPDATE mandatory_request_outcomes SET status=?,detail=? WHERE id=?',(status,raw,jid))

 def complete(self):
  rows=self.rows()
  return len(rows)==len(self.jobs) and all(r['status']=='checks_passed' for r in rows)
