"""Trusted local auditor ledger, separate from the untrusted prover host.

SQLite persistence enforces registration -> one fresh group challenge -> verified
submission or recorded abort. This is not a public transparency service and does
not prove honesty of the auditor/OS, global uniqueness, or compulsory delivery.
Approval files exported here must reach verifiers over an auditor-trusted path.
"""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import os
import secrets
import sqlite3
import subprocess
import time

SCHEMA = 'zkdrl-ctc-approval-v1'
IDENTITY_FIELDS = ('algorithm', 'elf_sha256', 'vk_bytes32')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def last_json(text):
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and 'operation' in value:
            return value
    raise ValueError('no host operation JSON')


def identifier(value):
    if (type(value) is not str or not value or len(value.encode()) > 256
            or any(ord(ch) < 32 for ch in value)):
        raise ValueError('invalid identifier')
    return value


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS groups(
                    id TEXT PRIMARY KEY, expected TEXT NOT NULL,
                    challenge TEXT, status TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS runs(
                    id TEXT PRIMARY KEY, group_id TEXT NOT NULL REFERENCES groups(id),
                    approval TEXT NOT NULL, status TEXT NOT NULL,
                    proof_sha TEXT, abort_reason TEXT);
                CREATE TABLE IF NOT EXISTS numerical_requirements(
                    group_id TEXT PRIMARY KEY REFERENCES groups(id), expected TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS numerical_declarations(
                    id TEXT PRIMARY KEY, group_id TEXT NOT NULL REFERENCES groups(id),
                    specification TEXT NOT NULL, sha256 TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS repetition_limits(
                    run_id TEXT PRIMARY KEY REFERENCES runs(id), repetitions INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS timing_requests(
                    run_id TEXT NOT NULL REFERENCES runs(id), repetition INTEGER NOT NULL,
                    status TEXT NOT NULL, detail TEXT NOT NULL,
                    PRIMARY KEY(run_id,repetition));
                CREATE TABLE IF NOT EXISTS timing_attempts(
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
                    repetition INTEGER NOT NULL, attempt_id TEXT NOT NULL, detail TEXT NOT NULL,
                    UNIQUE(run_id,repetition,attempt_id));
                CREATE TABLE IF NOT EXISTS proof_repetitions(
                    run_id TEXT NOT NULL REFERENCES runs(id), repetition INTEGER NOT NULL,
                    proof_sha TEXT NOT NULL, journal TEXT NOT NULL,
                    PRIMARY KEY(run_id,repetition));
                CREATE TABLE IF NOT EXISTS events(
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, epoch REAL NOT NULL,
                    action TEXT NOT NULL, subject TEXT NOT NULL, detail TEXT NOT NULL);
            ''')
        self.path.chmod(0o600)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('PRAGMA synchronous=FULL')
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def transaction(self):
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            try:
                yield db
                db.execute('COMMIT')
            except BaseException:
                db.execute('ROLLBACK')
                raise

    @staticmethod
    def event(db, action, subject, detail):
        db.execute('INSERT INTO events(epoch,action,subject,detail) VALUES(?,?,?,?)',
                   (time.time(), action, subject, canonical(detail)))

    def create_group(self, group_id, expected_runs, *, expected_numerical=()):
        identifier(group_id)
        names = sorted(identifier(x) for x in expected_runs)
        numerical=sorted(identifier(x) for x in expected_numerical)
        if (not names or len(names) != len(set(names)) or len(numerical)!=len(set(numerical))):
            raise ValueError('distinct prospective run sets required')
        with self.transaction() as db:
            db.execute('INSERT INTO groups VALUES(?,?,NULL,?)',
                       (group_id, canonical(names), 'registering'))
            db.execute('INSERT INTO numerical_requirements VALUES(?,?)',(group_id,canonical(numerical)))
            self.event(db, 'create_group', group_id, {'expected_runs': names,'expected_numerical':numerical})

    def register_numerical(self, group_id, declaration_id, specification):
        identifier(declaration_id)
        raw=canonical(specification)
        digest=hashlib.sha256(raw.encode()).hexdigest()
        with self.transaction() as db:
            group=db.execute('SELECT status FROM groups WHERE id=?',(group_id,)).fetchone()
            required=db.execute('SELECT expected FROM numerical_requirements WHERE group_id=?',(group_id,)).fetchone()
            if not group or group['status']!='registering' or not required or declaration_id not in json.loads(required['expected']):
                raise ValueError('numerical arm not in open prospective comparison group')
            db.execute('INSERT INTO numerical_declarations VALUES(?,?,?,?)',(declaration_id,group_id,raw,digest))
            self.event(db,'register_numerical',declaration_id,{'specification_sha256':digest})
        return digest

    def register(self, group_id, run_id, identity, inp, *, config_commit,
                 width=16, capacity=None, timing_repetitions=1):
        identifier(run_id)
        if type(timing_repetitions) is not int or not 0<=timing_repetitions<=3:
            raise ValueError('approved proof repetitions must be0..3; zero means CPU-only')
        profile = identity['profile']
        if (width, capacity) != (profile['width'], profile['capacity']):
            raise ValueError('registration profile differs from trusted program identity')
        if (identity.get('operation') != 'identity' or
                any(ch not in '0123456789abcdef' for ch in identity['elf_sha256']) or
                len(identity['elf_sha256']) != 64 or
                len(identity['vk_bytes32']) != 66 or not identity['vk_bytes32'].startswith('0x') or
                any(ch not in '0123456789abcdef' for ch in identity['vk_bytes32'][2:])):
            raise ValueError('invalid trusted program identity')
        if len(config_commit) != 32 or any(type(v) is not int or not 0 <= v <= 255 for v in config_commit):
            raise ValueError('config commitment must be 32 bytes')
        if inp['challenge'] != [0] * 32:
            raise ValueError('registration requires pre-challenge placeholder')
        alg = identity['algorithm']
        if alg not in ('dqn', 'a2c'):
            raise ValueError('algorithm')
        episodes = inp['slots']//6 if alg == 'dqn' else inp['episodes']
        scale = inp['scale'] if alg == 'dqn' else 65536
        if (type(episodes) is not int or not 1 <= episodes <= 4000 or
                type(inp['kind']) is not int or inp['kind'] not in (0, 1) or
                scale not in ((16, 256) if alg == 'dqn' else (65536,)) or
                (alg == 'dqn' and (inp['slots'] % 6 or inp['train'] != 1))):
            raise ValueError('invalid prospective workload')
        approval = dict(schema=SCHEMA, run_id=run_id, group_id=group_id,
            algorithm=alg, elf_sha256=identity['elf_sha256'],
            vk_bytes32=identity['vk_bytes32'], config_commit=config_commit,
            challenge=None, kind=inp['kind'],
            episodes=episodes, scale=scale,
            width=width, capacity=capacity)
        with self.transaction() as db:
            group = db.execute('SELECT * FROM groups WHERE id=?', (group_id,)).fetchone()
            if not group or group['status'] != 'registering':
                raise ValueError('group unavailable for registration')
            if run_id not in json.loads(group['expected']):
                raise ValueError('run absent from prospective group')
            for row in db.execute('SELECT approval FROM runs WHERE group_id=?', (group_id,)):
                existing=json.loads(row['approval'])
                if (existing['config_commit'],existing['vk_bytes32']) == (config_commit,approval['vk_bytes32']):
                    raise ValueError('duplicate statement in shared-challenge group; timing repeats are not new jobs')
            db.execute('INSERT INTO runs VALUES(?,?,?,?,NULL,NULL)',
                       (run_id, group_id, canonical(approval), 'registered'))
            db.execute('INSERT INTO repetition_limits VALUES(?,?)',(run_id,timing_repetitions))
            for repetition in range(1,timing_repetitions+1):
                db.execute('INSERT INTO timing_requests VALUES(?,?,?,?)',(run_id,repetition,'pending','{}'))
            self.event(db, 'register', run_id, {'approval': approval,'timing_repetitions':timing_repetitions})
        return approval

    def issue(self, group_id):
        with self.transaction() as db:
            group = db.execute('SELECT * FROM groups WHERE id=?', (group_id,)).fetchone()
            if not group or group['status'] != 'registering':
                raise ValueError('challenge already issued or group unknown; cannot reroll')
            rows = db.execute('SELECT id,status FROM runs WHERE group_id=? ORDER BY id', (group_id,)).fetchall()
            if [r['id'] for r in rows] != json.loads(group['expected']):
                raise ValueError('all prospective group members must register first')
            if any(r['status'] != 'registered' for r in rows):
                raise ValueError('invalid registration states')
            required=db.execute('SELECT expected FROM numerical_requirements WHERE group_id=?',(group_id,)).fetchone()
            declarations=db.execute('SELECT id FROM numerical_declarations WHERE group_id=? ORDER BY id',(group_id,)).fetchall()
            if not required or [r['id'] for r in declarations]!=json.loads(required['expected']):
                raise ValueError('all numerical comparison arms must be declared before challenge')
            challenge = list(secrets.token_bytes(32))
            db.execute('UPDATE groups SET challenge=?,status=? WHERE id=?',
                       (canonical(challenge), 'challenged', group_id))
            for row in rows:
                raw = db.execute('SELECT approval FROM runs WHERE id=?', (row['id'],)).fetchone()[0]
                value = json.loads(raw)
                value['challenge'] = challenge
                db.execute('UPDATE runs SET approval=?,status=? WHERE id=?',
                           (canonical(value), 'challenged', row['id']))
            self.event(db, 'issue_challenge', group_id,
                       {'challenge': challenge, 'source': 'auditor OS CSPRNG after atomic registration'})
        return challenge

    def approval(self, run_id):
        with self.connection() as db:
            row = db.execute('SELECT approval,status FROM runs WHERE id=?', (run_id,)).fetchone()
        if not row or row['status'] not in ('challenged', 'accepted'):
            raise ValueError('no active issued approval')
        return json.loads(row['approval'])

    def export_approval(self, run_id, path):
        value = self.approval(run_id)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x') as out:
            out.write(canonical(value) + '\n')
            out.flush()
            os.fsync(out.fileno())
        return value

    def record_request_attempt(self, run_id, repetition, attempt_id, detail):
        identifier(attempt_id)
        with self.transaction() as db:
            row=db.execute('SELECT status FROM timing_requests WHERE run_id=? AND repetition=?',(run_id,repetition)).fetchone()
            if not row:raise ValueError('timing request not predeclared')
            db.execute('INSERT INTO timing_attempts(run_id,repetition,attempt_id,detail) VALUES(?,?,?,?)',
                (run_id,repetition,attempt_id,canonical(detail)))
            if row['status']=='pending':db.execute('UPDATE timing_requests SET status=? WHERE run_id=? AND repetition=?',('running',run_id,repetition))
            self.event(db,'timing_attempt',run_id,{'repetition':repetition,'attempt_id':attempt_id,'detail':detail})

    def record_request_failure(self, run_id, repetition, outcome, detail):
        if outcome not in ('timeout','infrastructure_failure','verification_rejection','not_started'):
            raise ValueError('explicit terminal failure classification required')
        raw=canonical(detail)
        with self.transaction() as db:
            row=db.execute('SELECT status,detail FROM timing_requests WHERE run_id=? AND repetition=?',(run_id,repetition)).fetchone()
            if not row:raise ValueError('timing request not predeclared')
            if row['status']==outcome and row['detail']==raw:return
            if row['status'] not in ('pending','running'):raise ValueError('cannot replace terminal timing outcome')
            db.execute('UPDATE timing_requests SET status=?,detail=? WHERE run_id=? AND repetition=?',(outcome,raw,run_id,repetition))
            self.event(db,'timing_request_failed',run_id,{'repetition':repetition,'outcome':outcome,'detail':detail})

    def record_abort(self, run_id, reason):
        if not reason:
            raise ValueError('abort reason required')
        with self.transaction() as db:
            row = db.execute('SELECT status FROM runs WHERE id=?', (run_id,)).fetchone()
            if not row or row['status'] != 'challenged':
                raise ValueError('only pending issued jobs can abort')
            db.execute('UPDATE runs SET status=?,abort_reason=? WHERE id=?', ('aborted', reason, run_id))
            self.event(db, 'abort', run_id, {'reason': reason})

    def verify_submission(self, run_id, proof, binary, out_dir, *, timeout=300, repetition=1):
        """Only a trusted verifier binary result can transition a run to accepted.

        Proof and approval copies are frozen before invoking verifier. A malicious
        prover cannot replace a verified file between verification and accounting.
        """
        if type(repetition) is not int or not 1<=repetition<=3:
            raise ValueError('invalid timing repetition index')
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=False)
        approval = self.export_approval(run_id, out_dir/'approval.json')
        with Path(proof).open('rb') as src:
            raw = src.read(1024*1024+1)
        if len(raw) > 1024*1024:
            raise ValueError('oversized proof')
        frozen = out_dir/'submitted.proof'
        with frozen.open('xb') as dst:
            dst.write(raw)
        proof_sha = hashlib.sha256(raw).hexdigest()
        with self.transaction() as db:
            self.event(db, 'submit_attempt', run_id, {'proof_sha256': proof_sha,'timing_repetition':repetition})
        try:
            result = subprocess.run([str(binary), 'verify', approval['algorithm'],
                str(frozen), str(out_dir/'approval.json')], text=True,
                capture_output=True, timeout=timeout, check=False)
            (out_dir/'verify.stdout').write_text(result.stdout)
            (out_dir/'verify.stderr').write_text(result.stderr)
            if result.returncode:
                raise ValueError('submission rejected by trusted verifier')
            report = last_json(result.stdout)
            if not report.get('verified') or report.get('run_id') != run_id:
                raise ValueError('invalid trusted verifier result')
        except Exception as error:
            with self.transaction() as db:
                self.event(db, 'submit_rejected', run_id, {'proof_sha256': proof_sha,'timing_repetition':repetition,
                    'error': type(error).__name__ + ': ' + str(error)})
            raise
        with self.transaction() as db:
            row = db.execute('SELECT status,proof_sha,approval FROM runs WHERE id=?', (run_id,)).fetchone()
            if not row or row['approval'] != canonical(approval):
                raise ValueError('approval changed during verification')
            limit=db.execute('SELECT repetitions FROM repetition_limits WHERE run_id=?',(run_id,)).fetchone()
            if not limit or repetition>limit['repetitions']:
                raise ValueError('proof request exceeds predeclared timing repetitions')
            request=db.execute('SELECT status FROM timing_requests WHERE run_id=? AND repetition=?',(run_id,repetition)).fetchone()
            if not request or request['status'] not in ('pending','running','succeeded'):
                raise ValueError('failed timing request cannot silently become a successful measurement')
            previous=db.execute('SELECT proof_sha,journal FROM proof_repetitions WHERE run_id=? AND repetition=?',(run_id,repetition)).fetchone()
            journal=canonical(report['journal'])
            copied=db.execute('SELECT repetition FROM proof_repetitions WHERE run_id=? AND proof_sha=?',(run_id,proof_sha)).fetchone()
            if copied and copied['repetition']!=repetition:
                raise ValueError('copied proof cannot count as a fresh timing request')
            if previous:
                if previous['proof_sha']==proof_sha and previous['journal']==journal:
                    return dict(report,ledger_status='idempotent_same_submission',timing_repetition=repetition)
                raise ValueError('cannot replace a completed timing request')
            if repetition==1:
                if row['status']!='challenged':raise ValueError('job closed; cannot replace accepted or aborted result')
                db.execute('UPDATE runs SET status=?,proof_sha=? WHERE id=?',('accepted',proof_sha,run_id))
                status='accepted'
            else:
                first=db.execute('SELECT journal FROM proof_repetitions WHERE run_id=? AND repetition=1',(run_id,)).fetchone()
                if row['status']!='accepted' or not first or first['journal']!=journal:
                    raise ValueError('timing repetition requires identical authenticated statement to accepted first request')
                status='timing_repeat_verified'
            db.execute('INSERT INTO proof_repetitions VALUES(?,?,?,?)',(run_id,repetition,proof_sha,journal))
            db.execute('UPDATE timing_requests SET status=?,detail=? WHERE run_id=? AND repetition=?',
                ('succeeded',canonical({'proof_sha256':proof_sha}),run_id,repetition))
            self.event(db,status,run_id,{'proof_sha256':proof_sha,'timing_repetition':repetition,
                'verifier_sha256':digest(binary)})
        return dict(report,ledger_status=status,timing_repetition=repetition)

    def snapshot(self):
        with self.connection() as db:
            tables=('groups','runs','events','numerical_requirements','numerical_declarations','repetition_limits','proof_repetitions','timing_requests','timing_attempts')
            return {table:[dict(row) for row in db.execute('SELECT * FROM '+table)] for table in tables}
