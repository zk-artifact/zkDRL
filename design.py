"""Prospective full matrix: one coherent protocol,66proof requests and110CPU arms."""
SEEDS=(5000,5001,5002,5003,5004)
PROOF_SEEDS=SEEDS[:3]
CHECKPOINTS=(0,1000,2000,3000,4000)
PROGRAMS={
 'dqn-h16-c64':dict(algorithm='dqn',width=16,capacity=64),
 'dqn-h32-c64':dict(algorithm='dqn',width=32,capacity=64),
 'dqn-h64-c64':dict(algorithm='dqn',width=64,capacity=64),
 'dqn-h16-c128':dict(algorithm='dqn',width=16,capacity=128),
 'dqn-h16-c256':dict(algorithm='dqn',width=16,capacity=256),
 'a2c':dict(algorithm='a2c',width=16,capacity=None),
}
DQN_CONDITIONS=('main_F','main_N','main_Q','residual','precision_S256')
A2C_CONDITIONS=('main_F','main_N','main_Q','precision_Q20','operator_O1','operator_O2')


def cell_id(algorithm,condition,kind,seed):
 return f'{algorithm}-{condition}-'+('login' if kind==0 else 'exposure')+f'-s{seed}'


def group_id(algorithm,seed):return f'comparison-{algorithm}-s{seed}'


def numerical_jobs():
 rows=[]
 for algorithm,conditions in [('dqn',DQN_CONDITIONS),('a2c',A2C_CONDITIONS)]:
  for seed in SEEDS:
   for kind in (0,1):
    base=cell_id(algorithm,'main_Q',kind,seed)
    for condition in conditions:
     rows.append(dict(id=cell_id(algorithm,condition,kind,seed),algorithm=algorithm,
      condition=condition,kind=kind,seed=seed,episodes=4000,checkpoints=list(CHECKPOINTS),
      base_case=base,group_id=group_id(algorithm,seed)))
 assert len(rows)==110 and len({r['id'] for r in rows})==110
 return rows


def core_cases():
 cases=[]
 for algorithm in ('dqn','a2c'):
  for seed in SEEDS:
   for kind in (0,1):
    program='dqn-h16-c64' if algorithm=='dqn' else 'a2c'
    cases.append(dict(id=cell_id(algorithm,'main_Q',kind,seed),program=program,
     **PROGRAMS[program],kind=kind,seed=seed,episodes=4000,scale=16 if algorithm=='dqn' else 65536,
     axis='main' if seed in PROOF_SEEDS else 'numerical_Q',condition='main_Q',
     proof_repetitions=3 if seed in PROOF_SEEDS else 0,group_id=group_id(algorithm,seed)))
 for seed in SEEDS:
  for kind in (0,1):
   cases.append(dict(id=cell_id('dqn','precision_S256',kind,seed),program='dqn-h16-c64',
    **PROGRAMS['dqn-h16-c64'],kind=kind,seed=seed,episodes=4000,scale=256,
    axis='precision_probe',condition='precision_S256',proof_repetitions=0,group_id=group_id('dqn',seed)))
 for algorithm in ('dqn','a2c'):
  program='dqn-h16-c64' if algorithm=='dqn' else 'a2c'
  for episodes in (1000,2000,3000):
   cases.append(dict(id=f'{program}-ep{episodes}-s5000',program=program,**PROGRAMS[program],
    kind=1,seed=5000,episodes=episodes,scale=16 if algorithm=='dqn' else 65536,
    axis='episodes',condition='main_Q',proof_repetitions=3,group_id=group_id(algorithm,5000)))
 for program,axis in [('dqn-h32-c64','width'),('dqn-h64-c64','width'),('dqn-h16-c128','capacity'),('dqn-h16-c256','capacity')]:
  cases.append(dict(id=f'{program}-ep4000-s5000',program=program,**PROGRAMS[program],
   kind=1,seed=5000,episodes=4000,scale=16,axis=axis,condition='main_Q',proof_repetitions=3,
   group_id=group_id('dqn',5000)))
 cases.append(dict(id='a2c-canary-ep2-s5000',program='a2c',**PROGRAMS['a2c'],kind=1,seed=5000,
  episodes=2,scale=65536,axis='canary',condition='main_Q',proof_repetitions=1,group_id=group_id('a2c',5000)))
 assert len(cases)==41 and len({c['id'] for c in cases})==41
 assert sum(c['proof_repetitions']==3 for c in cases)==22
 return cases


def formal_cases():return [c for c in core_cases() if c['proof_repetitions']==3]


def proof_requests():
 cases=formal_cases();requests=[]
 for repetition,offset in [(1,0),(2,7),(3,14)]:
  for case in cases[offset:]+cases[:offset]:
   requests.append(dict(id=f'r{repetition}-{case["id"]}',case=case,repetition=repetition,
    timeout_seconds=7200 if case['width']==64 else 4800 if case['width']==32 else 3600))
 assert len(requests)==66
 return requests
