"""Independent schema-203 reference using local legacy integer A2C arithmetic.

Public deterministic fixtures are numerical test data, not fairness evidence.
Their zero challenge is a placeholder for an auditor's post-registration draw.

Add this python directory to sys.path, then import a2c_reference. Public API:
fixture(kind=1, seed=300, episodes=4000, *, random_seed=0)
run(i, capture=True), config_bytes(i), journal_bytes(j)
load_input(path) and loads_input(json_text) reject duplicate JSON keys.
Initial actor and critic fixtures use floor(Q40 / 2**24) * 2**24 masters.
"""
from pathlib import Path
import numpy as np
from legacy.a2c_fixed import fixed_math as fm
from legacy.a2c_fixed.fixtures import reward_params
from legacy.model import env_init, env_step
from random_tape import (sha, i64s, u32s, u64s, CounterRng, public_rng_seed,
                         integer, array, loads)

SCHEMA, STATE_BYTES = 203, 2036
TABLE_SHA = bytes.fromhex('39786b3a4a9ffc46842ae90d55beedaba84f1156cfd365e7eeb8b3f43b0f348d')
INPUT_FIELDS = frozenset(('schema','kind','episodes','seed','rng_seed','initial_actor',
                          'initial_critic','config_salt','final_salt','challenge'))


def validate(i):
    if type(i) is not dict or set(i) != INPUT_FIELDS:
        raise ValueError('PrivateInput requires exactly the schema-203 fields')
    for k in ('schema','kind','episodes'):
        integer(i[k], 0, (1<<32)-1, k)
    integer(i['seed'], 0, (1<<64)-1, 'seed')
    if i['schema'] != SCHEMA or i['kind'] not in (0,1) or not 1 <= i['episodes'] <= 4000:
        raise ValueError('schema/kind/episodes')
    for k,n in (('initial_actor',148),('initial_critic',97)):
        array(i[k], n, -(1<<63), (1<<63)-1, k)
        fm.bound(i[k], 256*fm.G, 'master')
        fm.bound([v >> 24 for v in i[k]], 1<<24, 'quantized master')
    for k in ('rng_seed','config_salt','final_salt','challenge'):
        array(i[k], 32, 0, 255, k)


def loads_input(data):
    i = loads(data)
    validate(i)
    return i


def load_input(path):
    return loads_input(Path(path).read_text())


def fixture(kind=1, seed=300, episodes=4000, *, random_seed=0):
    actor, critic = fm.initial_masters(seed)
    ratio = 1<<24
    label = f'PUBLIC-CTC203-{kind}-{seed}-{episodes}'.encode()
    i = dict(schema=SCHEMA, kind=kind, episodes=episodes, seed=seed,
             rng_seed=public_rng_seed(2, random_seed),
             initial_actor=(actor//ratio*ratio).tolist(),
             initial_critic=(critic//ratio*ratio).tolist(),
             config_salt=list(sha(label+b'/config')), final_salt=list(sha(label+b'/final')),
             challenge=[0]*32)
    validate(i)
    return i


def config_bytes(i):
    validate(i)
    return (b'zkDRL/ctc/a2c/config/v2\0'
            + u32s([SCHEMA,i['kind'],i['episodes'],16,40,32,48,16,6,7,8,1,32,64,128,256,32])
            + u64s([i['seed']]) + bytes(i['rng_seed'])
            + i64s(i['initial_actor']) + i64s(i['initial_critic'])
            + bytes(i['config_salt']) + bytes(i['final_salt']) + TABLE_SHA)


def state_bytes(s):
    if len(s['actor']) != 148 or len(s['critic']) != 97 or len(s['state']) != 4:
        raise ValueError('state dimensions')
    raw = (i64s(s['actor']) + i64s(s['critic']) + u64s([s['rng']]) + i64s(s['state'])
           + u32s([int(s['done'])]) + i64s([s['total']])
           + u32s([s[k] for k in ('episodes','updates','interactions','resets','padded','successes')]))
    assert len(raw) == STATE_BYTES
    return raw


def journal_bytes(j):
    raw = u32s([j['schema']]) + b''.join(bytes(j[k]) for k in ('config_commit','initial_commit','final_commit','challenge'))
    raw += u32s([j[k] for k in ('kind','episodes','slots','interactions','updates','resets','padded','successes')])
    raw += i64s([j['return_units']])
    if len(raw) != 172:
        raise ValueError('journal length')
    return raw


def model_bytes(actor, critic, updates):
    return b'zkRL/A2C-fixed/state/v1\0' + u32s([16]) + u64s([updates]) + i64s(actor) + i64s(critic)


def stage_bytes(z, v, mass, targets, advantages, dz, dv, ac, cc, ga, gc):
    raw = bytearray(b'zkRL/A2C-SP1/stages/v1\0' + u32s([len(z)]))
    for t in range(len(z)):
        raw.extend(i64s(z[t])); raw.extend(i64s(v[t])); raw.extend(u64s(mass[t]))
        raw.extend(i64s([targets[t], advantages[t]])); raw.extend(i64s(dz[t])); raw.extend(i64s(dv[t]))
        for row in (ac.pre[t], ac.hidden[t], cc.pre[t], cc.hidden[t]):
            raw.extend(i64s(row))
    raw.extend(i64s(ga)); raw.extend(i64s(gc))
    return bytes(raw)


def run(i, capture=True):
    validate(i)
    if type(capture) is not bool:
        raise ValueError('capture must be bool')
    actor, critic = fm.ints(i['initial_actor']), fm.ints(i['initial_critic'])
    config = sha(config_bytes(i))
    initial = sha(b'zkDRL/ctc/a2c/initial/v2\0' + config + bytes(i['config_salt']) + i64s(actor) + i64s(critic))
    tape = CounterRng(2, i['rng_seed'], i['challenge'])
    kind = 'browser' if i['kind'] == 0 else 'recommendation'
    params = reward_params(kind, 16)
    interactions = total = padded = successes = 0
    audit = []
    for ep in range(i['episodes']):
        aq, cq = actor >> 24, critic >> 24
        state, done = env_init(kind), False
        states, actions, rewards, terminals, continues = [], [], [], [], []
        trace = [] if capture else None
        for t in range(6):
            draw = tape.next_u32()
            before, previous_done = state.copy(), done
            if done:
                action = 4
            else:
                z, _ = fm.forward(aq, state[None,:], 4, 16)
                # Check critic on every active state, including before a later
                # transition might fail. The batched pass below computes VJPs.
                fm.forward(cq, state[None,:], 1, 16)
                action = fm.categorical(fm.masses(z,16)[0], draw)
            successor, reward, terminal = env_step(kind, state, action, params, 1<<16, done)
            reward, terminal = int(reward), bool(terminal)
            if np.any(successor < 0) or np.any(successor > 6):
                raise OverflowError('state domain')
            fm.bound([reward], 4*(1<<16), 'reward')
            if not previous_done:
                states.append(before); actions.append(action); rewards.append(reward)
                terminals.append(int(terminal)); continues.append(int(not terminal and t<5))
                interactions += 1
                successes += int(i['kind']==0 and terminal and reward>0)
            else:
                padded += 1
            total += reward
            if capture:
                trace.append([ep,t,*map(int,before),action,reward,*map(int,successor),
                              int(previous_done),int(terminal),ep,draw])
            state, done = successor, terminal
        z, ac = fm.forward(aq, states, 4, 16)
        v, cc = fm.forward(cq, states, 1, 16)
        mass = fm.masses(z, 16)
        targets = fm.finite_returns(rewards, terminals, continues, 16)
        advantages = fm.bound(targets-v[:,0]*(1<<24), 1<<56, 'advantage')
        dz = fm.actor_upstream(z, mass, actions, advantages, 16)
        dv = -advantages[:,None]
        ga, gc = fm.backward(ac,dz,4,16), fm.backward(cc,dv,1,16)
        # Both candidates and their asymmetric negative quantization edge are
        # checked before either master vector is replaced.
        next_actor, next_critic = fm.candidates(actor,critic,ga,gc,6)
        fm.bound(next_actor >> 24, 1<<24, 'quantized actor')
        fm.bound(next_critic >> 24, 1<<24, 'quantized critic')
        actor, critic = next_actor, next_critic
        if capture:
            audit.append(dict(episode=ep, trace=trace,
                              model_sha256=list(sha(model_bytes(actor,critic,ep+1))),
                              stages_sha256=list(sha(stage_bytes(z,v,mass,targets,advantages,dz,dv,ac,cc,ga,gc)))))
    fs = dict(actor=actor.tolist(), critic=critic.tolist(), rng=tape.position(), state=state.tolist(),
              done=done, total=total, episodes=i['episodes'], updates=i['episodes'], interactions=interactions,
              resets=i['episodes'], padded=padded, successes=successes)
    final = sha(b'zkDRL/ctc/a2c/final/v2\0' + config + bytes(i['challenge']) + bytes(i['final_salt']) + state_bytes(fs))
    journal = dict(schema=SCHEMA, config_commit=list(config), initial_commit=list(initial), final_commit=list(final),
                   challenge=i['challenge'], kind=i['kind'], episodes=i['episodes'], slots=i['episodes']*6,
                   interactions=interactions, updates=i['episodes'], resets=i['episodes'], padded=padded,
                   successes=successes, return_units=total)
    return dict(journal=journal, final_state=fs, actor_export=(actor>>24).tolist(),
                critic_export=(critic>>24).tolist(), audit=audit)
