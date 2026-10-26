# Full110 numerical harness

`runner.py` exposes `run(job, base_input) -> (record, arrays)` and:

```sh
python3 -B numerics/runner.py --job JOB.json --input INPUT.json --out DIR
python3 -B -m unittest discover -s tests/numerics -v
```

Paths above assume the artifact root. The parent registers the matrix, issues
the challenge, and starts **one fresh subprocess per job**.
This harness does not generate challenges, launch jobs, prove executions, tune
seeds, or select a best checkpoint. Serial API calls are supported for tests;
concurrent calls in threads are unsupported because legacy module bindings are
briefly scoped and restored in `finally`.

Job fields: `id` (nonempty string), `algorithm` (`dqn`/`a2c`), `kind` (0 browser,
1 recommendation), integer `seed`, `condition`, `episodes` (1..4000), and sorted
unique `checkpoints` including 0. Extra annotations are retained, without
changing arithmetic. The parent matrix uses seeds 5000..5004, episodes 4000,
and checkpoints `[0,1000,2000,3000,4000]`; the API also permits smoke inputs.

| Algorithm | Conditions |
| --- | --- |
| DQN | main_F, main_N, main_Q, residual, precision_S256 |
| A2C | main_F, main_N, main_Q, precision_Q20, operator_O1, operator_O2 |

The same baseline schema-202 S16 or schema-203 Q16 input is passed to every
condition of an algorithm/kind/seed group. Reference validators enforce exact
input fields, lengths, and bounds. Seed, kind, and episode budget must match
the job; challenge must be nonzero. DQN requires training enabled and the
registered reward vector. A2C Q40 initial masters must lie on the Q16 grid.
Schema 203 has no explicit reward field: it uses the frozen legacy reward
parameters `[6,23,11,7]/16` (browser parameters are zero).

DQN uses the registered hidden16/capacity64/batch8/warmup32 configuration,
gamma 1/2, lr divisor16, epsilon1/4, target period32. F/N/Q/residual share
`initial_coarse/16`; S256 starts with exactly `initial_coarse*16` coarse units.
F uses FP64, N coarse integers, residual the low-gradient residual master,
and Q/S256 the high-gradient residual master. Greedy checkpoints use the
real-unit MDP with lowest-index ties.

A2C uses hidden16, six planned slots, denominator6, gamma7/8, lambda1,
actor rate1/32, critic rate1/64, entropy1/128 and frozen episode networks.
The N commit floors both candidate Q40 masters to Q16 and rechecks guards
before replacing either vector. Q20 preserves the original Q40 initialization.
Default Q uses local legacy math; O1/O2 use the byte-copied frozen tables and
operator core in `operators/`, whose JSON hashes are checked against the
copied contract. O1 has 2049 knots at 1/64; O2 has 256 cubic segments. The
selected mass implementation also applies to checkpoints. A2C checkpoint
returns and browser success probabilities use state-enumerated FP64 dynamic
programming over implemented masses/P (ordinary FP probabilities for F),
with both stochastic and greedy policies, without Monte Carlo.

The only training randomness is `python/random_tape.py::CounterRng` with
algorithm 1/2 and the supplied rng_seed/challenge. Every planned DQN slot uses
10 words (low4 exploration, low2 action, eight replay quantiles word/2^32);
A2C uses one raw word. Padding consumes words. `raw_tape` stores the **full
planned** tape as little-endian uint32 even after an abort; `consumed_words`
is the attempted cursor, which can be smaller. A2C integer trace column 15
contains the raw draw, not the cursor.

`record.json` contains normal JSON values, the original job, status
`completed` or `arithmetic_guard_abort`, error/failed-attempt details,
completed episodes, checkpoint evaluations, final state, counter annotations,
diagnostics, and array shape/dtype/content hashes. `baseline_input_sha256`
hashes canonical sorted compact JSON plus newline, before any condition
adaptation. CLI adds `baseline_source_file_sha256` for exact input-file bytes.
`initial_grid_sha256` hashes signed little-endian i64 baseline grid units
(DQN S16; A2C Q16 actor then critic). `tape_sha256` hashes full raw `<u4` bytes.
Source hashes and the NumPy version are included for reproducibility.

Training status and checkpoint availability are independent. Every attempted
checkpoint, including episode 0 and the final requested episode, has `episode`
and `evaluation_status` (`available` or `unavailable`). Available evaluations
retain the original DQN `return`/`success`/`actions` or A2C
`greedy`/`stochastic` fields. A declared arithmetic guard during evaluation
records `error` and optional structured `failed_guard` detail, saves the
checkpoint parameter arrays, and continues the original training. Unavailable
evaluations omit return/policy metrics: they are missing observations, never
zero returns or unsuccessful policies, and must not be imputed. An unavailable
final evaluation does not invalidate completed training or its `causal_state`.
`evaluation_failures` lists these unavailable checkpoint records;
`evaluation_failure_count` is its length, and `evaluation_counts` counts
`available` and `unavailable` attempted checkpoints. Checkpoints not reached
because training aborted are absent from these counts and arrays.

Completed DQN Q/S256 include reference-compatible `causal_state`,
`coarse_export`, and target export. For reference comparison, S256 requires
multiplying baseline initial_coarse and params by 16 and setting scale=256.
Completed A2C Q includes `causal_state`, `actor_export`, `critic_export`.
Other conditions retain raw final states without proof/equality claims.
All runs declare `cryptographic_proof=false` and `guest_execution=false`.

`arrays.npz` contains no object/pickle arrays: raw tape, normalized initial
grid, common real initialization, training returns, traces, checkpoint
parameters, final parameters/buffers, and per-update diagnostics. DQN trace
columns are slot, episode, within-episode slot, state[4], previous_done,
action, reward, successor[4], terminal, replay_indices[8], interactions,
updates, next_word_cursor. A2C columns are episode, slot, state[4], action,
reward, successor[4], previous_done, terminal, policy_version, raw_draw.
Rewards are native integer units except F's real rewards; episode returns
are always real units. Checkpoint arrays follow record.checkpoints order.
`update_diagnostic_columns` labels each diagnostics matrix. Fixed A2C shadow
errors compare unrounded backward arithmetic on the **same actual forward,
masses, and advantages**, not an independently trained floating model.
Record diagnostic maxima range over all committed updates. A2C
`numerical_maxima` contains maxima from **passed** audited training bounds,
including passed bounds within a subsequently failed attempt. The legacy
audit does not include the value that failed its bound, and checkpoint
evaluation does not contribute to that training audit. A scoped wrapper of
`fixed_math.bound` and the trainer's imported alias captures a failed bound
separately as `failed_guard` with `name`, actual `max_abs`, `strict_limit`
(failure means `max_abs >= strict_limit`), and `phase`. Training failures put
this detail at record level; evaluation failures put it on the checkpoint
and in `evaluation_failures`. Peaks and limits are normal Python/JSON integers,
including raw intermediates beyond int64. Other guard errors retain their
original error text and `failed_guard=null` when no measured values are
available. No evaluation is rerun to infer a peak. The wrapper leaves the
legacy guard decision, thresholds, exception, and audit unchanged and restores
both bindings even after aborts or unexpected exceptions. FP runs have no
integer residual or fixed-backward shadow-error metric.

Declared arithmetic guard aborts preserve completed traces, the pending
attempt/rollout, and available mutated legacy state without retrying or
rolling back the run. Nonfinite failed FP values are strings in JSON and
remain raw in NPZ. Validation, checksum, invariant, programming, and I/O
errors, including unexpected checkpoint exceptions, propagate with a traceback
and nonzero CLI exit; they are not safety rejections or unavailable evaluations.
Guard aborts produce a normal three-file result and exit zero;
the parent must inspect status. The CLI requires an empty output directory
and writes only `record.json`, `arrays.npz`, `RESULT.json`. RESULT contains
status and SHA256 of the other two exact files (no self-hash). Interrupted
writes can leave incomplete output; accept a run only after verifying RESULT.

Tests use unittest and at most 24 episodes per run. They cover all 11 arms,
reference equality in both environments, pairing, padding, challenges,
operator bounds/CDF/restoration, guard handling and CLI hashes.
Checkpoint-only guard injections at zero, intermediate, and final smoke
episodes preserve reference causal states, consumed words, updates, and all
snapshot arrays. Tests also cover unexpected evaluator exceptions, measured
failed peaks versus passed audits, strict equality, large Python integers,
and binding restoration across serial runs. No full
4000-episode run, native/guest comparison, cryptographic proof, or full-matrix
validation has been performed here. FP64 results depend on the numerical
runtime; NPZ container bytes are not the pairing commitment—raw arrays are.
