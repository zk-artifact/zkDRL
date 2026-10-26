# Reproduction guide

This guide separates what can be checked on a laptop from what needs the full
CPU/GPU campaign. All commands assume the artifact root as the working
directory. The default output directory is `runs/full-023-unified/`; set
`ZKDRL_OUT` to relocate it. The harness creates files exclusively and never
overwrites a previous registration, so use a fresh directory per run.

## 1. Python-only checks

```sh
python3 -m pip install -r requirements.txt
make test
```

Expected: all unit tests pass. The tests exercise the independent Python
reference, the numerical harness and the auditor ledger. They do not check the
Rust guest or any cryptographic claim.

The numerical harness can be run on one job at a time:

```sh
python3 -B numerics/runner.py --job JOB.json --input INPUT.json --out DIR
```

`JOB.json` and `INPUT.json` must be an approved pair (seed/kind/budget/challenge
consistent). The harness writes `record.json`, `arrays.npz` and `RESULT.json`
into an empty directory and exits nonzero on infrastructure/programming errors.
Arithmetic-guard aborts are a normal `status: arithmetic_guard_abort` result,
not a crash and not an imputed success.

## 2. Building the guests and host

Prerequisites:

- Linux x86-64, Rust with the SP1 6.9.0 toolchain (`cargo prove` on `PATH`).
- `protoc` and `libclang` available to the Rust build.
- About 20 GB of free disk for Cargo caches and five host builds.
- No CUDA GPU is needed to build or to run `identity`, `native`, `execute`,
  `verify`, `tamper` or `audit`. `prove` requires a supported CUDA GPU.

```sh
./build.sh
```

`build.sh` writes guest ELFs under `dqn/elf*` and `a2c/elf`, host binaries under
`bin/`, and logs under `runs/full-023-unified/`. It runs the host unit tests for
each profile. If the toolchain lives in a nonstandard place, copy
`env.sh.example` to `env.sh` and edit it; `build.sh` sources `env.sh` when
present.

The host currently pins the SP1 SDK/verifier to 6.9.0 and checks circuit
v6.1.0. Version drift changes the program key and is rejected by design.

## 3. Registration and 41 cold gates

```sh
python3 -B prepare.py register
python3 -B prepare.py validate
```

`register`:

1. reads the six program identities from `bin/host-*` (`identity ALG`);
2. freezes every `.py/.rs/.json/.toml/.lock/.sh` source file into
   `SOURCE_FREEZE.json`;
3. writes 41 core fixtures and 110 numerical declarations;
4. creates an SQLite auditor ledger and registers all comparison groups *before*
   issuing any challenge;
5. issues one OS-CSPRNG challenge per algorithm/seed group and exports
   `approval.json` for every core case.

`validate` runs the 41 core cases in four workers. For each case it checks:

- Python reference state/audit equals the native core output;
- native `--capture` equals native quiet output modulo audit fields;
- the actual guest execution's public journal bytes equal the reference;
- cold `TMPDIR` and no reliance on a cached SP1 native helper;
- episode-prefix equality for the 1000/2000/3000 scaling cases.

Expected file: `runs/full-023-unified/preflight.json` with `status: passed`,
41 cases, and `cryptographic_proofs: 0`.

## 4. 110-run numerical study

```sh
python3 -B run_numerics.py run
python3 -B run_numerics.py audit
python3 -B analyze_numerics.py
python3 -B plot_results.py numeric
```

`run` starts one fresh subprocess per arm with
`OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`, retains failures, and
writes `NUMERICAL_RUN_LEDGER.json`. It does not retry or select a seed based on
results. `audit` verifies raw job/input/source/NPZ/grid/tape hashes, pairing and
the 30 compatible complete causal states, and writes `NUMERICAL_GATE.json`.
`analyze_numerics.py` writes descriptive CSV/JSON summaries; `plot_results.py`
writes PDF/SVG figures. Figures are intentionally not shipped; regenerate them
locally. Missing evaluations stay `NA` with explicit available `n`; they are
never zero-filled or silently dropped from the denominator.

This is the expensive CPU stage: 110 runs of up to 4000 episodes. The paper's
runs completed on the CPU host described in the paper; exact wall time depends
on the machine and BLAS build.

## 4b. Optional CPU cycle probe (nonformal)

The appendix's CPU cycle breakdown is a separate, unproved marker/control
measurement. It is not part of the 41 gates or the 66 proofs.

```sh
cd cpu-profile
python3 -B checksource.py
python3 -B build.py
python3 -B collect.py \
  ../runs/full-023-unified/fixtures/dqn-main_Q-exposure-s5000/input.json \
  ../runs/full-023-unified/fixtures/dqn-main_Q-exposure-s5000/reference.json
python3 -B report.py
```

`checksource.py` verifies byte-exact marker insertion against `../dqn` before
building. `collect.py` runs an 8-episode short input and the full 24000-slot
input, compares actual guest journals with the Python reference, and retains
raw logs under `cpu-profile/results/`. Counts are CPU instruction counts, not
GPU proving time. See `cpu-profile/README.md` for scope and inflation rules.

## 5. Optional GPU proof campaign

The paper's cohort is 22 fixed configurations × 3 timing requests = 66 formal
proofs plus one 2-episode A2C canary. `design.py` defines the exact cases and
request order; `prepare.py register` creates the approvals. A single request is:

```sh
SP1_PROVER=cuda CUDA_VISIBLE_DEVICES=0 \
ZKDRL_APPROVED_GPU_UUID=<uuid> \
bin/host-dqn-h16-c64 prove dqn \
  runs/full-023-unified/fixtures/<case>/input.json \
  runs/full-023-unified/fixtures/<case>/approval.json proof.bin
```

Then, on CPU:

```sh
bin/host-dqn-h16-c64 verify dqn proof.bin \
  runs/full-023-unified/fixtures/<case>/approval.json
bin/host-dqn-h16-c64 audit dqn proof.bin \
  runs/full-023-unified/fixtures/<case>/approval.json
bin/host-dqn-h16-c64 tamper dqn proof.bin \
  runs/full-023-unified/fixtures/<case>/approval.json
```

The expected public journal must be compared independently with
`fixtures/<case>/reference.journal.bin` and `reference.json`; the host does not
accept an expected-journal file as an argument. `tamper` must reject the proof
at the cryptographic stage, not merely fail a policy check.

The host enforces `SP1_PROVER=cuda`, device index 0 and a nonempty approved GPU
UUID. It has no CPU/mock/network proving fallback. A 40 GB GPU and roughly
32 GiB host memory were sufficient for the reported configurations; the largest
reported peak was about 32 GiB host-tree RSS and 26.6 GiB GPU memory.

## 6. Expected headline outputs

| Output | Expected |
| --- | --- |
| `preflight.json` | 41/41 cold gates, status passed |
| `NUMERICAL_RUN_LEDGER.json` | 110 attempted, 0 infrastructure failures |
| `NUMERICAL_GATE.json` | 110 runs, 10 seed groups, 30/30 causal-state matches |
| `SOURCE_FREEZE.json` | one digest per scientific source file |
| `REGISTRATION.json` | 41 core cases, 110 numerical jobs, 22 formal configs, 66 requests |
| 66 proof journals | 148/172 bytes, byte-identical to reference for each case |
| proof bundle | about 1.8 KiB per proof (paper cohort) |
| verification | lightweight call about 0.454–0.479 s; bundle/key setup separate |

The proof-timing values in this table are from the paper's single host and are
not a performance guarantee. The artifact does not ship raw proof bundles or the
paper's proof summary; regenerate them from your own proof campaign.

## 7. Common failure modes

- **`FileExistsError` on `register`**: the output directory already contains a
  registration. Challenges are one-time; choose a fresh `ZKDRL_OUT`.
- **`binary(...) not found`**: build the host variants first (`./build.sh`);
  `config.binary()` always looks in the artifact-root `bin/` directory.
- **`cargo prove: command not found`**: install the SP1 6.9.0 toolchain; plain
  `cargo` is not enough for guest builds.
- **`/dev/shm` too small**: the campaign needs tens of GiB for the largest
  configurations. `private_shm.sh` can provide an 8 GiB private mount when user
  namespaces are available; it never remounts the host/container mount.
- **CUDA/GPU mismatch**: `prove` requires the approved UUID and device 0; this
  is deliberate. Verification does not need CUDA.
- **Different floating-point results**: the FP64 arms depend on the numerical
  runtime; the fixed-point/QAT arms and journals are deterministic given the
  input and challenge. Do not compare NPZ container bytes as a commitment; the
  harness hashes the raw arrays.

## 8. What is intentionally not included

Raw proof bundles, cloud/rental orchestration, SSH material, provider instance
records, raw telemetry, auditor SQLite databases, per-run logs and paper result
tables are excluded. They are operational evidence or outputs, not source code,
and some contain account or host metadata. Figures are regenerated locally from
the outputs of the commands above.
