# zkDRL — anonymous artifact

Source code for **zkDRL: Verifying Closed-Loop Deep Reinforcement Learning for
Web Decision Services** (WWW '27 submission, anonymized for review).

The artifact contains the Rust guests and host driver that produce and verify
SP1/Groth16 proofs of complete DQN and A2C training runs, the independent Python
reference implementation, the 110-run numerical harness, the pre-registration
and audit harness, the analysis/plotting code, and the optional nonformal CPU
cycle probe. It does not contain author names, affiliations, email addresses,
cloud credentials, SSH keys, or Git history.

## What is in the box

| Path | Contents |
| --- | --- |
| `protocol/` | Shared SHA-256 counter RNG, canonical encodings, journal ABI, tests |
| `dqn/` | DQN core and SP1 guest (schema 202; H16/32/64, replay capacity 64/128/256) |
| `a2c/` | A2C core and SP1 guest (schema 203; Q16 forward/Q40 master/P32 mass) |
| `host/` | `zkdrl-ctc-host`: native, execute, prove, verify, tamper and audit CLI |
| `python/` | Independent NumPy reference/oracle for both learners and the MDPs |
| `numerics/` | Single-job harness for the 11 numerical conditions; no proof claims |
| `design.py` | Prospective 110-run / 41-gate / 22-configuration matrix |
| `prepare.py` | Freeze sources, register the matrix, issue challenges, run 41 cold gates |
| `run_numerics.py` | Run and audit all 110 numerical arms |
| `auditor.py` | Persistent local pre-registration/submission ledger |
| `acceptance.py` | Mandatory-outcome table separating checks from crypto acceptance |
| `analyze_numerics.py`, `plot_results.py` | Descriptive tables and paper figures |
| `cpu-profile/` | Optional nonformal CPU cycle probe (marker/control, no proof) |
| `reported-results/` | Sanitized summary tables from the paper campaign |
| `data/` | Derived KuaiRec fixture and its license; no raw dataset redistribution |
| `tests/` | Unit tests for the numerical harness, auditor ledger and acceptance table |
| `MANIFEST.sha256` | SHA-256 of every packaged file (excluding itself) |

`protocol/IMPLEMENTATION.md` documents the byte-level encodings. `host/README.md`
documents the CLI and trust boundary. `numerics/README.md` documents the
numerical harness and its missing-data rules. `docs/CODE_MAP.md` maps paper
sections to files.

## Quick check (Python only, no GPU)

```sh
python3 -m pip install -r requirements.txt
make test
python3 -B examples/python_smoke.py
```

`make test` runs the numerical-harness, ledger and acceptance unit tests.
`examples/python_smoke.py` runs both Python reference implementations on a tiny
12-episode fixture and prints their journals; it does not build a proof.
`bash scripts/check_manifest.sh` verifies the packaged file digests.

## Full reproduction ladder

The paper reports a single unified campaign with:

- 41 cold CPU gates: 20 Q, 10 S256, 10 scaling points and 1 A2C canary;
- 110 paired numerical runs: 2 tasks × 5 seeds × (5 DQN + 6 A2C conditions);
- 22 unique proof configurations × 3 fresh timing requests = 66 formal proofs,
  plus one separately labeled 2-episode A2C canary.

Reproduction is incremental; each level is useful on its own.

### 0. Environment

Tested environment: Python 3.13, NumPy 2.x, Rust/SP1 6.9.0 with `cargo prove`,
circuit v6.1.0 (BN254), and Linux. GPU proving used one A100-PCIE-40GB with
driver 595.91.07, 16 pinned EPYC 7713 CPUs and 62 GiB `/dev/shm`. Peak observed
GPU memory was about 26.6 GiB and host-tree RSS about 32 GiB. Python-only checks
need no GPU. See `REPRODUCE.md` for details.

### 1. Python reference and numerical harness

```sh
make test
python3 -B numerics/runner.py --job JOB.json --input INPUT.json --out DIR
```

The harness rejects missing/extra fields, wrong seeds or budgets, zero
challenges and non-grid initial masters. It never creates a challenge, proves an
execution, tunes a seed, or selects a checkpoint by performance.

### 2. Rust unit tests and guests

Install the SP1 6.9.0 toolchain so that `cargo prove` is on `PATH`, then:

```sh
./build.sh
```

This builds the A2C guest, the five DQN guest profiles, the five host variants,
and runs each host's unit tests. Guest ELFs and host binaries are generated
locally and are intentionally not shipped. Optional local toolchain variables
can be placed in `env.sh` (see `env.sh.example`); `build.sh` sources it if
present.

### 3. Registration, cold gates and numerical study

```sh
python3 -B prepare.py register     # freeze sources; register matrix; issue chi
python3 -B prepare.py validate     # 41 cold native/reference/guest gates
python3 -B run_numerics.py run     # 110 numerical arms, 4000 episodes each
python3 -B run_numerics.py audit   # hashes, tapes, grids, 30 causal states
python3 -B analyze_numerics.py     # CSV/JSON summaries
python3 -B plot_results.py numeric # figures
```

Use a fresh output directory for each registration; challenges are one-time.
By default outputs go to `runs/full-023-unified/`; override with `ZKDRL_OUT`.
`register` needs the host binaries from step 2. The full numerical study is
CPU-heavy; the 110 runs are independent processes and the harness uses four
workers.

### 4. GPU proofs (optional for code review)

`prepare.py register` writes an `approval.json` for every core case. A proof
request is then one host call:

```sh
SP1_PROVER=cuda CUDA_VISIBLE_DEVICES=0 \
ZKDRL_APPROVED_GPU_UUID=<uuid> \
bin/host-dqn-h16-c64 prove dqn \
  runs/full-023-unified/fixtures/<case>/input.json \
  runs/full-023-unified/fixtures/<case>/approval.json proof.bin
```

Verify on CPU with the same host binary:

```sh
bin/host-dqn-h16-c64 verify dqn proof.bin \
  runs/full-023-unified/fixtures/<case>/approval.json
```

The paper's 66 formal requests are three timing repetitions of 22 fixed
initialization/challenge pairs, not 66 learning samples. The host refuses CPU,
mock or network proving fallbacks; see `host/README.md`.

## Reported results

`reported-results/` contains the sanitized summary tables underlying the paper's
evaluation section: 22 proof configurations (median/min/max over three
requests), 22 numerical condition summaries, 110 endpoint rows and the A2C
checkpoint curves. Raw proof bundles, telemetry, cloud instance records,
auditor databases and per-run logs are not included; they contain operational
metadata rather than source code. The CSV/JSON files can be copied into an
output directory's `analysis/` folder to run `plot_results.py`. PDF/SVG figures
are intentionally not shipped because renderer metadata can carry timestamps or
tool fingerprints; they are regenerated locally from the tabular data.

## Scope and claim boundaries

The proofs establish **execution integrity** of a committed training run: the
public journal is the output of the approved guest on the committed private
input and challenge. They do not establish training optimality, input
provenance, privacy of the witness, honest auditor/OS behavior, or
representativeness of the public fixtures. The host/auditor trust boundary and
the explicit non-claims are documented in `host/README.md` and
`protocol/IMPLEMENTATION.md`. See `docs/CLAIMS.md` for the paper-level summary.

## Anonymity and data

- No author/institution metadata, email addresses, cloud account identifiers,
  SSH keys, API tokens, or `.git` history are included.
- Absolute workspace paths from the development environment were removed from
  source and documentation. Remaining URLs point to public dependencies,
  datasets, or specifications.
- The raw MiniWoB++ and KuaiRec archives are **not** redistributed. The training
  fixtures are deterministically generated from public seeds and the derived
  `data/recommendation.npz`; provenance and licenses are in `data/SOURCES.md`.
- `reported-results/` was copied from the accepted campaign and checked for
  paths, hostnames, instance IDs and credentials.
- Run `bash scripts/check_anonymity.sh` for the automated text scan. The full
  removal checklist and residual signals are in `ANONYMITY.md`.

If you find an identifying artifact, please report it through the anonymous
review system.

## License

No license is granted for the original code in this review copy. Add the
camera-ready license before public release. The bundled KuaiRec license is in
`data/KuaiRec-LICENSE`; third-party Rust crates retain their own licenses.
