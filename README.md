# zkDRL — anonymous artifact

Source code for **zkDRL: Verifying Closed-Loop Deep Reinforcement Learning for
Web Decision Services** (WWW '27 submission, anonymized for review).

The artifact contains the Rust guests and host driver that produce and verify
SP1/Groth16 proofs of complete DQN and A2C training runs, the independent Python
reference implementation, the 110-run numerical harness, the pre-registration
and audit harness, the analysis/plotting code, and the optional nonformal CPU
cycle probe.

## Contents

| Path | Purpose |
| --- | --- |
| `protocol/` | Shared SHA-256 counter RNG, canonical encodings, journal ABI |
| `dqn/` | DQN core and SP1 guest (schema 202; H16/32/64, replay capacity 64/128/256) |
| `a2c/` | A2C core and SP1 guest (schema 203; Q16 forward/Q40 master/P32 mass) |
| `host/` | `zkdrl-ctc-host`: native, execute, prove, verify, tamper and audit CLI |
| `python/` | Independent NumPy reference/oracle for both learners and the MDPs |
| `numerics/` | Single-job harness for the 11 numerical conditions |
| `design.py` | Prospective 110-run / 41-gate / 22-configuration matrix |
| `prepare.py` | Freeze sources, register the matrix, issue challenges, run 41 cold gates |
| `run_numerics.py` | Run and audit all 110 numerical arms |
| `auditor.py` | Persistent local pre-registration/submission ledger |
| `evaluate_openings.py` | External policy evaluation after final-commitment opening |
| `analyze_numerics.py`, `plot_results.py` | Descriptive tables and paper figures |
| `cpu-profile/` | Optional nonformal CPU cycle probe (marker/control, no proof) |
| `data/` | Derived KuaiRec fixture and license; no raw dataset redistribution |
| `tests/` | Unit tests for the numerical harness and auditor ledger |
| `build.sh`, `env.sh.example` | Guest/host build entrypoint and optional toolchain config |
| `requirements.txt`, `Makefile` | Python dependencies and local commands |

`protocol/IMPLEMENTATION.md`, `host/README.md`, `numerics/README.md` and
`cpu-profile/README.md` document the byte-level protocol, CLI, numerical
harness, and profiling probe.

## Quick check (Python only, no GPU)

```sh
python3 -m pip install -r requirements.txt
make test
```

The tests exercise the independent Python reference, the numerical harness and
the auditor ledger. They do not check the Rust guest or any cryptographic claim.

## Full reproduction ladder

The paper reports one unified campaign with:

- 41 cold CPU gates: 20 Q, 10 S256, 10 scaling points and 1 A2C canary;
- 110 paired numerical runs: 2 tasks × 5 seeds × (5 DQN + 6 A2C conditions);
- 22 unique proof configurations × 3 fresh timing requests = 66 formal proofs,
  plus one separately labeled 2-episode A2C canary.

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

### 2. Rust guests and host

Install the SP1 6.9.0 toolchain so that `cargo prove` is on `PATH`, then:

```sh
./build.sh
```

This builds the A2C guest, the five DQN guest profiles, the five host variants,
and runs each host's unit tests. Guest ELFs and host binaries are generated
locally and are intentionally not shipped. Optional toolchain variables can be
placed in `env.sh` (see `env.sh.example`); `build.sh` sources it if present.

### 3. Registration, cold gates and numerical study

```sh
python3 -B prepare.py register     # freeze sources; register matrix; issue chi
python3 -B prepare.py validate     # 41 cold native/reference/guest gates
python3 -B run_numerics.py run     # 110 numerical arms, 4000 episodes each
python3 -B run_numerics.py audit   # hashes, tapes, grids, 30 causal states
python3 -B analyze_numerics.py     # CSV/JSON summaries
python3 -B plot_results.py numeric # numerical figures
```

Use a fresh output directory for each registration; challenges are one-time.
By default outputs go to `runs/full-023-unified/`; override with `ZKDRL_OUT`.
`register` needs the host binaries from step 2. The full numerical study is
CPU-heavy; the 110 runs are independent processes and the harness uses four
workers.

### 4. GPU proofs

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

## Scope and claim boundaries

The proofs establish **execution integrity** of a committed training run: the
public journal is the output of the approved guest on the committed private
input and challenge. They do not establish training optimality, input
provenance, privacy of the witness, honest auditor/OS behavior, or
representativeness of the public fixtures. The host/auditor trust boundary and
the explicit non-claims are documented in `host/README.md` and
`protocol/IMPLEMENTATION.md`.

## Data and licenses

The raw MiniWoB++ and KuaiRec archives are not redistributed. Training fixtures
are generated deterministically from public seeds; the small derived
`data/recommendation.npz` is kept for the legacy external evaluator. Provenance,
hashes and the KuaiRec CC BY 4.0 notice are in `data/SOURCES.md` and
`data/KuaiRec-LICENSE`. This review copy contains no author, account or cloud
operational metadata.

No license is granted for the original code in this review copy; add the
camera-ready license before public release. Third-party Rust crates and SP1
retain their upstream licenses.
