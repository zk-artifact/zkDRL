# zkDRL host

`zkdrl-ctc-host` is the prover/verifier driver. It embeds one DQN guest and the
A2C guest selected at compile time, derives the program key from the embedded
ELF, and exposes a deliberately small CLI.

```text
host identity ALG
host native   ALG INPUT [--capture]
host execute  ALG INPUT
host prove    ALG INPUT APPROVAL PROOF
host verify   ALG PROOF APPROVAL
host tamper   ALG PROOF APPROVAL
host audit    ALG PROOF APPROVAL
```

`ALG` is exactly `dqn` or `a2c`. Extra arguments and legacy command forms are
rejected.

## Build

The parent builds the real guest ELF first; the host embeds it with
`include_bytes!`. From the artifact root:

```sh
./build.sh
```

`build.sh` builds the A2C guest, the five DQN profiles, then each host variant
and runs its unit tests. The selected profile is compile-time only:

| Host feature | DQN profile | Embedded DQN guest |
| --- | --- | --- |
| default (none) | H16/C64 | `../dqn/elf/zkdrl-ctc-dqn-guest` |
| `h32` | H32/C64 | `../dqn/elf-h32/zkdrl-ctc-dqn-guest` |
| `h64` | H64/C64 | `../dqn/elf-h64/zkdrl-ctc-dqn-guest` |
| `c128` | H16/C128 | `../dqn/elf-c128/zkdrl-ctc-dqn-guest` |
| `c256` | H16/C256 | `../dqn/elf-c256/zkdrl-ctc-dqn-guest` |

Every host embeds the same A2C guest at `../a2c/elf/zkdrl-ctc-a2c-guest`.
At most one profile feature may be enabled; crossed combinations are
compile-time errors. The common DQN core itself supports all nine H/C
combinations, but the experiment uses only the five listed above.

SP1 SDK and verifier are pinned to **6.9.0**; the runtime circuit version is
**v6.1.0** (BN254). The host build keeps the SDK's `profiling` feature enabled
for the portable CPU executor. Guest build settings are fixed in the learner
workspaces: `opt-level=3`, fat LTO, one codegen unit, overflow checks off with
explicit arithmetic guards in the source.

## Trust boundary

`APPROVAL` is a trusted auditor/controller-provided JSON file, not prover
self-authorization. The parent owns registration, trusted delivery, ledger
checks, one-time challenge consumption, run/group association and approval
issuance. The host never creates an approval and cannot establish those
properties from an arbitrary local file.

Required approval fields are validated strictly: `schema` (exactly
`zkdrl-ctc-approval-v1`), nonblank `run_id`/`group_id`, `algorithm`,
`elf_sha256`, `vk_bytes32`, 32-byte `config_commit`/`challenge`, `kind`,
`episodes`, `scale`, `width`, and `capacity` (null for A2C). Unknown, duplicate,
missing, mistyped and out-of-range fields are rejected. Approval files are
limited to 16 KiB; inputs and proof bundles to 1 MiB.

## Operations

- `identity` performs a real CPU setup and reports the embedded ELF hash, actual
  program key, profile, SDK/circuit versions, verifier-key hash and timing.
- `native` runs the selected core natively and prints only its exact serde
  output; `--capture` adds audit data. Native outputs are for differential
  comparison, not proofs.
- `execute` runs the guest on the portable CPU executor and reports the decoded
  journal, actual canonical `journal_hex`, instruction cycles, syscall counts,
  exit code and timing. It never creates a proof.
- `prove` performs CUDA Groth16 proving after validating the approval, typed
  input, config commitment, challenge, task, budget and scale. It requires:

  ```sh
  SP1_PROVER=cuda
  CUDA_VISIBLE_DEVICES=0
  ZKDRL_APPROVED_GPU_UUID=<nonempty parent-approved GPU UUID>
  ```

  It writes exactly three artifacts with exclusive creation: the SDK proof
  bundle (`PROOF`), a canonical 356-byte Groth16 wrapper
  (`PROOF.wrapper.bin`), and the actual public journal bytes
  (`PROOF.journal.bin`).
- `verify` checks bounded canonical decoding, rejects trailing bytes and
  noncanonical re-encodings, requires Groth16 without TEE, circuit v6.1.0 and a
  356-byte wrapper, binds the actual journal to the embedded ELF key, applies
  the approval policy, and returns the computed journal.
- `tamper` accepts the honest proof, mutates every journal field at its
  cryptographic input, requires `ProofVerificationFailed`, and separately
  requires policy rejection of a changed challenge or config commitment.
- `audit` runs the original SDK `CpuProver::verify` API with the embedded key
  and default `SUCCESS`; it does not generate a proof.

The parent must independently compare the journal with the Python reference;
no operation accepts an expected full-journal file. Verification also does not
prove input provenance, auditor honesty, optimality, privacy, or that a
particular GPU produced the proof.
