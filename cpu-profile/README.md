# Standalone CPU cycle probe (nonformal)

This directory contains the separately labeled, **unproved** CPU instruction
profiling study for the schema-202 H16/C64 DQN. It is a marker/control
measurement, not a cryptographic proof and not part of the 66-proof cohort.
It is included because the paper's appendix reports CPU cycle breakdowns.

## What is measured

Two guest variants are built from the same DQN core:

- **control**: the frozen schema-202 H16/C64 guest, no markers;
- **instrumented**: the same source with marker-only insertion around the
  declared leaves (init/config/RNG key, slot/reset counters, random draws,
  policy/action, transition guards, replay insert/select, batch forward/target/
  backprop, master refresh/target copy, final encoding/commitment).

`source_layout.py` defines the insertion recipe. `checksource.py` verifies that
the copied common is byte-identical to the frozen `../dqn/common` after
stripping the prescribed markers and unwrapping the instrumented `Output`, and
that manifests, lock relations and the SHA-256 patch are unchanged. This is a
structural source comparison, not formal refinement or compiled equivalence.

The host uses SP1 SDK 6.9.0 with the `profiling` feature, which selects the
portable CPU executor; it calls `CpuProver::new()` and `.execute(...)` only. No
CUDA, proof, setup or registration entrypoint exists in this probe. Marker
emission is feature-gated and native marker calls are no-ops.

## Build

From the artifact root, with Rust/SP1 6.9.0 and `cargo prove` available:

```sh
cd cpu-profile
python3 -B checksource.py
python3 -B build.py
```

`build.py` refuses to overwrite an existing `build/`, `bin/`, `elf-control/` or
`elf-instrumented/`, keeps logs and commands, builds both guests and both host
variants with process-local `TMPDIR`/Cargo targets, and checks sources again
after the build.

## Collect and report

The full fixture must be the 24000-slot schema-202 exposure input created by
`prepare.py register` (4000 episodes). The reference must contain the Python
top-level `journal`, `final_state` and `coarse_export`.

```sh
python3 -B collect.py \
  ../runs/full-023-unified/fixtures/dqn-main_Q-exposure-s5000/input.json \
  ../runs/full-023-unified/fixtures/dqn-main_Q-exposure-s5000/reference.json
python3 -B report.py
```

`collect.py` first runs an 8-episode short input (only `slots` changed to 48),
then the full input. It executes native control/instrumented and actual guest
control/instrumented/original variants, requires all actual 148-byte journals
to match the independently encoded Python journal, and retains raw logs,
commands, hashes and complete native outputs under `results/`. `report.py`
writes `results/CPU_PROFILE.json` with per-leaf instruction counts, calibration,
residual, marker inflation and SHA/syscall checks.

## Interpretation

- Counts are RISC-V instruction counts from CPU execution, **not GPU proving
  time** and not precompile AIR rows.
- Marker inflation is disclosed and not subtracted as if it were free.
- Nested `total_run` is never double-counted; calibration is outside it.
- SDK syscall counts are AIR-mapped and are not a complete runtime census.
- No instruction count, proof, or equivalence claim is made by the mere
  presence of this directory; use the generated report for the measured values.

The paper's CPU profiling was run separately from the 41 cold gates and the
66 formal GPU proofs. This directory is intentionally not wired into
`prepare.py` or `run_numerics.py`.
