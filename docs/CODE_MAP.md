# Paper-to-code map

| Paper component | Code |
| --- | --- |
| §4.1 guest loop and Web task adapters | `dqn/common/src/lib.rs`, `a2c/common/src/lib.rs`, `python/legacy/model.py` |
| MiniWoB++ Boolean login abstraction | `python/legacy/model.py` (`kind == 'browser'`), `python/dqn_reference.py`, `python/a2c_reference.py` |
| KuaiRec four-item exposure | `python/legacy/model.py` (`kind == 'recommendation'`), `python/legacy/data/recommendation.npz` for legacy evaluator compatibility |
| §4.2 unified counter RNG and tapes | `protocol/src/lib.rs`, `python/random_tape.py` |
| Dual-scale fixed point / signed flooring | `dqn/common/src/lib.rs`, `python/legacy/qat_model.py`, `python/legacy/a2c_fixed/fixed_math.py` |
| Straight-through backward rules | `dqn/common/src/lib.rs`, `a2c/common/src/lib.rs`, `python/legacy/qat_model.py`, `python/legacy/a2c_fixed/fixed_math.py` |
| §4.3 DQN ring replay, stopped targets, fine masters | `dqn/common/src/lib.rs`, `dqn/common/src/tests.rs` |
| §4.3 episodic A2C, frozen rollouts, exact masses, atomic update | `a2c/common/src/lib.rs`, `a2c/common/src/tests.rs`, `a2c/common/src/exp_table.rs` |
| §4.4 commitments, journal ABI, proof wrapping | `host/src/proof.rs`, `host/src/journal.rs`, `host/src/approval.rs`, `host/src/artifact.rs`, `protocol/IMPLEMENTATION.md` |
| Guest execution / native differential | `host/src/main.rs`, `host/src/input.rs`, `host/src/registry.rs`, `host/src/cli.rs` |
| §5 threat model and claim boundaries | `host/README.md`, `docs/CLAIMS.md` |
| §6.1 unified matrix | `design.py` |
| Pre-registration and challenges | `auditor.py`, `prepare.py` |
| 41 cold CPU gates | `prepare.py` (`validate`, `worker`) |
| §6.2 binding / tamper checks | `host` modes `verify`, `tamper`, `audit`; `acceptance.py` |
| §6.3 proving/memory/scaling measurements | `host/src/proof.rs`, `host/src/main.rs`, `reported-results/proof-configuration-summary.csv` |
| §6.4 numerical study | `numerics/runner.py`, `python/legacy/`, `design.py`, `run_numerics.py` |
| Numerical tables/figures | `analyze_numerics.py`, `plot_results.py` |
| Appendix CPU cycle breakdown (nonformal probe) | `cpu-profile/`, `cpu-profile/README.md` |
| Appendix arithmetic bounds and tapes | `protocol/IMPLEMENTATION.md`, `python/legacy/*/fixed_math.py`, `protocol/src/tests.rs` |

## End-to-end data flow

```text
fixture (public seed/salts)
  -> config_bytes (typed, canonical)
  -> kappa = SHA256(config_bytes)
  -> auditor challenge chi
  -> PrivateInput = fixture + chi
  -> Python reference -> reference journal + final state
  -> guest execution -> actual journal bytes
  -> prove(kappa, chi, program) -> Groth16 bundle + journal
  -> verify(approval) -> computed journal
  -> compare with reference journal byte-for-byte
```

The host never mints an approval and never accepts an expected journal as an
input to verification. The auditor ledger and trusted delivery path are outside
the cryptographic statement; see `host/README.md`.
