# Claim boundaries

This file summarizes what the artifact's proofs and experiments do and do not
establish. It follows the paper's security analysis and evaluation scope.

## Established by a successful proof

- There exists a complete execution of the approved guest program on the
  committed private input and issued challenge that produces the returned
  public journal.
- The journal is bound to the configuration commitment, challenge, program key,
  initial/final commitments and the guest's canonical state transition.
- The proof verifies under the embedded verifier key and the pinned circuit
  version (v6.1.0, Groth16, BN254).
- The host's `tamper` checks reject mutations of every public field, wrong keys,
  wrong nonces/challenges and noncanonical encodings at the cryptographic stage.
- The host's `audit` path accepts the honest proof with the original SDK
  verifier.

## Not established

- **Privacy.** The experiment uses public test fixtures. The proof reveals the
  journal and whatever the verifier is given; it does not hide the witness from
  the prover or make repeated journal queries private.
- **Optimality or utility.** A valid proof says the training run was executed,
  not that the learned policy is good, fair, or better than a baseline.
- **Input provenance.** Initial weights and task parameters are committed, but
  their origin and legitimacy are outside the proof.
- **Auditor/OS honesty.** The local ledger and trusted delivery path are
  assumed; there is no external transparency log, smart contract, or
  compulsory-delivery mechanism.
- **Universal anti-cherry-picking.** The campaign pre-registers conditions and
  retains failures, but it is not a public multi-party commitment scheme.
- **Generality beyond the implemented profiles.** The formal host registry
  covers five DQN profiles plus A2C; the common core supports more combinations
  but those are not claimed as evaluated.
- **Production readiness.** The paper lists deployment mechanisms for input
  provenance, private evaluation and repeated-run leakage; they are not part of
  this artifact.

## Statistical scope

The numerical study has five pre-declared seeds per algorithm and task. The
paper reports observed minima/maxima and paired descriptive comparisons, not
population confidence intervals or significance tests. The 22 proof
configurations are not 66 independent learning samples: each configuration has
three timing repetitions on the same committed initialization and challenge.
Scaling points are prefix/width/capacity probes, not new training seeds.

## Numerical scope

DQN compares FP64 (F), coarse integer (N), mixed QAT (Q), a residual-only
ablation, and S256-Q. A2C compares FP64, Q16 commit (N), Q40 master (Q), a Q20
forward probe, and two exp/softmax operator variants. Backward rules are
declared straight-through estimators, not exact derivatives of the quantized
forward map. Guards abort a run on arithmetic-domain violations; failed
evaluations are reported as unavailable, never imputed as zero.
