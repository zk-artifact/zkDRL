# Third-party components

The artifact pins the following external components; their licenses remain in
effect.

- **SP1 / SP1 SDK and verifier 6.9.0**, circuit v6.1.0, BN254 Groth16.
  Upstream: <https://github.com/succinctlabs/sp1>.
- **Rust crates** resolved by six `Cargo.lock` files: `protocol/`, `dqn/`,
  `a2c/`, `host/`, `cpu-profile/`, and `cpu-profile/host/`. The DQN/A2C guests
  pin the SP1 SHA-256 patch at commit
  `75b15faa7ce44d25435d0a3209285455540f0b7f`.
- **MiniWoB++** (Farama Foundation), frozen commit
  `33c3b4ddef8c6eb67c57a29663d844b1eda7e614`; not redistributed.
- **KuaiRec** (Gao et al., CIKM '22; Zenodo 18164998), CC BY 4.0; the derived
  fixture and license notice are in `data/`.
- **NumPy** and **Matplotlib** for the Python harness and figures.

No vendor SDK binaries, CUDA runtime, proving keys, or SP1 public-parameter
archives are bundled. They must be obtained from their upstream distributions.
