# Data provenance and licenses

The artifact does **not** redistribute the raw MiniWoB++ or KuaiRec archives.
The training fixtures are generated deterministically by
`python/dqn_reference.py` and `python/a2c_reference.py` from public seeds, fixed
reward parameters and the documented task abstractions. The only data file
included is the small derived `recommendation.npz` (at `data/` and, for legacy
path compatibility, `python/legacy/data/`).

## MiniWoB++

- Upstream: <https://github.com/Farama-Foundation/miniwob-plusplus>
- Frozen commit: `33c3b4ddef8c6eb67c57a29663d844b1eda7e614`
- Archive SHA-256: `2290178d6159bbd10a9dabcc49c6a739341bc5a16cdb5f29a3f6abf035aa362c`
- The experiment uses the `login-user.html` Boolean workflow abstraction
  described in the paper; it does not ship browser traces or raw HTML.

## KuaiRec

- Record: Zenodo 18164998, DOI `10.5281/zenodo.18164998`
- Archive: `KuaiRec.zip`, SHA-256
  `c23f211cf109c7a060a25a4afe9e8c681b1ccbbc816cca921d87ad37dd11c70c`
- License: CC BY 4.0; the bundled notice is `data/KuaiRec-LICENSE`.
- Citation: Gao, Li, Lei, Chen, Li, Jiang, He, Mao, and Chua. *KuaiRec: A
  Fully-Observed Dataset and Insights for Evaluating Recommender Systems.*
  CIKM '22, pp. 540–550. DOI `10.1145/3511808.3557220`.
- `recommendation.npz` is a derived 32-user × 4-item matrix:
  mean per-observation watch ratio clipped to `[0,2]`, first 24 complete users
  for train/development and the last 8 for evaluation, sorted IDs. It is not a
  representative sample and is used only for the external evaluation described
  in the appendix.
- Derived-file SHA-256:
  `258d96c186b4061842552dbd8028ecfd768c2e05329d4fb5351b07c1c87ecc91`

## Fixtures and secrecy

Public salts, seeds and challenges in the shipped fixtures are test values,
not confidential inputs. The paper's production privacy claims are explicitly
limited; see `README.md`. Do not reuse the public fixture salts for a
deployment that expects witness privacy.
