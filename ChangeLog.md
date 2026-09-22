# Change record

PearlMind is Cazandra Aporbo’s open learning project. This record links changes to code and checks; the [commit history](https://github.com/Cazzy-Aporbo/PearlMind-ML-Journey/commits/main/) retains authorship and timing. The [earlier ledger](docs/reference/earlier-changelog.md) is preserved as historical material, not a verified release history.

## 21 September 2026 — executable lessons and a connected learning room

| Area | What changed | Where to check it |
| :-- | :-- | :-- |
| Installation | Repaired package discovery, missing modules and optional dependency groups; added Codespaces and a non-root container | [Setup](docs/SETUP.md), `pyproject.toml`, `.devcontainer/`, `Dockerfile` |
| Training and serving | Replaced placeholder CLI results with executed training, native model persistence and validated prediction inputs | `src/pearlmind/cli/`, `models/`, `deployment/`; CLI/API tests |
| Evaluation | Retained undefined rates as unknown; added held-out row exports, prior/linear/tree comparison and paired bootstrap uncertainty | [Comparison method](docs/COMPARISON.md), `tests/test_comparison.py` |
| PyTorch | Added a small CPU neural experiment with saved preprocessing, weights and loss curve | `src/pearlmind/lessons/torch_lab.py` |
| Language | Split text before overlapping windows; repaired tied embeddings, full-prompt generation and checkpoint loading | `Learning/cnn_rnn_api_demo/`, language tests |
| Health sequence study | Repaired pairwise graph dimensions, causal attention and state isolation; replaced unmeasured improvements with shape/parameter inspection | `programs/delphi-2m/`, synthetic shape and causality tests |
| Learning design | Connected original lessons to walkthroughs, contracts, a terminology map and searchable source views | [Learning guide](docs/LEARNING_GUIDE.md), [concept dictionary](docs/CONCEPTS.md) |
| Delivery | Replaced unrelated Django steps with Python 3.11/3.12 tests, wheel installation, measured artifacts and a checked Pages deployment | [Actions](https://github.com/Cazzy-Aporbo/PearlMind-ML-Journey/actions) |
| Presentation | New README artwork, responsive experiments, share image, descriptive page metadata and crawlable continuation links | [Learning room](https://cazzy-aporbo.github.io/PearlMind-ML-Journey/) |

The first repair is recorded in [9f3e487](https://github.com/Cazzy-Aporbo/PearlMind-ML-Journey/commit/9f3e487), health checks in [3aed010](https://github.com/Cazzy-Aporbo/PearlMind-ML-Journey/commit/3aed010), and setup/disclosure corrections in [1573ce7](https://github.com/Cazzy-Aporbo/PearlMind-ML-Journey/commit/1573ce7). Subsequent entries are traceable in the history above.

## Reading the evidence

Each site build records its commit, UTC timestamp, dependency versions, seeds and measured outputs in `data/evidence.json`. CI retains the original output files as downloadable artifacts. A green build verifies the tested contracts; it does not validate every research hypothesis or establish clinical effectiveness.
