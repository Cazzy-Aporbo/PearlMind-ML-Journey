# A small environment, on purpose

Use Python 3.11 or 3.12. Clone the repository, create a virtual environment, then install `python -m pip install -e ".[ci,api,learning]"`. Start with `pearlmind train configs/default.yaml` and `pytest`.

## Codespaces

Choose **Code → Codespaces → Create codespace**. The included devcontainer installs the learning environment, then leaves training under your control. Port 8000 is labeled for the local teaching API and 8080 for the learning site; keep forwarded ports private. Codespaces usage follows your GitHub billing plan.

## Local learning site

```bash
python scripts/run_evidence.py
python scripts/build_site.py
python -m http.server 8080 --directory site --bind 127.0.0.1
```

Open `http://127.0.0.1:8080`. The build copies source into readable reference pages, generates the file atlas, and adds measured results from the evidence run. No backend or secret is needed. The CI Pages workflow performs the same steps.

## Troubleshooting

- **macOS: XGBoost cannot load libomp.** Install the official Homebrew `libomp` package, then retry. Do not replace the model with a stub just to get a green check. Some PyTorch installations also bundle this library; the CI runs on Ubuntu with `libgomp1`.
- **No module named pearlmind.** Activate the environment and install the editable package from the repository root. Avoid running commands from a different Python installation.
- **Unseen character in the text lab.** The vocabulary is learned from training text only. Review your corpus or deliberately add an unknown-token policy; silently deleting characters changes the task.
- **Missing optional framework.** Install only the extra for that experiment, in a separate environment where needed. The core path does not require every framework in the historical atlas.
- **A score changed.** Compare seeds, library versions, data and hardware first. Do not edit the expected result to conceal a behavioral regression.

## Packaging and releases

CI builds a wheel and installs it into a fresh environment. The manual release workflow creates wheel/source artifacts; it does not publish to PyPI. Configure a separately reviewed trusted-publisher release process if public package distribution becomes appropriate.
