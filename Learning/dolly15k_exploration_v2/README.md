
# Dolly-15k Generative Discovery (Pastel Edition)

This mini-package analyzes Dolly-15k and produces pastel-themed figures for README use.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# place databricks-dolly-15k.jsonl next to analyze.py, or pass --data
python analyze.py  # or: python analyze.py --data "/full/path/to/databricks-dolly-15k.jsonl"
```
Outputs go to `assets/`.
