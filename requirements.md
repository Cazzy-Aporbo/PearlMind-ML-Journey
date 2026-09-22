# Choose an environment for the question

The installable dependency source is [pyproject.toml](pyproject.toml). Use `python -m pip install -e ".[ci,api,learning]"` for the tested learning path. See [setup](docs/SETUP.md) for Codespaces and troubleshooting.

The earlier multi-file requirements sketch is retained in [the reference archive](docs/reference/original-requirements.md). It described files that did not exist; do not use it as an installation checklist. Optional framework extras remain available for separate experimental environments and are not jointly tested as an `[all]` installation.
