# Bring a question we can test

A useful contribution makes one idea easier to understand or one failure easier to see. Begin with the smallest reproducible input, the output you expected, and what actually happened.

1. Create a branch and install `.[ci,api,learning]` in a virtual environment.
2. Add the explanation alongside the code. Include shapes, units, assumptions and a counterexample when they matter.
3. Test the behavior: `pytest`, `python scripts/check_source.py`, `python scripts/run_evidence.py`, `python scripts/build_site.py`, `python scripts/check_site.py`.
4. Keep public examples synthetic or appropriately licensed. Never commit credentials or personal health data.
5. Open a pull request describing the evidence and the limits. Do not infer clinical, legal or production readiness from a toy benchmark.

Preserve earlier learning material unless there is a specific reason to remove it. If an example moves, keep a clear pointer. New files should have a place in the source atlas and an explanation of how to run them.

Please follow [the code of conduct](Code_of_Conduct.md). Accessibility, clearer error messages and thoughtful documentation count as engineering work here.
