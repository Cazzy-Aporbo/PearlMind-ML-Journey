# Health-sequence study: inspect before interpreting

This is a research prototype. The tests use synthetic tensors, not patient data. They check shapes, finite outputs, forward-pass isolation and the absence of future-token leakage. They do not measure predictive value, calibrated uncertainty, clinical effectiveness or superiority to Delphi-2M.

```bash
python programs/delphi-2m/comparison_script.py
pytest tests/test_health_shapes.py
```

The inspection uses a reduced configuration so it runs on CPU. Tokens have shape `[2,4]`, ages use days with the same shape, and the disease head returns `[2,4,32]` scores. Optional biomarker/genetic tensors must align to each time step; inputs should contain only information available at that step. Survival-like outputs are constructed to be non-increasing, but monotonicity alone does not make them calibrated survival probabilities.

The repair makes graph bias pairwise over observed tokens, expands causal masks to the attention-head shape, retains an unmasked diagonal, and prevents forward passes from mutating a shared memory parameter. Modal fusion now attends across modalities at the same time step. Memory slots learn through optimizer updates; they are not a patient-history store.

Earlier numerical improvement estimates and clinical-readiness assertions are preserved in `docs/reference/health-comparison-original.py.txt` as unvalidated historical material. The current comparison script reports only what it measures. Before a scientific comparison, specify the external baseline, dataset rights, patient/time splits, outcome definitions, missing-data policy, uncertainty evaluation and a matched compute budget.

Training, inference and experimental loss modules require their own fuller evaluation. This shape repair does not establish that every research method is scientifically sound or ready for patient-facing use.
