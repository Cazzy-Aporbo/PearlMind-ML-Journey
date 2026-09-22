# Useful experiments, explicit boundaries

PearlMind is an educational repository. “It runs” is a beginning, not a deployment assessment. Core experiments are deliberately small so a learner can inspect every output.

## Before adapting a model

| Question | What the core path supplies | What a real deployment still needs |
| :-- | :-- | :-- |
| Can we reproduce the run? | Fixed seeds, config, saved predictions, dependency environment in CI | Data/version governance, pinned deployment dependencies, repeatability across hardware |
| Is evaluation independent? | Held-out rows; train-only scaling; separated text windows | Validation design, temporal/group tests, external samples, confidence intervals |
| Is a model safe to load? | Native XGBoost JSON; PyTorch state dictionaries with weights-only loading | Trusted artifact origin, signatures, review of dependencies and model behavior |
| Does the service reject bad inputs? | Shape/finite checks, bounded rows, real-label audit requirement | Authentication, authorization, rate and body limits, observability, incident response |
| Are decisions fair? | Confusion matrix, sample counts and group-rate gaps | Appropriate labels, consent, context, uncertainty, stakeholder review and monitoring |
| Can a system act autonomously? | A bounded, allowlisted teaching state machine | Tool authority, isolation, adversarial tests, human escalation and rollback |
| Does a health model help patients? | Research code to inspect | Suitable data rights, clinical study design, regulatory assessment and independent validation |

## Model cards in brief

**Synthetic XGBoost classifier:** binary educational task; numeric features; seeded 480/120 split. It illustrates persistence and evaluation, not any commercial use case. Tree importance is associative. A changed feature order can invalidate inference.

**Two-moons neural network:** 82 trainable parameters; 400 training and 100 test examples; CPU. Two logits and cross entropy illustrate supervised learning. The synthetic geometry is deliberately simple. Its accuracy has no direct meaning in healthcare, finance or governance.

**Character LSTM / TCN:** next-character modeling on a local corpus. Data source, rights and representativeness must be reviewed for any replacement corpus. The bundled corpus supports teaching, not a general-purpose language product. Sampling is stochastic; generated text can be incoherent or unsuitable.

**Experimental health, temporal and quantum-themed studies:** preserved as work to examine and challenge. Names and mathematical imagery are not evidence of biological validity, quantum advantage, forecasting accuracy or clinical utility. The file atlas marks these as extended studies; current CI does not train or validate every such model. The health-sequence prototype now has synthetic forward/backward, finite-output, causality and state-isolation tests; those tests are explicitly not clinical evidence.

## Compute, cost and growth

The core path has no paid API calls and needs no GPU. A modern laptop can run the small experiments; the PyTorch installation is much larger than the model itself. Allow several GB for the environment. Runtime measurements in the evidence artifact describe that runner, not a service-level promise.

Codespaces and Actions may consume account quotas; check your GitHub plan. Optional tracking, hosted inference, larger datasets and GPUs create separate costs. Estimate training as GPU-hours × provider rate, storage as retained GB-months, and inference as measured requests × cost/request. Add retries, egress, observability and human review. No universal enterprise cost is inferred from these toy runs.

The historical broad extras remain for reference. TensorFlow, JAX and legacy integrations should be installed in separate environments after checking their compatibility. Installing `[all]` is not the recommended learning path.

## Reading the checks

CI tests core contracts and failure cases, runs the CPU lessons, builds and installs the wheel, checks Python syntax across the repository, and verifies the static site’s local links. Coverage is reported honestly. The earlier blanket 90% gate was attached to missing modules; the repaired suite uses an 85% package-coverage baseline and named end-to-end checks, not fabricated percentages.

Public Pages contains source learning material and synthetic metrics. Keep credentials, personal data, private datasets and model weights from confidential work out of this repository.
