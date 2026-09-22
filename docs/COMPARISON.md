# Does the extra complexity earn its place?

A model can win an accuracy table and still give worse probabilities. This experiment compares three fixed choices on the same observations, then asks how much the observed difference depends on which test rows happened to arrive.

## The question and the contract

**Question:** On a small curved classification problem, does a shallow tree improve probability estimates over a linear decision boundary?

**Input:** 600 synthetic two-dimensional observations from `make_moons`, noise 0.25, seed 42. A stratified split reserves 150 rows for testing and 450 for training. The two features are coordinates; the binary target is the generating crescent. There are no patients or personal records.

**Output:** `outputs/comparison/predictions.csv` retains each held-out row ID, outcome and all three positive-class probabilities. `metrics.json` retains accuracy, Brier score, log loss, the paired interval and the experiment’s assumptions.

```bash
python -m pearlmind.lessons.comparison
```

This uses the core dependencies, a CPU and no external service. It is a statistical exercise, not a cloud workload.

## Three models, one fair comparison

| Choice | What it can learn | Why include it? | What it cannot establish |
| :-- | :-- | :-- | :-- |
| Prior predictor | Training-set class frequency; no feature relationship | Sets a minimum useful reference. Complexity should at least beat knowing the base rate | Whether any feature is informative |
| Scaled logistic regression | A linear log-odds boundary with L2 regularization | Tests whether a simple, inspectable boundary is sufficient | A curved boundary without transformed features |
| Shallow decision tree | Axis-aligned partitions, depth at most 4; at least 10 training rows per leaf | Tests a limited increase in flexibility | Smooth probabilities or stable behavior outside the sample |

Configurations are fixed before testing. There is no search across hundreds of candidates and no tuning against the held-out outcomes. The scaler is inside the logistic regression pipeline, so it learns its mean and variance from training rows only.

The prior is not a joke model. It is the question every elaborate model must answer: did the features help?

## Read the calculation before the leaderboard

For outcomes `yᵢ ∈ {0,1}` and positive-class probabilities `pᵢ`, the binary Brier score is:

```text
Brier = (1/n) Σᵢ (pᵢ − yᵢ)²
```

Smaller is better. If outcomes are `[0, 1]` and probabilities are `[0.2, 0.8]`, the score is `(0.04 + 0.04)/2 = 0.04`. Reversing the probabilities to `[0.8, 0.2]` gives `0.64`. Both predictions carry information about confidence that a correct/incorrect count discards.

Log loss penalizes confident errors more sharply. Accuracy answers a different question: after converting probabilities to decisions, how often did the predicted class match? Neither Brier score nor log loss isolates calibration; both also reflect discrimination and the outcome distribution. A reliability plot can investigate calibration, but small bins may be noisy. [Scikit-learn’s calibration guide](https://scikit-learn.org/stable/modules/calibration.html) explains the distinction.

## Why pairing matters

For each test row, form a loss difference:

```text
dᵢ = (p_tree,i − yᵢ)² − (p_linear,i − yᵢ)²
Δ = mean(d)
```

A negative Δ favors the tree for this metric and sample. Resample **row indices**, carrying both predictions and the outcome together. For each of 2,000 resamples, calculate the mean difference; the 2.5th and 97.5th percentiles form the displayed 95% percentile interval.

Resampling each model independently would discard the fact that both met the same easy and difficult cases. The implementation needs O(n) temporary memory per resample and O(Bn) work for B resamples. It retains only B means, rather than allocating a B-by-n matrix. This is deliberate restraint: more machinery would not improve the question.

This is a simple percentile bootstrap, not SciPy’s default BCa method. It assumes independent, representative test rows and holds the fitted models fixed. It does **not** include retraining variability or uncertainty introduced by choosing the model after seeing results. With tiny or degenerate samples, nominal 95% coverage can be poor. [SciPy’s bootstrap reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) documents pairing and interval alternatives.

## Follow the outputs, step by step

1. Run the command and open `metrics.json`. Check the seed and 450/150 split before comparing scores.
2. Open `predictions.csv`. Confirm that every candidate has a probability for the same `row_id` and `actual` outcome.
3. Recompute the Brier score from that file. `tests/test_comparison.py` independently checks it against scikit-learn.
4. Read `tree_minus_linear`. Does the interval include zero? Treat that as unresolved directional evidence under this design, not proof that the models are equivalent.
5. Compare accuracy with probability loss. A winner under one decision rule need not win under another.
6. Write down the next experiment before running it. Changing the seed repeatedly until a preferred result appears is model selection by another name.

## Common mistakes worth catching early

| Temptation | Why it misleads | Better next step |
| :-- | :-- | :-- |
| “The interval is below zero, so the tree is universally better” | The interval concerns this task, metric, sample and fixed training run | Replicate on an independently justified population and report all runs |
| Tune depth after reading the test scores | The test has become part of training decisions | Use training-only cross-validation; reserve a fresh final test |
| Bootstrap individual visits from the same patient | Repeated visits are dependent | Split and resample at the patient or other independent unit |
| Randomly split a forecasting dataset | Future patterns can leak into training | Use a time-respecting evaluation and consider block resampling |
| Call a probability a causal explanation | Prediction does not identify intervention effects | Specify causal assumptions and a suitable experimental design |
| Publish accuracy without the base rate | A majority predictor may already score highly | Keep the prior baseline and the confusion matrix |

## A measured extension

Use the original [decision-tree lesson](../Learning/02_decision_trees_to_forest.py) to inspect splits, then compare its assumptions with this experiment. For a larger study, predefine the task, independent sampling unit, loss and evaluation budget; use nested cross-validation if selecting configurations; keep a record of every candidate. Fit calibration on validation data, never on the final test. These extensions require additional implementation and validation—they are not implied by the current score.

[Concept dictionary](CONCEPTS.md) · [Input/output contracts](CONTRACTS.md) · [Comparison source](../src/pearlmind/lessons/comparison.py) · [DummyClassifier reference](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html)
