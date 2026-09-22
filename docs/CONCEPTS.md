# A dictionary that leads somewhere

Use this as a translation between a question, its mathematical name and a piece of working code. Related words are not always interchangeable. The distinctions often matter more than the definitions.

[TOC]

## Data and experimental design

| Term / nearby language | Precise meaning | Follow it here | Easy confusion |
| :-- | :-- | :-- | :-- |
| Feature / predictor / covariate | An observed input represented by a column or tensor channel | [Tabular contract](CONTRACTS.md), `X` in the tabular lesson | A correlated input is not necessarily a cause |
| Target / label / outcome | The quantity the supervised task predicts | [Comparison](COMPARISON.md#the-question-and-the-contract) | A proxy label may miss the actual human objective |
| Sample / observation / row | One recorded example | `row_id` in exported predictions | A row need not be an independent person or event |
| Population / distribution | The wider process from which observations are drawn | [Readiness](READINESS.md) | A convenient dataset is not automatically representative |
| Train / validation / test | Fit parameters / make development choices / estimate final performance | [Language split](LEARNING_GUIDE.md#4-a-sequence-is-more-than-its-last-character) | A test set reused for tuning becomes validation data |
| Leakage | Information unavailable at intended prediction time enters fitting or selection | [Text windows](../Learning/cnn_rnn_api_demo/Readme.md) | A random seed cannot repair a leaking design |
| Baseline / reference model | A deliberately simple alternative used to judge added value | [Prior, linear, tree](COMPARISON.md#three-models-one-fair-comparison) | “Simple” does not mean disposable |
| Ablation | A controlled comparison with one component removed or changed | Try removing scaling from the logistic pipeline in a separate validation study | Comparing unrelated datasets is not an ablation |
| Covariate shift | The input distribution changes; assumptions about the conditional outcome mechanism still need checking | [Deployment questions](READINESS.md) | A changed input distribution alone does not establish concept drift |

## Optimization and tensor reasoning

| Term / nearby language | Precise meaning | Follow it here | Easy confusion |
| :-- | :-- | :-- | :-- |
| Parameter / weight | A quantity adjusted during fitting | [82-parameter network](LEARNING_GUIDE.md#2-a-neural-network-you-can-account-for) | Hyperparameters control the fitting process or model family |
| Gradient / derivative | Local sensitivity of a scalar objective to a parameter; the gradient collects partial derivatives | [Slope desk](https://cazzy-aporbo.github.io/PearlMind-ML-Journey/#gradient) | A gradient indicates a local direction, not a guarantee of the global optimum |
| Loss / objective | The numerical criterion optimized during training | [Linear regression](../Learning/01_linear_regression.py) | A lower loss need not mean a better real-world decision |
| Learning rate / step size | Multiplier on an update direction | [First lesson](LEARNING_GUIDE.md#1-a-line-learns-a-slope) | Larger steps can diverge rather than accelerate |
| Tensor / shape / axis | An array and the interpretation of each dimension | [PyTorch source](../src/pearlmind/lessons/torch_lab.py) | Equal element counts do not make layouts semantically interchangeable |
| Logit / probability | An unnormalized score / a normalized value in [0,1] | The final `Linear` layer before cross-entropy in the PyTorch lesson | Passing softmax output into a loss expecting logits changes the calculation |
| Backpropagation / autograd | Chain-rule computation / software that constructs derivatives | `loss.backward()` in [PyTorch](../src/pearlmind/lessons/torch_lab.py) | Calculating gradients does not itself update parameters |
| Regularization | A constraint or penalty that limits fitted complexity | L2 penalty in the [comparison](COMPARISON.md) | It does not repair unrepresentative data or label errors |
| Causal mask | A computational restriction preventing access to later sequence positions | [Sequence models](../Learning/cnn_rnn_api_demo/Readme.md) | “Causal” here is not a claim of causal inference |
| Receptive field | The input positions that can influence an output | Causal convolution in the language lab | More layers do not guarantee useful long-range memory |

## Measurement and uncertainty

| Term / nearby language | Precise meaning | Follow it here | Easy confusion |
| :-- | :-- | :-- | :-- |
| Accuracy | Correct class decisions divided by evaluated observations | [Threshold desk](https://cazzy-aporbo.github.io/PearlMind-ML-Journey/#threshold) | Class imbalance can make accuracy reassuring and unhelpful |
| Precision / positive predictive value | TP/(TP+FP) | [Metric implementation](../src/pearlmind/evaluation/fairness.py) | Undefined when there are no predicted positives |
| Recall / sensitivity / true-positive rate | TP/(TP+FN) | [Error allocation](LEARNING_GUIDE.md#5-a-threshold-allocates-errors) | Precision and recall have different denominators |
| Specificity / true-negative rate | TN/(TN+FP) | Derive it from the exported confusion matrix | Not the same as the fraction of negative predictions that are correct |
| Brier score | Mean squared error of binary probabilities against 0/1 outcomes | [Worked arithmetic](COMPARISON.md#read-the-calculation-before-the-leaderboard) | It reflects more than calibration alone |
| Log loss / cross-entropy | Negative log probability assigned to observed labels, averaged over examples | [Comparison](COMPARISON.md) | Log loss is not bounded above; confident errors can be expensive |
| Calibration / reliability | Agreement between predicted probabilities and observed frequencies | [Calibration reference](https://scikit-learn.org/stable/modules/calibration.html) | Good ranking can coexist with poor calibration |
| Discrimination / ranking | Ability to separate outcomes by score | Contrast score order with the threshold desk | A ranking is not a calibrated probability |
| Paired bootstrap | Resample matched observations together to estimate sampling variability | [Runnable method](COMPARISON.md#why-pairing-matters) | An interval conditional on fitted models omits retraining uncertainty |
| Confidence interval | An interval procedure with a specified repeated-sampling coverage target | [Interpretation and limits](COMPARISON.md#follow-the-outputs-step-by-step) | It is not a posterior probability that this fixed parameter lies inside this realized interval |
| Group disparity | A measured difference in a chosen outcome or error rate across groups | [Fairness code](../src/pearlmind/evaluation/fairness.py) | One parity metric cannot certify fairness |
| Counterexample | A case that falsifies a proposed universal claim | [Comparison tests](../tests/test_comparison.py) | Passing many examples is not a proof for every input |

## Systems and decision boundaries

| Term / nearby language | Precise meaning | Follow it here | Easy confusion |
| :-- | :-- | :-- | :-- |
| Weighted aggregation | Combine local statistics according to their contribution to the intended global objective | [Worker lesson](LEARNING_GUIDE.md#6-many-workers-one-update) | Equal weighting of unequal shards generally changes the objective |
| Invariant / contract | A condition that must remain true / an explicit expectation at an interface | [Contracts](CONTRACTS.md) | A type annotation alone does not enforce runtime validity |
| Determinism / reproducibility | Same specified conditions produce the same result / enough recorded context to repeat an experiment | Seeds, versions and commit in `data/evidence.json` | A seed alone does not control every device or library behavior |
| State / persistence | Information carried across operations / retained beyond a process | Native model JSON and saved scaler statistics | Saving weights without preprocessing may save the wrong predictor |
| Tool routing / bounded execution | Selecting permitted actions under explicit conditions and a finite budget | [Systems source](../src/pearlmind/lessons/systems.py) | A deterministic demonstration is not an autonomous deployed agent |
| Provenance / lineage | Where an artifact came from and the transformations that produced it | [Change record](../ChangeLog.md), per-build evidence | A polished graph is not its own evidence |

## From the original atlas to an executable question

The [original model atlas](original-model-atlas.md) remains the broad map. These routes connect its vocabulary to a practical next step without implying that every family has a complete implementation.

| Earlier topic | A concrete question | Current route | Further work |
| :-- | :-- | :-- | :-- |
| Regression and optimization | How does the sign of a derivative change a fitted slope? | [Original regression lesson](../Learning/01_linear_regression.py) → gradient desk | Conditioning, regularization paths and residual diagnostics |
| Tree ensembles | Does flexibility improve held-out probability loss? | [XGBoost lesson](../src/pearlmind/lessons/tabular.py) → [paired comparison](COMPARISON.md) | Nested selection, calibration and shift testing |
| Deep learning | Where do all 82 parameters appear in the forward pass? | [Tensor walkthrough](LEARNING_GUIDE.md#2-a-neural-network-you-can-account-for) | Capacity studies and learning curves |
| NLP and sequence models | Can changing a future token alter an earlier output? | LSTM/TCN lab and its causality tests | Larger corpora, tokenizer studies and representative evaluation |
| Responsible AI | What happens to errors when a decision threshold moves? | Threshold desk and group metrics | Context-specific harm analysis and stakeholder review |
| Federated or distributed learning | When does a weighted local gradient equal the full-batch gradient? | [Systems arithmetic](LEARNING_GUIDE.md#6-many-workers-one-update) | Real communication, failure recovery, privacy and asynchronous updates |
| RAG, transformers, quantum-inspired and health studies | Which claim can be tested independently before expanding? | Original atlas and source views marked extended study | Module-specific data, dependencies and scientific validation |

[Return to the learning guide](LEARNING_GUIDE.md) · [Read the change record](../ChangeLog.md)
