# Make a prediction. Keep the evidence.

A learning path by Cazandra Aporbo. You do not need to understand every file before beginning. Each stop below makes one connection: a question becomes data, data becomes a calculation, and the calculation meets a decision.

## Three doors into the same idea

**First encounter:** a model is a rule adjusted using examples. A *feature* is something observed; a *target* is the outcome we want to predict. Try the browser desks and describe what changed without using technical vocabulary.

**Building fluency:** run the command, inspect each artifact, and change one parameter. Explain its effect using a test, a plot or a counterexample.

**Deeper inquiry:** derive the update, identify assumptions, and design an evaluation that could disprove your interpretation. Faster training and higher accuracy are not interchangeable objectives.

## 1. A line learns a slope

For a deliberately small model, predict `ŷ = wx` with no intercept. Use `x = [1,2,3]`, `y = [2,4,6]`. Mean squared error is `L = Σ(wx−y)² / n`; the gradient is `dL/dw = 2Σx(wx−y) / n`.

At `w=0`, the loss is `56/3 ≈ 18.667`, and the gradient is `−56/3`. With learning rate `0.05`, the new weight is `0.9333`. The prediction moves closer to the observations. Try `0.3` and watch the same rule diverge: for this quadratic, convergence requires a rate below `3/14 ≈ 0.2143`.

**Predict → run → explain:** sketch the curve first, take a step in the [browser desk](https://cazzy-aporbo.github.io/PearlMind-ML-Journey/#gradient), then inspect [the original Python lesson](../Learning/01_linear_regression.py). That lesson includes a bias term and richer synthetic examples; the browser isolates one parameter so the derivative is visible.

**Common mistakes:** confusing a prediction error with the average loss; forgetting the factor `1/n`; interpreting training fit as future performance. Units matter: squared dollars and squared centimetres are different objectives.

**Extension:** add noise, fit an intercept, compare a closed-form solution with gradient descent. Why can zero initialization work for a linear model but identical initialization be a problem for a multilayer neural network?

## 2. A neural network you can account for

Run `python -m pearlmind.lessons.torch_lab --epochs 80`. The data are two synthetic curved groups, not observations about people. The network learns a boundary between them.

| Stage | Shape / type | Why it exists |
| :-- | :-- | :-- |
| Training features | `[400, 2]`, float32 | Two measurements for each training example |
| Hidden transformation | `[400, 16]` | Learn sixteen intermediate responses with `tanh` |
| Class logits | `[400, 2]` | Two unnormalized scores per example |
| Labels | `[400]`, int64 | The correct class index, 0 or 1 |
| Cross entropy | scalar | Compare logits with observed classes |
| Test logits | `[100, 2]` | Evaluate rows the optimizer has not trained on |

There are **82 parameters**: `2×16+16 + 16×2+2`. Parameters are learned values; hyperparameters, such as width and learning rate, are choices made around learning.

Read the loop in [torch_lab.py](../src/pearlmind/lessons/torch_lab.py): `zero_grad()` clears accumulated gradients; `backward()` differentiates the loss; `step()` changes parameters. `eval()` changes layers with train/eval behavior, while `no_grad()` stops gradient recording. They solve different problems even though this tiny network has no dropout or batch normalization.

Inspect `weights.pt`, `preprocessing.json`, `metrics.json` and `loss.png`. To reuse the model, instantiate the same network, load the state dictionary with `weights_only=True`, and apply `(X−mean)/scale` using the saved training statistics. Changing the architecture invalidates the weight contract.

**Common mistakes:** applying softmax before `CrossEntropyLoss`; fitting the scaler on the test set; comparing logits directly with class labels; leaving gradients accumulated accidentally. Do not tune repeated decisions against the held-out test result; create a validation split for that work.

**Extension:** compare three seeds and add an independent validation split. Report the distribution of results, not just the best run. Try a linear baseline first—complexity earns its place through evidence.

## 3. A table becomes a reusable model

Run `pearlmind train configs/default.yaml`. The default is 600 synthetic rows with eight numeric features. A stratified split retains 120 test rows. The group labels are random synthetic labels for demonstrating metric calculations; they say nothing about real demographic outcomes.

Open the native XGBoost JSON model beside its JSON metadata, held-out predictions and feature-importance CSV. Importance summarizes the fitted model; it does not prove a causal relationship. Correlated features can redistribute importance.

For your own CSV, remove identifiers and post-outcome fields before training. Repeated observations from the same person need a group-aware split; forecasts need a temporal split. The convenient random split here is an assumption you must review.

**Counterexample:** if a loan dataset includes a field created after repayment, a near-perfect score might be leakage. Ask when each feature became available, not merely whether it correlates with the target.

## 4. A sequence is more than its last character

The [language lab](../Learning/cnn_rnn_api_demo/Readme.md) compares an LSTM with a temporal convolutional network. Both return `[batch, time, vocabulary]` logits. Inputs and targets shift by one character: `abcd → bcde`.

An LSTM passes a hidden and cell state forward. A causal convolution uses a finite left-hand context with dilation. “Causal” here means no future token enters a prediction; it is not a claim about real-world cause and effect.

Split raw text before constructing windows. Train vocabulary on training text, then detect characters the vocabulary cannot represent. This implementation raises an error instead of silently dropping them. A production tokenizer needs an explicit unknown-token or byte-level policy.

**Inspect:** tests change future input tokens and verify earlier logits do not change. Generation first consumes the full prompt, then carries the LSTM state forward. The TCN recomputes the growing context; understand that cost before scaling it.

**Extension:** calculate the receptive field of stacked dilated convolutions. Compare per-token validation loss at similar parameter counts. Generated fluency alone is not an evaluation.

## 5. A threshold allocates errors

A probability does not decide what to do. A threshold turns a score into an action. The [threshold desk](https://cazzy-aporbo.github.io/PearlMind-ML-Journey/#threshold) uses twelve explicitly synthetic observations.

Precision is `TP/(TP+FP)`; recall is `TP/(TP+FN)`. A missing denominator means undefined, not zero. Group selection-rate differences and TPR/FPR gaps describe a sample; they do not certify fairness. Uncertainty, group sizes, labels, intervention effects and context still matter.

**Exercise:** raise the cost of a false negative. Which threshold minimizes the chosen sample cost? Would that choice remain acceptable if one group had systematically less reliable labels? Distinguish a calculation from a justified policy.

## 6. Many workers, one update?

Run `python -m pearlmind.lessons.systems`. Two workers hold two and three observations. Each computes a mean gradient at the same weights. Combine them as `(2g₁ + 3g₂)/5` to match the full-batch gradient. An equal average gives the smaller shard too much influence.

This is a **single-process mathematical simulation**. Real distributed training adds synchronization, network failures, collective communication, numerical differences and stragglers. Multiple local optimizer steps are not generally equivalent to one global step. Try unequal class distributions and stale weights as follow-up experiments.

The same module contains a bounded tool-routing state machine. Read-only inspection can complete; publishing needs review; unsupported requests defer. It does not use an LLM or deploy anything. The lesson is about explicit authority and termination before adding a probabilistic planner.

## Sources for the next question

- [PyTorch: complete training workflow](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
- [scikit-learn: common pitfalls and leakage](https://scikit-learn.org/stable/common_pitfalls.html)
- [XGBoost Python documentation](https://xgboost.readthedocs.io/en/stable/python/python_api.html)
- [PyTorch distributed overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html)

These sources explain library behavior. This repository’s tests and artifacts show what its own examples actually did.
