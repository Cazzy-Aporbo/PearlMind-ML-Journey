# A small text model with something to inspect

This lab compares a character LSTM and a causal temporal convolutional network. Neither downloads a pretrained model or calls a paid API. Start from the repository root with the learning environment installed.

```bash
python Learning/cnn_rnn_api_demo/src/train_lm.py \
  --data-path Learning/cnn_rnn_api_demo/data/shakespeare.txt \
  --model rnn --epochs 1 --seq-len 64 --batch-size 32 \
  --emb 32 --hidden 32 --layers 1 --sample-after
```

For the convolutional comparison, change `--model rnn` to `--model tcn`. Start with a small corpus subset if your computer is slow; the bundled full text contains many overlapping windows.

**Input:** local UTF-8 text. Raw text is split 90/10 before windows are created. Each input has shape `[batch, sequence]` and integer character IDs; targets shift by one character. Vocabulary comes from training text, and unseen validation characters raise an explicit error.

**Output:** `models/<rnn-or-tcn>/<version>/checkpoint.pt`, `meta.json`, and a parent `manifest.json` containing arguments, timestamp and validation loss. Model dimensions are in the manifest; use matching values when serving. Generated text is a stochastic demonstration, not a quality guarantee.

Read [the tensor and causality walkthrough](../../docs/LEARNING_GUIDE.md#4-a-sequence-is-more-than-its-last-character). Tests cover forward/backward shapes, future-token isolation, tied weights and consumption of the full prompt. The local FastAPI sample bounds generation requests; it is not an authenticated public inference service.

The [earlier tutorial](../../docs/reference/language-lab-original.md) is preserved for context. Use the commands and contracts here when they differ.
