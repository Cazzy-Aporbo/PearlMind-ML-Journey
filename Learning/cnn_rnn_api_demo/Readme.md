
# CNN/RNN Language Models + Retained Model API (Shakespeare Demo)

This runnable project shows advanced CNN (dilated causal **TCN**) and **RNN (LSTM)** text models,
versioned **retained checkpoints**, and a FastAPI service that **auto-loads the latest retained model**.

## Setup

```bash
cd cnn_rnn_api_demo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Put training text at `data/shakespeare.txt` (this repo already copied your uploaded file if present).

## Train

```bash
# LSTM
python src/train_lm.py --model rnn --epochs 2 --seq-len 256 --batch-size 64

# TCN (dilated causal CNN)
python src/train_lm.py --model tcn --epochs 2 --seq-len 256 --batch-size 64 --hidden 384 --layers 6
```

Each best checkpoint is saved under `models/<type>/vYYYY-MM-DD_HH-MM-SS/` and is **registered** in `models/<type>/manifest.json`.

## Serve latest retained model

```bash
# Default serves latest RNN
uvicorn src.serve:app --reload --port 8080

# Serve TCN
MODEL_TYPE=tcn uvicorn src.serve:app --reload --port 8080
```

## Generate via API

```bash
curl -X POST http://127.0.0.1:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"To be, or not to be","max_new_tokens":150}'
```
