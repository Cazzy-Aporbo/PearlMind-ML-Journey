# src/serve.py
import os, json, glob
from typing import Optional
from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from models import RNNLanguageModel, TCNLanguageModel
from tokenizer import CharTokenizer


@asynccontextmanager
async def lifespan(app):
    _load()
    yield


app = FastAPI(title="Retained LM API", version="1.0", lifespan=lifespan)


class GenRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4096)
    max_new_tokens: int = Field(200, ge=1, le=512)
    temperature: float = Field(0.9, gt=0, le=5)
    top_k: int = Field(40, ge=0)


def latest_version(model_type: str, models_root="models") -> Optional[str]:
    manifest = os.path.join(models_root, model_type, "manifest.json")
    if not os.path.exists(manifest):
        return None
    with open(manifest, "r") as f:
        entries = json.load(f)
    if not entries:
        return None
    # sort by created_at
    entries.sort(key=lambda e: e.get("created_at", ""))
    return entries[-1]["version_path"]


def load_model_and_tokenizer(model_type: str, version_path: Optional[str] = None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if version_path is None:
        version_path = latest_version(model_type)
        if version_path is None:
            raise RuntimeError(f"No retained versions for model_type={model_type}")
    with open(os.path.join(version_path, "meta.json")) as f:
        meta = json.load(f)
    vocab = meta["vocab"]
    tok = CharTokenizer(stoi={ch: i for i, ch in enumerate(vocab)}, itos=vocab)
    V = len(vocab)
    # infer hidden/emb from args if present, else fall back to defaults
    with open(os.path.join(os.path.dirname(version_path), "manifest.json")) as f:
        manifest = json.load(f)
    rec = [e for e in manifest if e["version_path"] == version_path]
    if rec and rec[0].get("args"):
        a = rec[0]["args"]
        emb = a.get("emb", 256)
        hidden = a.get("hidden", 512)
        layers = a.get("layers", 4)
        dropout = a.get("dropout", 0.2)
        kernel = a.get("kernel", 5)
    else:
        emb, hidden, layers, dropout, kernel = 256, 512, 4, 0.2, 5

    if model_type == "rnn":
        model = RNNLanguageModel(
            V,
            emb_dim=emb,
            hidden_dim=hidden,
            num_layers=layers,
            dropout=dropout,
            tie_weights=(hidden == emb),
        )
    else:
        model = TCNLanguageModel(
            V, emb_dim=emb, channels=hidden, n_blocks=layers, kernel_size=kernel, dropout=dropout
        )
    ckpt = torch.load(
        os.path.join(version_path, "checkpoint.pt"), map_location=device, weights_only=True
    )
    model.load_state_dict(ckpt)
    model.to(device).eval()
    return model, tok, device


# global single load on startup (can be lazy if desired)
MODEL_TYPE = os.environ.get("MODEL_TYPE", "rnn")
MODEL_VERSION_PATH = os.environ.get("MODEL_VERSION_PATH")
_MODEL, _TOK, _DEVICE = None, None, None


def _load():
    global _MODEL, _TOK, _DEVICE
    _MODEL, _TOK, _DEVICE = load_model_and_tokenizer(MODEL_TYPE, MODEL_VERSION_PATH)


@torch.no_grad()
def _generate(prompt: str, max_new_tokens=200, temperature=0.9, top_k=40):
    from train_lm import sample

    return sample(_MODEL, _TOK, _DEVICE, prompt, max_new_tokens, temperature, top_k)


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_type": MODEL_TYPE}


@app.post("/generate")
def generate(req: GenRequest):
    try:
        out = _generate(req.prompt, req.max_new_tokens, req.temperature, req.top_k)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"prompt": req.prompt, "completion": out}
