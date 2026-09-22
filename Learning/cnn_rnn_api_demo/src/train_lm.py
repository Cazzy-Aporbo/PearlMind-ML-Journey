# src/train_lm.py
import argparse, os, json, time, math, datetime, pathlib, random
from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tokenizer import CharTokenizer
from models import RNNLanguageModel, TCNLanguageModel


class CharDataset(Dataset):
    def __init__(self, data_ids, seq_len):
        if seq_len < 1 or len(data_ids) <= seq_len:
            raise ValueError("Text must contain at least seq_len + 1 characters")
        self.data = data_ids
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def load_text(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Training text not found at {path}. Please place a .txt file there."
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_tokens = 0.0, 0
    for X, Y in loader:
        X, Y = X.to(device), Y.to(device)
        logits = model(X)[0] if isinstance(model, RNNLanguageModel) else model(X)
        B, T, V = logits.size()
        loss = loss_fn(logits.reshape(B * T, V), Y.reshape(B * T))
        total_loss += loss.item() * (B * T)
        total_tokens += B * T
    model.train()
    return total_loss / max(1, total_tokens)


def top_k_logits(logits, k):
    k = min(k, logits.shape[-1])
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[..., [-1]]] = -float("Inf")
    return out


@torch.no_grad()
def sample(model, tokenizer, device, prompt: str, max_new_tokens=200, temperature=1.0, top_k=0):
    model.eval()
    if not prompt or temperature <= 0 or max_new_tokens < 0:
        raise ValueError("Use a nonempty prompt, positive temperature and nonnegative token limit")
    ids = tokenizer.encode(prompt)
    ids = torch.tensor([ids], dtype=torch.long, device=device)
    h = None
    for _ in range(max_new_tokens):
        if isinstance(model, RNNLanguageModel):
            logits, h = model(ids if h is None else ids[:, -1:].contiguous(), h)
        else:
            logits = model(ids)
            logits = logits[:, -1:, :]
        logits = logits[:, -1, :] / max(1e-6, temperature)
        if top_k > 0:
            logits = top_k_logits(logits, top_k)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
    return tokenizer.decode(ids[0].tolist())


def save_retained(run_dir, model, tokenizer_meta, score=None, args=None):
    os.makedirs(run_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(run_dir, "checkpoint.pt"))
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(tokenizer_meta, f)
    manifest_path = os.path.join(os.path.dirname(run_dir), "manifest.json")
    manifest = []
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            try:
                manifest = json.load(f)
            except Exception:
                manifest = []
    manifest.append(
        {
            "version_path": run_dir,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "score": score,
            "args": vars(args) if args else None,
        }
    )
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def train(args):
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError("epochs, batch size and learning rate must be positive")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = load_text(args.data_path)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(1)
    split = int(len(text) * 0.9)
    train_text, validation_text = text[:split], text[split:]
    tokenizer = CharTokenizer.build(train_text)
    # Split raw text BEFORE creating windows; no window crosses the boundary.
    train_ds = CharDataset(tokenizer.encode(train_text), args.seq_len)
    val_ds = CharDataset(tokenizer.encode(validation_text), args.seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    vocab_size = len(tokenizer.itos)
    if args.model == "rnn":
        model = RNNLanguageModel(
            vocab_size,
            emb_dim=args.emb,
            hidden_dim=args.hidden,
            num_layers=args.layers,
            dropout=args.dropout,
            tie_weights=(args.hidden == args.emb),
        )
    else:
        model = TCNLanguageModel(
            vocab_size,
            emb_dim=args.emb,
            channels=args.hidden,
            n_blocks=args.layers,
            kernel_size=args.kernel,
            dropout=args.dropout,
        )
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            logits = model(X)[0] if isinstance(model, RNNLanguageModel) else model(X)
            B, T, V = logits.size()
            loss = loss_fn(logits.reshape(B * T, V), Y.reshape(B * T))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
        val_loss = evaluate(model, val_loader, device)
        print(f"epoch {epoch} val_loss {val_loss:.3f}")
        if val_loss < best_val:
            best_val = val_loss
            # Save retained model
            ts = datetime.datetime.now().strftime("v%Y-%m-%d_%H-%M-%S-%f")
            run_dir = os.path.join(args.models_root, args.model, ts)
            meta = {"vocab": tokenizer.itos}
            save_retained(run_dir, model, tokenizer_meta=meta, score=val_loss, args=args)

    # Optional demo generation
    if args.sample_after:
        model.load_state_dict(
            torch.load(
                os.path.join(run_dir, "checkpoint.pt"), map_location=device, weights_only=True
            )
        )
        prompt = train_text[: min(20, len(train_text))]
        print(
            sample(model, tokenizer, device, prompt, max_new_tokens=80, temperature=0.9, top_k=40)
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="data/shakespeare.txt")
    p.add_argument("--models-root", default="models")
    p.add_argument("--model", choices=["rnn", "tcn"], default="rnn")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--emb", type=int, default=256)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--kernel", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--sample-after", action="store_true")
    args = p.parse_args()
    train(args)
