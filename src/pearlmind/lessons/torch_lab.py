"""Train a tiny PyTorch classifier and retain the evidence, on CPU without downloads."""

from pathlib import Path
import argparse
import json
import time
import torch
from torch import nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def run(output="outputs/torch", epochs=80, seed=42):
    if epochs < 1 or epochs > 2000:
        raise ValueError("Choose 1 to 2000 epochs for this CPU lesson")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    X, y = make_moons(n_samples=500, noise=0.18, random_state=seed)
    train, test = train_test_split(range(len(y)), test_size=0.2, stratify=y, random_state=seed)
    scaler = StandardScaler().fit(X[train])
    features = torch.tensor(scaler.transform(X), dtype=torch.float32)
    targets = torch.tensor(y, dtype=torch.long)
    model = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 2))
    opt = torch.optim.Adam(model.parameters(), lr=0.03)
    criterion = nn.CrossEntropyLoss()
    losses = []
    start = time.perf_counter()
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(features[train])
        loss = criterion(logits, targets[train])
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        logits = model(features[test])
        accuracy = float((logits.argmax(1) == targets[test]).float().mean())
    report = {
        "seed": seed,
        "train_shape": [len(train), 2],
        "test_shape": [len(test), 2],
        "logits_shape": list(logits.shape),
        "parameters": sum(p.numel() for p in model.parameters()),
        "epochs": epochs,
        "loss_history": losses,
        "test_accuracy": accuracy,
        "seconds": time.perf_counter() - start,
        "device": "cpu",
        "data": "synthetic moons",
        "limitations": "One small split; not evidence for real-world deployment.",
    }
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "weights.pt")
    (path / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    (path / "preprocessing.json").write_text(
        json.dumps({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()})
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses, color="#7961ad")
    ax.set(
        xlabel="Training step", ylabel="Cross entropy", title="Learning is a change we can measure"
    )
    fig.tight_layout()
    fig.savefig(path / "loss.png", dpi=150)
    plt.close(fig)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/torch")
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.epochs), indent=2))
