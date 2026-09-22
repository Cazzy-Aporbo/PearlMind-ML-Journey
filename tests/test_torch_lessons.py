import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
from pearlmind.lessons.torch_lab import run

LM = Path(__file__).parents[1] / "Learning/cnn_rnn_api_demo/src"
sys.path.insert(0, str(LM))
from models import RNNLanguageModel, TCNLanguageModel
from train_lm import CharDataset, sample, top_k_logits
from tokenizer import CharTokenizer

torch.set_num_threads(1)


def test_torch_experiment_learns_and_exports(tmp_path):
    result = run(output=tmp_path, epochs=30)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["loss_history"][-1] < metrics["loss_history"][0]
    assert 0 <= metrics["test_accuracy"] <= 1
    state = torch.load(tmp_path / "weights.pt", weights_only=True)
    assert state
    assert (tmp_path / "preprocessing.json").exists()
    with pytest.raises(ValueError):
        run(output=tmp_path, epochs=0)


@pytest.mark.parametrize("kind", ["rnn", "tcn"])
def test_language_models_shapes_gradients_and_causality(kind):
    torch.manual_seed(42)
    model = RNNLanguageModel(8, 8, 8, 1, 0) if kind == "rnn" else TCNLanguageModel(8, 8, 8, 2, 3, 0)
    ids = torch.tensor([[0, 1, 2, 3, 4]])
    out = model(ids)
    logits = out[0] if kind == "rnn" else out
    assert logits.shape == (1, 5, 8)
    torch.nn.functional.cross_entropy(logits.reshape(-1, 8), ids.reshape(-1)).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    changed = ids.clone()
    changed[:, 3:] = 7
    new = model(changed)
    new = new[0] if kind == "rnn" else new
    torch.testing.assert_close(logits[:, :3], new[:, :3])


def test_windows_shift_by_one_and_respect_source_bounds():
    data = CharDataset(list(range(10)), 3)
    assert len(data) == 7
    x, y = data[6]
    assert x.tolist() == [6, 7, 8] and y.tolist() == [7, 8, 9]
    with pytest.raises(ValueError):
        CharDataset([1, 2], 2)


def test_prompt_consumed_in_full_and_tied_embedding_not_zero():
    tokenizer = CharTokenizer.build("abc")
    model = RNNLanguageModel(3, 8, 8, 1, 0)
    assert torch.count_nonzero(model.encoder.weight) > 0
    lengths = []
    handle = model.register_forward_pre_hook(lambda module, args: lengths.append(args[0].shape[1]))
    value = sample(model, tokenizer, "cpu", "abc", max_new_tokens=2, top_k=100)
    handle.remove()
    assert lengths == [3, 1] and len(value) == 5 and value.startswith("abc")
    with pytest.raises(ValueError):
        tokenizer.encode("z")
    with pytest.raises(ValueError):
        sample(model, tokenizer, "cpu", "", max_new_tokens=2)
    assert top_k_logits(torch.ones(1, 3), 100).shape == (1, 3)


@pytest.mark.parametrize("kind", ["rnn", "tcn"])
def test_small_language_training_checkpoint_serves(tmp_path, kind):
    from argparse import Namespace
    from train_lm import train
    from serve import load_model_and_tokenizer

    corpus = tmp_path / "text.txt"
    corpus.write_text("abc cab bac " * 30)
    args = Namespace(
        data_path=str(corpus),
        seed=42,
        seq_len=8,
        batch_size=16,
        model=kind,
        emb=8,
        hidden=8,
        layers=1,
        dropout=0.0,
        kernel=3,
        lr=0.01,
        epochs=1,
        grad_clip=1.0,
        models_root=str(tmp_path / "models"),
        sample_after=True,
    )
    train(args)
    manifest = json.loads((tmp_path / "models" / kind / "manifest.json").read_text())
    assert np.isfinite(manifest[-1]["score"])
    model, tok, device = load_model_and_tokenizer(kind, manifest[-1]["version_path"], device="cpu")
    assert sample(model, tok, device, "abc", max_new_tokens=3).startswith("abc")
