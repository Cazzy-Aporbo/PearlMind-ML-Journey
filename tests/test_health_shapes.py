"""Synthetic execution contracts, deliberately not clinical validation."""

import importlib.util
from pathlib import Path
import sys
import torch
import pytest

path = Path(__file__).parents[1] / "programs/delphi-2m/cazzy_aporbo_model.py"
spec = importlib.util.spec_from_file_location("health_study", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def test_health_shapes_finite_causal_and_no_forward_mutation():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    model = module.create_model(
        {
            "vocab_size": 16,
            "hidden_dim": 16,
            "num_heads": 4,
            "num_layers": 2,
            "ff_dim": 32,
            "biomarker_dim": 3,
            "genetic_dim": 4,
            "dropout": 0.0,
        }
    ).eval()
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    ages = torch.tensor([[1.0, 1.0, 20.0, 30.0], [1.0, 10.0, 20.0, 30.0]])
    bio = torch.randn(2, 4, 3)
    gen = torch.randn(2, 4, 4)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    out = model(tokens, ages, bio, gen, return_uncertainty=True)
    assert out["disease_logits"].shape == (2, 4, 16)
    assert all(torch.isfinite(t).all() for t in out.values())
    assert (out["survival_curves"][..., 1:] <= out["survival_curves"][..., :-1]).all()
    changed = tokens.clone()
    changed[:, 2:] = 7
    bio2 = bio.clone()
    bio2[:, 2:] += 100
    gen2 = gen.clone()
    gen2[:, 2:] -= 100
    out2 = model(changed, ages, bio2, gen2)
    torch.testing.assert_close(out["disease_logits"][:, :2], out2["disease_logits"][:, :2])
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, before[k])
    model.train()
    train = model(tokens, ages, bio, gen)
    train["disease_logits"].square().mean().backward()
    assert model.token_embedding.weight.grad is not None
    assert torch.isfinite(model.token_embedding.weight.grad).all()


def test_health_config_rejects_incompatible_or_unknown_settings():
    with pytest.raises(ValueError):
        module.create_model({"hidden_dim": 15})
    with pytest.raises(ValueError):
        module.create_model({"imaginary": True})
