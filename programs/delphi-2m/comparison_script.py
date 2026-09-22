"""Inspect this prototype without inferring superiority from architectural features.

No external model benchmark or clinical performance is measured here.
"""

import json
import torch
from cazzy_aporbo_model import create_model


def inspect_prototype():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    config = {
        "vocab_size": 32,
        "hidden_dim": 16,
        "num_layers": 2,
        "num_heads": 4,
        "ff_dim": 32,
        "biomarker_dim": 3,
        "genetic_dim": 4,
        "dropout": 0.0,
    }
    model = create_model(config).eval()
    tokens = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    ages = torch.tensor([[1.0, 10.0, 20.0, 30.0], [1.0, 8.0, 16.0, 24.0]])
    with torch.no_grad():
        result = model(tokens, ages, return_uncertainty=True)
    return {
        "task": "synthetic shape inspection",
        "seed": 42,
        "configuration": config,
        "parameters": sum(p.numel() for p in model.parameters()),
        "inputs": {"tokens": list(tokens.shape), "ages_days": list(ages.shape)},
        "outputs": {
            name: {"shape": list(value.shape), "finite": bool(torch.isfinite(value).all())}
            for name, value in result.items()
        },
        "clinical_performance": None,
        "external_comparison": "Not measured",
        "next_questions": [
            "Does the evaluation respect patient and time boundaries?",
            "Are uncertainty estimates calibrated on independent data?",
            "Does each additional component outperform a simpler baseline under matched conditions?",
            "Are data rights, labels, missingness and subgroup uncertainty understood?",
        ],
        "limitations": "No patient data, clinical efficacy, novel-method claim or deployment approval is established by this run.",
    }


if __name__ == "__main__":
    print(json.dumps(inspect_prototype(), indent=2))
