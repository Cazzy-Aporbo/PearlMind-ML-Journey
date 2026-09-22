"""Run the small, documented experiments. Never synthesize a success score."""

from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import subprocess
from importlib.metadata import version
from pearlmind.lessons.tabular import run as tabular
from pearlmind.lessons.torch_lab import run as torch_lab
from pearlmind.lessons.systems import run as systems
from pearlmind.lessons.comparison import run as comparison

root = Path(__file__).resolve().parents[1]
report = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    "working_tree_dirty": bool(
        subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    ),
    "python": platform.python_version(),
    "platform": platform.platform(),
    "dependencies": {name: version(name) for name in ["numpy", "scikit-learn", "xgboost", "torch"]},
    "tabular": tabular(root / "outputs/tabular"),
    "torch": torch_lab(root / "outputs/torch"),
    "systems": systems(root / "outputs/systems"),
    "comparison": comparison(root / "outputs/comparison"),
}
(root / "outputs/evidence.json").write_text(json.dumps(report, indent=2, allow_nan=False))
print("Measured outputs written to outputs/evidence.json")
