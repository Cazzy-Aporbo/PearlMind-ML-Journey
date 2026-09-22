"""Compile every tracked/new Python source without importing optional frameworks."""

import ast
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
paths = (
    subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root
    )
    .decode()
    .split("\0")
)
count = 0
for relative in sorted(set(paths)):
    if not relative.endswith(".py"):
        continue
    path = root / relative
    ast.parse(path.read_text(), filename=relative)
    count += 1
print(f"Parsed {count} Python files. Syntax validity is not runtime or scientific validation.")
