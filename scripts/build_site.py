"""Build a portable static learning site with a source-derived file atlas."""

import ast
import html
import json
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import quote
import markdown

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "site"
REPO = "https://github.com/Cazzy-Aporbo/PearlMind-ML-Journey/blob/main/"
if OUT.exists():
    shutil.rmtree(OUT)
shutil.copytree(ROOT / "web", OUT)
(OUT / "data").mkdir(exist_ok=True)
(OUT / "files").mkdir(exist_ok=True)
evidence = ROOT / "outputs/evidence.json"
if not evidence.exists():
    raise SystemExit("Run scripts/run_evidence.py before building; measured evidence is required")
shutil.copy(evidence, OUT / "data/evidence.json")
DOCS = {
    "docs/LEARNING_GUIDE.md": "guide.html",
    "docs/READINESS.md": "readiness.html",
    "docs/SETUP.md": "setup.html",
    "docs/CONTRACTS.md": "contracts.html",
}


def shell(title, content, prefix=""):
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)} · PearlMind</title><link rel="stylesheet" href="{prefix}style.css"><link rel="icon" href="{prefix}assets/loopchii-cloud-icon-v7.svg"></head><body><header class="nav"><a class="wordmark" href="{prefix}index.html">PearlMind<span>the learning room</span></a><nav><a href="{prefix}index.html#atlas">Source atlas</a><a href="{prefix}guide.html">Walkthrough</a><a href="https://github.com/Cazzy-Aporbo/PearlMind-ML-Journey">GitHub ↗</a></nav></header><main class="document">{content}</main><footer><a href="{prefix}index.html">Back to the experiments →</a><span>Read the code. Question the result.</span></footer></body></html>'


for source, target in DOCS.items():
    rendered = markdown.markdown(
        (ROOT / source).read_text(), extensions=["tables", "fenced_code", "toc"]
    )

    def link(m):
        url = m.group(1)
        if url.startswith(("https:", "http:", "#", "mailto:")):
            return m.group(0)
        base, sep, fragment = url.partition("#")
        path = (ROOT / source).parent / base
        relative = path.resolve().relative_to(ROOT).as_posix()
        if relative in DOCS:
            destination = DOCS[relative]
        else:
            destination = REPO + quote(relative)
        return 'href="' + destination + (sep + fragment if sep else "") + '"'

    rendered = re.sub(r'href="([^"]+)"', link, rendered)
    (OUT / target).write_text(shell(source.split("/")[-1].replace("_", " "), rendered))

paths = (
    subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
    )
    .decode()
    .split("\0")
)
paths = sorted(set(p for p in paths if p and (ROOT / p).is_file()))
catalog = []
for path in paths:
    if path.endswith((".pyc", ".png", ".jpg", ".gif", ".svg")):
        continue
    if path.startswith(("web/", ".git/")):
        continue
    imports = []
    definitions = []
    data = (ROOT / path).read_text(errors="replace")
    if path.endswith(".py"):
        tree = ast.parse(data)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(n.name for n in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                definitions.append(
                    {
                        "name": node.name,
                        "line": node.lineno,
                        "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                        "signature": node.name
                        if isinstance(node, ast.ClassDef)
                        else node.name
                        + "("
                        + ast.unparse(node.args)
                        + ")"
                        + (" → " + ast.unparse(node.returns) if node.returns else ""),
                        "doc": ast.get_docstring(node) or "",
                    }
                )
    if path.startswith("src/pearlmind/lessons/"):
        area, status, description = (
            "Runnable experiments",
            "tested path",
            "Small CPU experiments with retained outputs. See the contracts and tests before adapting.",
        )
    elif path.startswith(
        (
            "src/pearlmind/evaluation",
            "src/pearlmind/utils",
            "src/pearlmind/cli",
            "src/pearlmind/deployment",
            "src/pearlmind/models/",
        )
    ):
        area, status, description = (
            "Models & runtime",
            "tested path",
            "Configuration, model persistence, metrics or local serving used by the core experiment.",
        )
    elif path.startswith("Learning/cnn_rnn_api_demo/"):
        area, status, description = (
            "Language & sequences",
            "core tests",
            "Character-level LSTM and causal-convolution lab. Model contracts are tested; full-corpus training is an optional local run.",
        )
    elif path.startswith("Learning/dolly"):
        area, status, description = (
            "Dataset exploration",
            "local data / reference",
            "Dataset inspection and historical plots. Check the selected script’s data path and dataset terms.",
        )
    elif path.startswith("Learning/0"):
        area, status, description = (
            "Original foundations",
            "extended lesson",
            "Preserved long-form lesson. Syntax checked; use the new small experiments for the CI-measured path.",
        )
    elif path.startswith(
        (
            "programs/",
            "scripts/living",
            "scripts/quantum",
            "scripts/neuro",
            "scripts/AI_Ethics",
            "scripts/MLForge",
            "scripts/avacado",
        )
    ) or path.startswith(("pearl mind", "pearlmind_ml_platform")):
        area, status, description = (
            "Research studies",
            "extended study",
            "Exploratory code requiring module-specific dependencies and validation. No clinical or production claim.",
        )
    elif path.startswith(
        ("tests/", ".github/", ".devcontainer/", "scripts/check", "scripts/build", "scripts/run")
    ) or path in ["Dockerfile", "pyproject.toml"]:
        area, status, description = (
            "Verification & environment",
            "build infrastructure",
            "Reproducibility, source checks, test contracts, packaging or site generation.",
        )
    else:
        area, status, description = (
            "Guides & references",
            "reference",
            "Supporting explanation or earlier work, retained for context. Follow the current README for runnable commands.",
        )
    if path.endswith(".py") and ast.get_docstring(tree):
        description = ast.get_docstring(tree).split("\n")[0][:200]
    if definitions:
        description += (
            " Defines "
            + ", ".join(d["name"] for d in definitions[:4])
            + ("." if len(definitions) <= 4 else "…")
        )
    slug = re.sub(r"[^a-zA-Z0-9_.-]", "_", path.replace("/", "--")) + ".html"
    page = "files/" + slug
    record = {
        "path": path,
        "page": page,
        "area": area,
        "status": status,
        "description": description,
        "imports": sorted(set(imports)),
        "definitions": definitions,
    }
    catalog.append(record)
    defs = "".join(
        f"<li><code>{html.escape(d['signature'])}</code> · line {d['line']}<p>{html.escape(d['doc'])}</p></li>"
        for d in definitions
    )
    content = f'<p class="eyebrow">{area} / {status}</p><h1>{html.escape(Path(path).name)}</h1><p>{html.escape(description)}</p><a class="source" href="{REPO + quote(path)}">Open source on GitHub ↗</a><div class="file-contract"><div><h3>Before you run</h3><p>Read the input, output and limitation notes in <a href="../contracts.html">the contracts</a>. An import inventory describes dependencies; it does not prove that every code path was executed.</p></div><div><h3>Connected imports</h3><p>{html.escape(", ".join(sorted(set(imports))) or "No Python imports in this file.")}</p></div></div>'
    if defs:
        content += "<h2>Definitions to follow</h2><ul>" + defs + "</ul>"
    content += (
        '<h2>The file, in full</h2><pre class="source-code"><code>'
        + html.escape(data)
        + "</code></pre>"
    )
    (OUT / page).write_text(shell(Path(path).name, content, "../"))
for old_guide in ["ds-ml-guide.html", "pearlmind_animations.html"]:
    shutil.copy(ROOT / old_guide, OUT / old_guide)
(OUT / "data/catalog.json").write_text(json.dumps(catalog, indent=2))
(OUT / ".nojekyll").touch()
(OUT / "robots.txt").write_text(
    "User-agent: *\nAllow: /\nSitemap: https://cazzy-aporbo.github.io/PearlMind-ML-Journey/sitemap.xml\n"
)
urls = ["", "guide.html", "readiness.html", "setup.html", "contracts.html"]
(OUT / "sitemap.xml").write_text(
    '<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
    + "".join(
        "<url><loc>https://cazzy-aporbo.github.io/PearlMind-ML-Journey/" + url + "</loc></url>"
        for url in urls
    )
    + "</urlset>"
)
print(f"Built {len(catalog)} file views and four connected guides.")
