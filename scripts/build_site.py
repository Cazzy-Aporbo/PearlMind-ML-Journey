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
    "docs/CONCEPTS.md": "concepts.html",
    "docs/COMPARISON.md": "comparison.html",
    "ChangeLog.md": "changes.html",
}


BASE = "https://cazzy-aporbo.github.io/PearlMind-ML-Journey/"
PAGE_INFO = {
    "guide.html": (
        "Machine learning walkthrough",
        "Trace regression, PyTorch tensors, language models, error costs and weighted gradients through runnable Python lessons.",
    ),
    "readiness.html": (
        "Model readiness and limitations",
        "Understand which PearlMind experiments are tested, which remain research studies, and what deployment would still require.",
    ),
    "setup.html": (
        "Install and run PearlMind",
        "Set up Python, Codespaces or Docker; run CPU machine-learning lessons and resolve common environment errors.",
    ),
    "contracts.html": (
        "Machine learning input and output contracts",
        "Find data shapes, file outputs, prerequisites and failure cases for PearlMind training, evaluation and local prediction.",
    ),
    "concepts.html": (
        "Machine learning concept dictionary",
        "Connect mathematical definitions, related terms, common confusions and runnable examples across data science and systems.",
    ),
    "comparison.html": (
        "Compare models with paired uncertainty",
        "Run a prior, logistic regression and shallow tree on matched test rows; inspect Brier scores and paired bootstrap intervals.",
    ),
    "changes.html": (
        "PearlMind change record",
        "Trace the learning-room updates to code, tests, dated commits and reproducible experiment artifacts.",
    ),
    "source-index.html": (
        "PearlMind source index",
        "Browse the complete source atlas by learning area, from original lessons to tested models and extended research studies.",
    ),
}


def shell(title, content, prefix="", page="", description="", historical=False):
    title, description = PAGE_INFO.get(
        page,
        (
            title,
            description
            or f"Read {title}: source, definitions and connected learning notes in PearlMind.",
        ),
    )
    canonical = BASE + quote(page)
    schema = {
        "@context": "https://schema.org",
        "@type": "LearningResource",
        "name": title,
        "description": description,
        "url": canonical,
        "inLanguage": "en",
        "author": {
            "@type": "Person",
            "name": "Cazandra Aporbo",
            "url": "https://github.com/Cazzy-Aporbo",
        },
        "isPartOf": {"@type": "WebSite", "name": "PearlMind ML Journey", "url": BASE},
    }
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} · PearlMind</title><meta name="description" content="{html.escape(description, quote=True)}"><meta name="author" content="Cazandra Aporbo">
<link rel="canonical" href="{canonical}"><meta name="robots" content="{"noindex,follow" if historical else "index,follow"}">
<meta property="og:title" content="{html.escape(title, quote=True)} · PearlMind"><meta property="og:description" content="{html.escape(description, quote=True)}"><meta property="og:type" content="website"><meta property="og:url" content="{canonical}"><meta property="og:image" content="{BASE}assets/social-preview.png"><meta property="og:image:alt" content="PearlMind — make a prediction, meet its consequences"><meta name="twitter:card" content="summary_large_image">
<script type="application/ld+json">{json.dumps(schema).replace("<", chr(92) + "u003c")}</script>
<link rel="stylesheet" href="{prefix}style.css"><link rel="icon" href="{prefix}assets/loopchii-cloud-icon-v7.svg"><script src="{prefix}guide.js" defer></script></head>
<body><a class="skip" href="#document">Skip to content</a><header class="nav"><a class="wordmark" href="{prefix}index.html">PearlMind<span>the learning room</span></a><nav aria-label="Learning navigation"><a href="{prefix}source-index.html">Source atlas</a><a href="{prefix}concepts.html">Dictionary</a><a href="{prefix}guide.html">Walkthrough</a><a href="https://github.com/Cazzy-Aporbo/PearlMind-ML-Journey">GitHub ↗</a></nav></header>
<main class="document" id="document">{content}</main><footer><a href="{prefix}index.html">Back to the experiments →</a><a href="{prefix}changes.html">Change record</a><span>Read the code. Question the result.</span></footer></body></html>'''


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
        if not path.exists():
            raise ValueError(f"Broken guide link in {source}: {url}")
        if relative in DOCS:
            destination = DOCS[relative]
        else:
            destination = REPO + quote(relative)
        return 'href="' + destination + (sep + fragment if sep else "") + '"'

    rendered = re.sub(r'href="([^"]+)"', link, rendered)
    if target == "comparison.html":
        measured = json.loads(evidence.read_text())
        experiment = measured["comparison"]
        rows = "".join(
            f'<tr><th scope="row">{name}</th><td>{v["accuracy"]:.1%}</td><td>{v["brier"]:.4f}</td><td>{v["log_loss"]:.4f}</td></tr>'
            for name, v in experiment["metrics"].items()
        )
        interval = experiment["tree_minus_linear"]
        crossing = interval["lower"] <= 0 <= interval["upper"]
        interpretation = (
            "The interval crosses zero: the direction of the probability-loss difference is unresolved under this design."
            if crossing
            else "The interval excludes zero under this fixed-model, independent-row design; it does not include retraining uncertainty."
        )
        observation = ""
        if (
            experiment["metrics"]["tree"]["accuracy"] > experiment["metrics"]["linear"]["accuracy"]
            and experiment["metrics"]["tree"]["log_loss"]
            > experiment["metrics"]["linear"]["log_loss"]
        ):
            observation = "In this run the tree makes more correct class decisions, yet pays a larger log-loss penalty. Inspect its confident errors before declaring a winner."
        panel = f'<aside class="measured-comparison"><p class="eyebrow">Measured during this build</p><h2>Same rows. Different answers.</h2><table><thead><tr><th>Model</th><th>Accuracy ↑</th><th>Brier ↓</th><th>Log loss ↓</th></tr></thead><tbody>{rows}</tbody></table><p>{observation}</p><p>Tree minus linear Brier: <strong>{interval["mean_difference"]:.4f}</strong>; paired 95% percentile interval <strong>[{interval["lower"]:.4f}, {interval["upper"]:.4f}]</strong>. {interpretation}</p><p class="fine">150 synthetic held-out rows · seed {experiment["seed"]} · commit {measured["commit"][:8]} · {measured["created_at"]}. <a href="data/evidence.json">Read the full evidence</a>.</p></aside>'
        rendered = rendered.replace("</h1>", "</h1>" + panel, 1)
    if target == "concepts.html":
        rendered = rendered.replace(
            '<div class="toc">',
            '<div class="dictionary-tools"><label for="term-search">Find a term or related idea</label><input id="term-search" type="search" placeholder="Try: probability, leakage, gradient…"><output id="term-count" aria-live="polite"></output></div><div class="toc">',
        )
    (OUT / target).write_text(shell(source.split("/")[-1].replace("_", " "), rendered, page=target))

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
    (OUT / page).write_text(
        shell(
            path,
            content,
            "../",
            page=page,
            description=f"{path}: {description}",
            historical=path.startswith("docs/reference/") or path == "docs/original-model-atlas.md",
        )
    )
# A server-rendered index makes every source continuation discoverable without JavaScript.
index_content = "<h1>The source atlas</h1><p>Browse by purpose, then follow the file to its imports, definitions and full source.</p>"
for area in sorted({f["area"] for f in catalog}):
    index_content += "<h2>" + html.escape(area) + '</h2><ul class="source-directory">'
    for f in catalog:
        if f["area"] == area:
            index_content += f'<li><a href="{f["page"]}">{html.escape(f["path"])}</a><span>{html.escape(f["status"])}</span></li>'
    index_content += "</ul>"
(OUT / "source-index.html").write_text(
    shell("Source index", index_content, page="source-index.html")
)
for old_guide in ["ds-ml-guide.html", "pearlmind_animations.html"]:
    legacy = (ROOT / old_guide).read_text()
    legacy_description = (
        "Explore PearlMind’s earlier illustrated data-science guide, with links to the current executable lessons and source atlas."
        if old_guide == "ds-ml-guide.html"
        else "Explore PearlMind’s original animated mathematical concepts, alongside the current tested machine-learning walkthrough."
    )
    legacy = re.sub(r'<meta\b[^>]*name=[\'"]description[\'"][^>]*>', "", legacy, flags=re.I)
    metadata = f'<meta name="description" content="{legacy_description}"><link rel="canonical" href="{BASE}{old_guide}"><meta property="og:image" content="{BASE}assets/social-preview.png"><meta name="twitter:card" content="summary_large_image"><link rel="icon" href="assets/loopchii-cloud-icon-v7.svg">'
    legacy = legacy.replace("</head>", metadata + "</head>")
    legacy = re.sub(
        r"(<body[^>]*>)",
        r'\1<nav style="padding:1rem;background:#090e1c;color:#b8dded;text-align:center"><a style="color:inherit" href="index.html">PearlMind learning room</a> · <a style="color:inherit" href="guide.html">Current walkthrough</a> · <a style="color:inherit" href="concepts.html">Concept dictionary</a></nav>',
        legacy,
        count=1,
    )
    (OUT / old_guide).write_text(legacy)
(OUT / "data/catalog.json").write_text(json.dumps(catalog, indent=2))
(OUT / ".nojekyll").touch()
(OUT / "robots.txt").write_text(
    "User-agent: *\nAllow: /\nSitemap: https://cazzy-aporbo.github.io/PearlMind-ML-Journey/sitemap.xml\n"
)
urls = ["", *DOCS.values(), "source-index.html"] + [
    f["page"]
    for f in catalog
    if f["path"].endswith(".py") and f["status"] in {"tested path", "core tests", "extended lesson"}
]
(OUT / "sitemap.xml").write_text(
    '<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
    + "".join(
        "<url><loc>https://cazzy-aporbo.github.io/PearlMind-ML-Journey/" + url + "</loc></url>"
        for url in urls
    )
    + "</urlset>"
)
print(
    f"Built {len(catalog)} file views, {len(DOCS)} connected guides and a crawlable source index."
)
