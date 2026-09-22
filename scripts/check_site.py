"""Fail on broken local links, missing assets or duplicate IDs in generated HTML."""

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit, unquote
import json
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parents[1] / "site"


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.ids = []
        self.links = []
        self.meta = {}
        self.canonical = None
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "meta":
            self.meta[a.get("name", a.get("property", ""))] = a.get("content", "")
        if tag == "link" and a.get("rel") == "canonical":
            self.canonical = a.get("href")
        if "id" in a:
            self.ids.append(a["id"])
        for key in ["href", "src"]:
            if key in a:
                self.links.append(a[key])


pages = {p: Page(p.read_text()) for p in root.rglob("*.html")}
errors = []
for p, page in pages.items():
    if len(page.ids) != len(set(page.ids)):
        errors.append(f"{p}: duplicate ID")
    for link in page.links:
        u = urlsplit(link)
        if u.scheme or u.netloc:
            continue
        target = (p.parent / unquote(u.path)).resolve() if u.path else p
        if target.is_dir():
            target = target / "index.html"
        if not target.exists():
            errors.append(f"{p.name}: missing {link}")
        elif u.fragment and target in pages and u.fragment not in pages[target].ids:
            errors.append(f"{p.name}: missing anchor {link}")
base = "https://cazzy-aporbo.github.io/PearlMind-ML-Journey/"
for path, page in pages.items():
    relative = path.relative_to(root).as_posix()
    expected = base if relative == "index.html" else base + relative
    if page.canonical != expected:
        errors.append(f"{relative}: incorrect canonical")
    if not page.meta.get("description") or not page.meta.get("og:image"):
        errors.append(f"{relative}: missing search/share metadata")
for node in ET.parse(root / "sitemap.xml").iter("{http://www.sitemaps.org/schemas/sitemap/0.9}loc"):
    if (
        not node.text.startswith(base)
        or not (root / unquote(node.text[len(base) :] or "index.html")).is_file()
    ):
        errors.append(f"Invalid sitemap URL: {node.text}")
for f in json.loads((root / "data/catalog.json").read_text()):
    if not (root / unquote(f["page"])).exists():
        errors.append(f"Missing catalog page {f['path']}")
if errors:
    raise SystemExit("\n".join(errors))
print(
    f"Checked {len(pages)} pages: local links, assets, anchors, search metadata and sitemap resolve."
)
