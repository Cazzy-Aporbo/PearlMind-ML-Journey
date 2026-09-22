"""Fail on broken local links, missing assets or duplicate IDs in generated HTML."""

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit, unquote
import json

root = Path(__file__).resolve().parents[1] / "site"


class Page(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.ids = []
        self.links = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
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
for f in json.loads((root / "data/catalog.json").read_text()):
    if not (root / unquote(f["page"])).exists():
        errors.append(f"Missing catalog page {f['path']}")
if errors:
    raise SystemExit("\n".join(errors))
print(f"Checked {len(pages)} pages: local links, assets and anchors resolve.")
