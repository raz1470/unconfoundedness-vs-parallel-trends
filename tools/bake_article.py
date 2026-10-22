"""Bake the article's JS-rendered numbers and charts into docs/article.html.

The page draws its charts and fills its numbers with JavaScript from the DATA
block. Readers that do not run scripts (link previews, RSS readers, crawlers,
some in-app browsers, publishing platforms) would see blanks. This script
renders the page in headless Chromium and writes the result back into the
HTML as static markup. The scripts stay in place and redraw on load, so the
page behaves as before in a browser.

Run after tools/article_data.py:

    uv run --with playwright python -m playwright install chromium
    uv run --with playwright python tools/bake_article.py

Safe to re-run: baked charts sit between <!--BAKE--> markers and are replaced.
"""

import re
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

ARTICLE = Path("docs/article.html")
PANELS = re.compile(
    r'(<div class="panels" id="(\w+)">)(?:<!--BAKE-->.*?<!--/BAKE-->)?(</div>)', re.S
)
SPANS = re.compile(r'(<span data-v="[^"]*"(?: data-f="[^"]*")?>)[^<]*(</span>)')


def render(html: str) -> tuple[dict[str, str], list[str]]:
    """Return the rendered inner HTML of each chart container and each number."""
    with tempfile.TemporaryDirectory() as tmp:
        page_path = Path(tmp) / "article.html"
        page_path.write_text(html, encoding="utf-8")
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1000, "height": 900})
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(page_path.as_uri())
            page.wait_for_timeout(500)
            if errors:
                raise SystemExit(f"page errors: {errors}")
            panels = page.evaluate(
                "Object.fromEntries([...document.querySelectorAll('div.panels[id]')]"
                ".map(d => [d.id, d.innerHTML]))"
            )
            values = page.evaluate(
                "[...document.querySelectorAll('span[data-v]')].map(s => s.textContent)"
            )
            browser.close()
    return panels, values


def main() -> None:
    html = ARTICLE.read_text(encoding="utf-8")
    panels, values = render(html)

    def bake_panel(m: re.Match) -> str:
        inner = panels.get(m.group(2), "")
        if not inner:
            raise SystemExit(f"chart {m.group(2)} rendered empty")
        return f"{m.group(1)}<!--BAKE-->{inner}<!--/BAKE-->{m.group(3)}"

    html, n_panels = PANELS.subn(bake_panel, html)
    spans = iter(values)
    html, n_spans = SPANS.subn(lambda m: f"{m.group(1)}{next(spans)}{m.group(2)}", html)
    if n_panels != len(panels) or n_spans != len(values):
        raise SystemExit(
            f"mismatch: {n_panels}/{len(panels)} charts, {n_spans}/{len(values)} numbers"
        )
    ARTICLE.write_text(html, encoding="utf-8")
    print(f"baked {n_panels} charts and {n_spans} numbers into {ARTICLE}")


if __name__ == "__main__":
    main()
