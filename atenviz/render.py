"""Write the viewer as one self-contained HTML page, with one or more graphs in it."""

import base64
import gzip
import html
import json
from pathlib import Path

VIEWER = Path(__file__).parent / "viewer"
SLOT = '<script id="graph-data" class="graph-data" type="application/gzip-base64"></script>'


def render_html(graphs: list[dict], slugs: list[str] | None = None) -> str:
    """Return the viewer page with its stylesheet, its scripts and ``graphs`` in it.

    The page works offline. With more than one graph, the page has a picker, and ``slugs`` name the
    graphs in the page address (``#pi0``). With no graph, the page asks for a graph file.
    """
    slugs = slugs or [f"graph{i + 1}" for i in range(len(graphs))]
    # The graph JSON is gzip-compressed and base64-encoded: a large model gives tens of MB of JSON, and
    # base64 has no "<" that can end the script element.
    data = "\n".join(
        f'<script class="graph-data" type="application/gzip-base64" data-slug="{html.escape(slug)}" '
        f'data-label="{html.escape(g["model"]["name"].split(" (")[0])}">'
        f'{base64.b64encode(gzip.compress(json.dumps(g, separators=(",", ":")).encode(), 6)).decode()}</script>'
        for g, slug in zip(graphs, slugs))
    page = (VIEWER / "index.html").read_text()
    for tag, inline in [
        ('<link rel="stylesheet" href="viewer.css">', f"<style>\n{(VIEWER / 'viewer.css').read_text()}</style>"),
        ('<script src="vendor/elk.bundled.js"></script>',
         f"<script>{(VIEWER / 'vendor/elk.bundled.js').read_text()}</script>"),
        (SLOT, data or SLOT),
        ('<script src="viewer.js"></script>', f"<script>\n{(VIEWER / 'viewer.js').read_text()}</script>"),
    ]:
        assert tag in page, tag
        page = page.replace(tag, inline)
    return page
