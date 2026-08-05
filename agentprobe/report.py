"""Self-contained HTML diff report for failing snapshots.

Terminal diffs die with the CI log. This renders every failing snapshot with
its similarity score and unified diff into one shareable HTML file, so it can
live in CI artifacts or a PR comment. Everything is escaped: snapshot content
is untrusted model output and must never land in the DOM as markup.
"""

from __future__ import annotations

import html

_CSS = """
body { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; margin: 2rem; color: #1f2328; background: #fff; }
h1 { font-size: 1.2rem; }
.snap { border: 1px solid #d0d7de; border-radius: 8px; margin: 1rem 0; }
.snap h2 { font-size: 1rem; margin: 0; padding: .6rem .9rem; background: #f6f8fa; border-bottom: 1px solid #d0d7de; }
.meta { padding: .4rem .9rem; color: #57606a; font-size: .85rem; }
pre { margin: 0; padding: .8rem .9rem; overflow-x: auto; font-size: .82rem; line-height: 1.45; }
.ins { background: #dafbe1; color: #116329; display: block; }
.del { background: #ffebe9; color: #82071e; display: block; }
.ctx { color: #57606a; display: block; }
"""


def _row(line: str) -> str:
    esc = html.escape(line)
    if line.startswith(("--- ", "+++ ", "@@")):
        return f'<span class="ctx">{esc}</span>'
    if line.startswith("+"):
        return f'<span class="ins">{esc}</span>'
    if line.startswith("-"):
        return f'<span class="del">{esc}</span>'
    return f'<span class="ctx">{esc}</span>'


def render_diff_report(items: list[dict]) -> str:
    """Render failing snapshots as a self-contained HTML report."""
    sections = []
    for item in items:
        rows = "".join(_row(line) for line in item["diff"].splitlines())
        sim = item.get("similarity")
        meta = ""
        if sim is not None:
            meta = (
                f"similarity: {sim:.4f} (mode={item.get('mode')}, "
                f"threshold={item.get('threshold')})"
            )
        sections.append(
            f'<section class="snap"><h2>{html.escape(str(item["name"]))}</h2>'
            f'<div class="meta">{html.escape(meta)}</div>'
            f"<pre>{rows}</pre></section>"
        )
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        "<title>AgentProbe diff report</title>"
        f"<style>{_CSS}</style></head><body>"
        "<h1>AgentProbe diff report</h1>" + "".join(sections) + "</body></html>"
    )
