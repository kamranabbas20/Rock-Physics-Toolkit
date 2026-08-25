"""Turn one well's analysis into a single self-contained HTML document.

The toolkit already exports seven CSVs, and a CSV is a good way to move
numbers and a poor way to move a *result*: it carries none of the settings
that produced it, none of the figures, and none of the caveats.  Six months
later nobody can tell what wavelet was used or which events were filtered out.

This builds one file that answers those questions alongside the numbers.  Two
constraints shape it:

**Self-contained.** Plotly is inlined once rather than pulled from a CDN, so
the file opens with no network and keeps working when a CDN version moves on.
It costs a few megabytes, which is the right trade for something meant to be
emailed and archived.  ``no_external_references`` exists so a test can hold
that property rather than trusting it.

**Provenance before results.** The settings that produced the numbers come
first, because a reflector table without the wavelet and the amplitude cut
that made it is not reproducible.

Streamlit-free: it takes plain figures and frames, so it can be tested — and
in principle scripted — without a browser in the loop.
"""

from __future__ import annotations

import datetime as _datetime
import html as _html
import re
from dataclasses import dataclass, field

__all__ = [
    "Section",
    "figure_html",
    "frame_html",
    "key_values_html",
    "paragraph_html",
    "report_html",
    "no_external_references",
]

_CSS = """
:root { --ink:#1a1a1a; --muted:#666; --rule:#e3e3e3; --accent:#1565c0;
        --warn:#c62828; --paper:#fff; }
* { box-sizing:border-box; }
body { margin:0; padding:0 0 5rem; background:var(--paper); color:var(--ink);
       font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,
       "Helvetica Neue",Arial,sans-serif; }
.wrap { max-width:1100px; margin:0 auto; padding:2.5rem 1.5rem; }
header { border-bottom:3px solid var(--ink); padding-bottom:1rem;
         margin-bottom:.5rem; }
h1 { font-size:1.9rem; margin:0 0 .3rem; letter-spacing:-.01em; }
.sub { color:var(--muted); font-size:.95rem; }
h2 { font-size:1.25rem; margin:2.6rem 0 .2rem; padding-top:.9rem;
     border-top:1px solid var(--rule); }
h2:first-of-type { border-top:0; }
.lead { color:var(--muted); margin:.15rem 0 1rem; max-width:70ch; }
table { border-collapse:collapse; width:100%; font-size:.88rem;
        margin:.4rem 0 1rem; }
th,td { text-align:left; padding:.4rem .6rem; border-bottom:1px solid var(--rule);
        vertical-align:top; }
th { background:#fafafa; font-weight:600; white-space:nowrap; }
td.num,th.num { text-align:right; font-variant-numeric:tabular-nums; }
.kv { display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr));
      gap:.15rem 1.5rem; margin:.4rem 0 1rem; }
.kv div { display:flex; justify-content:space-between; gap:1rem;
          border-bottom:1px dotted var(--rule); padding:.25rem 0; }
.kv .k { color:var(--muted); }
.kv .v { font-variant-numeric:tabular-nums; text-align:right; }
.scroll { overflow-x:auto; }
.note { border-left:3px solid var(--accent); background:#f6f9fd;
        padding:.7rem .9rem; margin:.8rem 0; font-size:.92rem; }
.note.warn { border-left-color:var(--warn); background:#fdf6f6; }
footer { margin-top:3rem; padding-top:1rem; border-top:1px solid var(--rule);
         color:var(--muted); font-size:.85rem; }
@media print {
  h2 { break-before:auto; break-after:avoid; }
  .wrap { max-width:none; padding:0; }
}
"""


@dataclass
class Section:
    """One heading, an optional sentence under it, and a list of HTML blocks."""

    title: str
    blocks: list = field(default_factory=list)
    lead: str = ""

    def render(self):
        parts = [f"<h2>{_html.escape(self.title)}</h2>"]
        if self.lead:
            parts.append(f'<p class="lead">{self.lead}</p>')
        parts.extend(b for b in self.blocks if b)
        return "\n".join(parts)


def paragraph_html(text, kind=None):
    """A paragraph, or a called-out note when ``kind`` is given."""
    if kind in ("note", "warn"):
        css = "note warn" if kind == "warn" else "note"
        return f'<div class="{css}">{text}</div>'
    return f"<p>{text}</p>"


def key_values_html(pairs):
    """A two-column grid of label/value rows — the settings blocks."""
    rows = "".join(
        f'<div><span class="k">{_html.escape(str(k))}</span>'
        f'<span class="v">{_html.escape(str(v))}</span></div>'
        for k, v in pairs
    )
    return f'<div class="kv">{rows}</div>'


def frame_html(frame, max_rows=None, float_format="{:,.4g}"):
    """A DataFrame as a plain table, numbers right-aligned.

    ``pandas.DataFrame.to_html`` would be shorter, but it emits its own
    classes and inline styling that fight the stylesheet, and it has no way to
    right-align only the numeric columns.
    """
    import numpy as np
    import pandas as pd

    if frame is None or not len(frame):
        return "<p><em>Nothing to show.</em></p>"

    shown = frame if max_rows is None else frame.head(int(max_rows))
    numeric = {c for c in shown.columns
               if pd.api.types.is_numeric_dtype(shown[c])
               and not pd.api.types.is_bool_dtype(shown[c])}

    head = "".join(
        f'<th class="num">{_html.escape(str(c))}</th>' if c in numeric
        else f"<th>{_html.escape(str(c))}</th>" for c in shown.columns)

    body = []
    for _, row in shown.iterrows():
        cells = []
        for column in shown.columns:
            value = row[column]
            if column in numeric:
                text = ("" if value is None or (isinstance(value, float)
                                                and not np.isfinite(value))
                        else float_format.format(value))
                cells.append(f'<td class="num">{_html.escape(text)}</td>')
            else:
                cells.append(f"<td>{_html.escape(str(value))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")

    more = ""
    if max_rows is not None and len(frame) > int(max_rows):
        more = (f"<p class=\"lead\">Showing {int(max_rows)} of {len(frame)} "
                "rows; the CSV export carries them all.</p>")
    return (f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
            f"<tbody>{''.join(body)}</tbody></table></div>{more}")


def figure_html(fig, first=False, height=None):
    """One Plotly figure, with the library inlined on the first call only.

    Inlining rather than linking a CDN is what makes the report open with no
    network at all; repeating it per figure would multiply several megabytes
    by the figure count, so only the first carries it.
    """
    if fig is None:
        return ""
    if height is not None:
        fig.update_layout(height=int(height))
    return fig.to_html(full_html=False,
                       include_plotlyjs="inline" if first else False,
                       config={"displaylogo": False})


def report_html(title, sections, subtitle=None, generated=None, footer=None):
    """Assemble the whole document.

    ``generated`` is injectable so a test can pin the output byte for byte
    rather than chasing the clock.
    """
    stamp = generated or _datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>{_html.escape(str(title))}</title>",
        f"<style>{_CSS}</style></head><body><div class=\"wrap\">",
        "<header>",
        f"<h1>{_html.escape(str(title))}</h1>",
    ]
    if subtitle:
        lines.append(f'<div class="sub">{subtitle}</div>')
    lines.append(f'<div class="sub">Generated {_html.escape(stamp)}</div>')
    lines.append("</header>")
    lines.extend(s.render() for s in sections)
    lines.append(
        "<footer>" + (footer or
                      "Generated by the Rock Physics Toolkit. Every figure is "
                      "interactive and embedded; this file needs no network.")
        + "</footer>")
    lines.append("</div></body></html>")
    return "\n".join(lines)


#: A ``src`` or ``href`` pointing off the machine.
_EXTERNAL = re.compile(
    r"""(?:src|href)\s*=\s*['"]\s*(?:https?:)?//""", re.IGNORECASE)
#: A script that loads its code from somewhere else — the thing that would
#: actually break the document offline.
_REMOTE_SCRIPT = re.compile(
    r"""<script[^>]*\ssrc\s*=\s*['"]\s*(?:https?:)?//""", re.IGNORECASE)
#: Script *bodies*, which are excluded from the markup scan below.
_SCRIPT_BODY = re.compile(r"<script\b[^>]*>.*?</script\s*>",
                          re.IGNORECASE | re.DOTALL)


def no_external_references(document):
    """True when opening the document fetches nothing from the network.

    The point of the report is that it still works on a laptop with no network
    six months from now, so this is asserted rather than assumed.

    Script *bodies* are excluded from the markup scan, and this is the subtle
    part.  The inlined Plotly bundle carries map support, and that support has
    OpenStreetMap and MapLibre attribution links and an unpkg icon URL written
    into its source.  A naive scan finds those strings and calls a perfectly
    self-contained file external.  They are unreachable here — nothing in a
    report draws a map — and they are inside the very bundle that was inlined
    to avoid a network call in the first place.

    What genuinely would break offline is a ``<script src=...>`` pointing
    elsewhere, so that is checked on its own and never excused.
    """
    document = document or ""
    if _REMOTE_SCRIPT.search(document):
        return False
    return not _EXTERNAL.search(_SCRIPT_BODY.sub("<script></script>", document))
