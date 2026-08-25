"""AVO & QI Well Analysis Toolkit — Streamlit entry point.

Run from the repository root::

    streamlit run avo_qi/app.py
"""

from __future__ import annotations

import os
import sys

# Streamlit puts this script's directory on sys.path, not the repository root,
# so make the `avo_qi` package importable before anything else is imported.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st  # noqa: E402

from avo_qi import __version__  # noqa: E402
from avo_qi.ui import get_well, load_demo_well, page_setup, sidebar  # noqa: E402

page_setup("AVO & QI Well Analysis Toolkit")
settings = sidebar()

st.markdown(
    """
Takes a well that **already has Vp, Vs and RHOB** and carries it through the
standard quantitative-interpretation workflow: rock-physics crossplots, a
synthetic angle gather, and an intercept–gradient AVO classification of every
reflector picked off the trace itself.

Where a well arrives short of something — a vertical depth reference, a
petrophysical interpretation, fluid cases — the toolkit **asks** rather than
assuming, and labels anything it computes *(computed)* wherever it appears, so
a model of the well is never mistaken for a measurement of it.
"""
)

PAGES = [
    (":material/search:", "Load & QC",
     "Read a LAS, CSV or Excel well and decide what the physics is: curves "
     "and units, TVD from a survey with TVDSS and TVDBML, whether to keep the "
     "file's own VSH/PHI/SW, fluid cases assigned or modelled, zonation, then "
     "nulls, spikes and the elastic consistency checks."),
    (":material/scatter_plot:", "Data & Crossplots",
     "Log tracks and the QI crossplots — AI vs Vp/Vs, λρ–μρ, IP–IS, Poisson "
     "vs AI, and EEI over a χ sweep that reports the χ best correlated with "
     "Sw, Vsh or φ. Every fluid case can be overlaid to show the fluid vector."),
    (":material/waves:", "Synthetic Gather",
     "Ricker, Ormsby or an uploaded wavelet; exact Zoeppritz or the "
     "Aki-Richards linearisation; the gather as variable density or wiggle, "
     "the near / mid / far and full stacks, and CSV, NPY or SEG-Y export."),
    (":material/analytics:", "AVO Classification",
     "The trace decides where the reflectors are — every turning point above "
     "the amplitude cut. Each is blocked on its own lobe, fitted for A and B, "
     "classified, and set against a robust background trend. Then the zone "
     "summary, tuning, a wedge model and a self-contained HTML report."),
    (":material/landscape:", "Rock Physics",
     "A rock physics template in AI–Vp/Vs, the diagnostic bounds and trends — "
     "Hashin-Shtrikman, Hertz-Mindlin, Castagna, Gardner — a per-sample "
     "forward model with its misfit and an optional Monte Carlo band, and "
     "Gassmann substitution at Batzle-Wang reservoir conditions."),
    (":material/compare_arrows:", "Multi-well",
     "Every well in the library through the same pipeline, side by side: a "
     "background trend fitted across wells rather than in one hole, curves and "
     "picked events against TVDSS or TVDBML, the class mix per well, and "
     "events by zone using each well's own tops."),
]

_cards = list(st.columns(3)) + list(st.columns(3))
for _column, (_icon, _name, _blurb) in zip(_cards, PAGES):
    with _column:
        with st.container(border=True):
            st.markdown(f"#### {_icon} {_name}")
            st.caption(_blurb)

st.divider()

well = get_well()
if well is None:
    st.subheader("Get started")
    st.write(
        "Load the bundled three-layer demo well — shale over a Class III gas sand "
        "over shale, with a brine sand and a cemented streak below, carrying its "
        "brine, oil and gas cases — or upload your own on the **Load & QC** page. "
        "Load a second well there too and they join a library rather than "
        "replacing each other, each keeping its own tops, datum and fluid cases."
    )
    if st.button("Load demo well", type="primary"):
        load_demo_well()
        st.rerun()
else:
    st.subheader(f"Loaded well: {well.name}")
    c1, c2, c3, c4 = st.columns(4)
    depth = well.depth
    c1.metric("Samples", f"{len(well.df):,}")
    c2.metric("Depth range", f"{depth.min():.0f} – {depth.max():.0f} m")
    c3.metric("Curves", len(well.df.columns))
    c4.metric("Angles", f"{settings.angles.size}")
    if well.notes:
        with st.expander("Load notes"):
            for note in well.notes:
                st.write(f"- {note}")
    st.dataframe(well.df.head(20), use_container_width=True)
    st.caption("Pick a page from the sidebar to continue.")

st.divider()
st.caption(
    f"AVO & QI Well Analysis Toolkit v{__version__} · "
    "Reflectivity after Aki & Richards (2002); AVO classes after Rutherford & "
    "Williams (1989) and Castagna & Swan (1997); intercept–gradient after Shuey (1985)."
)
