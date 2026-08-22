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

page_setup("AVO & QI Well Analysis Toolkit", icon=":chart_with_upwards_trend:")
settings = sidebar()

st.markdown(
    """
Takes a well that **already has Vp, Vs and RHOB** and carries it through the
standard quantitative-interpretation workflow: rock-physics crossplots, a
synthetic angle gather, and an intercept–gradient AVO classification of every
reflector.

A well that arrives with its fluid cases already substituted is the normal
path, and they are read as they are. The **Rock Physics** page can also model
them here: a forward model driven from VSH, PHIT and SW, and Gassmann fluid
substitution at Batzle-Wang reservoir conditions. Anything computed here is
labelled *(computed)* wherever it appears, so a model of the well is never
mistaken for a measurement of it.
"""
)

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.subheader("1 · Data & Crossplots")
    st.write(
        "Load a LAS, CSV or Excel well, remap its mnemonics, and inspect the log "
        "tracks. Then the QI crossplots: AI vs Vp/Vs, λρ–μρ, IP–IS, Poisson vs AI, "
        "and EEI over a χ sweep."
    )
with col_b:
    st.subheader("2 · Synthetic Gather")
    st.write(
        "Choose a wavelet and an angle range, build the angle gather with exact "
        "Zoeppritz or the Aki-Richards linearisation, and read the near / mid / far "
        "and full stacks. Export as CSV, NPY or SEG-Y."
    )
with col_c:
    st.subheader("3 · AVO Classification")
    st.write(
        "Fit intercept A and gradient B for every reflector with both Shuey and "
        "Aki-Richards, classify I / IIp / IIn / III / IV, and inspect the A–B "
        "crossplot against a robust background trend."
    )

st.divider()

well = get_well()
if well is None:
    st.subheader("Get started")
    st.write(
        "Load the bundled three-layer demo well — shale over a Class III gas sand "
        "over shale, with a brine sand and a cemented streak below — or upload your "
        "own on the **Data & Crossplots** page."
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
