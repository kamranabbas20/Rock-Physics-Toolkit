"""Page 1 — load a LAS, assign its curves, and QC them before anything else.

Everything downstream treats the standardised well as physics, so this is the
page that decides what that physics is: which curve is Vp, what units it is
in, which samples are nulls or spikes, and which interval to analyse.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from avo_qi.core.qc import (  # noqa: E402
    DEFAULT_RANGES,
    curve_summary,
    depth_qc,
    despike,
    qc_flags,
    replace_nulls,
)
from avo_qi.io.loader import (  # noqa: E402
    CANONICAL,
    detect_fluid_cases,
    guess_mnemonics,
    standardise,
)
from avo_qi.ui import (  # noqa: E402
    DEMO_WELL,
    load_demo_well,
    load_uploaded_well,
    log_track_figure,
    page_setup,
    set_well,
    sidebar,
)

page_setup("Load & QC", icon=":mag:")
settings = sidebar(show_wavelet=False, show_angles=False, show_classifier=False)

# ------------------------------------------------------------------ load ---
st.subheader("1 · Load a well")
c1, c2 = st.columns([3, 1])
uploaded = c1.file_uploader(
    "LAS, CSV or Excel — the well must already contain Vp, Vs and RHOB "
    "(or sonic equivalents). Fluid-substituted curves such as VP_BR are picked "
    "up automatically.",
    type=["las", "csv", "txt", "xlsx", "xls"],
)
depth_unit = c2.selectbox("Depth unit in file", ["m", "ft"])
if c2.button("Load demo well", use_container_width=True):
    load_demo_well()
    st.session_state.pop("_uploaded_name", None)
    st.rerun()

if uploaded is not None and st.session_state.get("_uploaded_name") != uploaded.name:
    try:
        load_uploaded_well(uploaded, depth_unit=depth_unit)
        st.session_state["_uploaded_name"] = uploaded.name
        st.success(f"Loaded {uploaded.name}")
        st.rerun()
    except Exception as exc:                    # a bad file must not kill the page
        st.error(f"Could not read {uploaded.name}: {exc}")

well = st.session_state.get("well")
raw = st.session_state.get("raw_df")
raw_units = st.session_state.get("raw_units") or {}

if well is None:
    st.info(
        "No well loaded. Upload one above, or load the bundled demo well — a "
        "three-layer model with brine, oil and gas cases."
    )
    st.caption(f"Demo file: `{os.path.relpath(DEMO_WELL, _ROOT)}`")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Well", well.name)
c2.metric("Samples", f"{len(well.df):,}")
c3.metric("Curves in file", len(raw.columns) if raw is not None else len(well.df.columns))
c4.metric("Fluid cases", len(well.cases) or 1)
if well.notes:
    with st.expander("Load notes", expanded=False):
        for note in well.notes:
            st.write(f"- {note}")

# ------------------------------------------------------------ assignment ---
st.divider()
st.subheader("2 · Assign curves")

if raw is None:
    st.caption("This well was not loaded from a file, so there is nothing to remap.")
else:
    st.caption(
        "Sonic curves (DT / DTS) are converted to velocity on load, as are ft/s "
        "and kg/m³. Check the guesses below — a wrong assignment is silent "
        "everywhere else in the toolkit."
    )
    with st.expander("Curves in the file", expanded=False):
        rows = []
        for column in raw.columns:
            values = pd.to_numeric(raw[column], errors="coerce").to_numpy(float)
            finite = values[np.isfinite(values)]
            rows.append({
                "curve": column,
                "unit": raw_units.get(column, ""),
                "n": int(finite.size),
                "min": float(finite.min()) if finite.size else np.nan,
                "max": float(finite.max()) if finite.size else np.nan,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    options = ["— none —"] + list(raw.columns)
    current = well.mapping or guess_mnemonics(raw.columns)
    cols = st.columns(3)
    new_mapping = {}
    for i, canonical in enumerate(CANONICAL):
        source = current.get(canonical)
        index = options.index(source) if source in options else 0
        picked = cols[i % 3].selectbox(canonical, options, index=index,
                                       key=f"assign_{canonical}")
        if picked != "— none —":
            new_mapping[canonical] = picked

    cases = detect_fluid_cases(raw.columns)
    if len(cases) > 1:
        st.caption("Fluid cases detected: " + " · ".join(
            f"**{name}** ({', '.join(curves.values())})" for name, curves in cases.items()
        ))

    if st.button("Apply assignment", type="primary"):
        try:
            set_well(standardise(raw, mapping=new_mapping, units=raw_units,
                                 depth_unit=depth_unit, name=well.name),
                     raw=raw, units=raw_units)
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

missing = [c for c in ("VP", "VS", "RHOB") if c not in well.df.columns]
if missing:
    st.error(
        f"Missing required curve(s): {', '.join(missing)}. Assign them above — "
        "the rest of the toolkit cannot run without all three."
    )
    st.stop()

# -------------------------------------------------------------------- QC ---
st.divider()
st.subheader("3 · Quality control")

frame = well.frame(settings.case)
numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]

c1, c2, c3 = st.columns(3)
strip_nulls = c1.toggle("Convert null sentinels to blanks", value=True,
                        help="−999.25 and friends become missing values.")
do_despike = c2.toggle("Despike Vp, Vs and RHOB", value=False)
spike_threshold = c3.slider("Spike sensitivity (robust σ)", 3.0, 12.0, 5.0, 0.5,
                            help="Lower is more aggressive.")

work = frame.copy()
if strip_nulls:
    for column in numeric:
        work[column] = replace_nulls(work[column].to_numpy(float))

repaired_counts = {}
if do_despike:
    for column in ("VP", "VS", "RHOB"):
        if column in work.columns:
            values, flags = despike(work[column].to_numpy(float),
                                    threshold=spike_threshold)
            work[column] = values
            repaired_counts[column] = int(flags.sum())
    if repaired_counts:
        st.caption("Repaired spikes — " + ", ".join(
            f"{k}: {v}" for k, v in repaired_counts.items()))

# --- depth axis
report = depth_qc(work["DEPTH"].to_numpy(float))
st.markdown("**Depth axis**")
d1, d2, d3, d4 = st.columns(4)
d1.metric("Interval", f"{report['start']:.1f} – {report['stop']:.1f} m")
d2.metric("Step", f"{report['step']:.4f} m" if np.isfinite(report["step"]) else "—")
d3.metric("Duplicates", report["duplicates"],
          delta=None if report["duplicates"] == 0 else "check", delta_color="inverse")
d4.metric("Gaps", report["gaps"],
          delta=None if report["gaps"] == 0 else "check", delta_color="inverse")
if not report["monotonic_increasing"]:
    st.warning("Depth is not strictly increasing — duplicated or unsorted samples.",
               icon=":material/warning:")
if report["step_varies"]:
    st.info("Sample interval varies through the well; the time conversion "
            "integrates the actual spacing, so this is tolerated.",
            icon=":material/info:")

# --- per-curve summary
st.markdown("**Curves**")
summary = curve_summary(work[numeric], units=well.units)
display = summary.copy()
display["present"] = (display["present"] * 100).round(1)
st.dataframe(
    display.rename(columns={"present": "present %"}).round(4),
    use_container_width=True, hide_index=True,
    column_config={
        "present %": st.column_config.ProgressColumn(
            "present %", min_value=0, max_value=100, format="%.1f%%"),
    },
)
short = summary[summary["present"] < 0.9]["curve"].tolist()
if short:
    st.warning(f"Under 90% coverage: {', '.join(short)}.", icon=":material/warning:")

# --- per-sample flags
flags = qc_flags(work, spike_threshold=spike_threshold)
flagged = int(flags["any_flag"].sum())
st.markdown("**Sample checks**")
counts = flags.drop(columns=["any_flag"]).sum()
counts = counts[counts > 0]
if counts.empty:
    st.success(f"No sample failed a check across {len(work):,} samples.",
               icon=":material/check_circle:")
else:
    st.dataframe(
        counts.rename("samples").reset_index().rename(columns={"index": "check"}),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        f"{flagged:,} of {len(work):,} samples ({flagged / max(len(work), 1):.1%}) "
        "fail at least one check. `vs_exceeds_vp` and `vpvs_below_limit` usually "
        "mean a bad or swapped shear log; range breaches are often unconverted units."
    )

# --- where the flags are
if flagged:
    fig = go.Figure()
    for name in counts.index:
        mask = flags[name].to_numpy(bool)
        if not mask.any():
            continue
        fig.add_trace(go.Scatter(
            x=np.full(int(mask.sum()), name), y=work["DEPTH"].to_numpy(float)[mask],
            mode="markers", name=name, marker=dict(size=6, opacity=0.7),
        ))
    fig.update_yaxes(autorange="reversed", title_text="DEPTH (m)")
    fig.update_layout(height=420, margin=dict(l=60, r=20, t=30, b=80),
                      showlegend=False, xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------- analysis window ---
st.divider()
st.subheader("4 · Analysis window")

depth = work["DEPTH"].to_numpy(float)
lo, hi = float(np.nanmin(depth)), float(np.nanmax(depth))
window = st.slider("Depth range to carry forward (m)", lo, hi, (lo, hi), step=0.5)
drop_flagged = st.toggle("Also drop samples that failed a check", value=False)

keep = (depth >= window[0]) & (depth <= window[1])
if drop_flagged:
    keep &= ~flags["any_flag"].to_numpy(bool)

st.caption(f"{int(keep.sum()):,} of {len(work):,} samples kept.")
st.plotly_chart(
    log_track_figure(work[keep].reset_index(drop=True),
                     curves=[c for c in ("VP", "VS", "RHOB", "GR", "VSH") if c in work.columns],
                     height=620),
    use_container_width=True,
)

c1, c2 = st.columns(2)
if c1.button("Apply to the rest of the toolkit", type="primary",
             use_container_width=True,
             help="Writes the cleaned, windowed well back so every other page uses it."):
    cleaned = well.frame(settings.case) if not (strip_nulls or do_despike) else work
    trimmed = cleaned[keep].reset_index(drop=True)
    updated = type(well)(df=trimmed, mapping=well.mapping, units=well.units,
                         notes=list(well.notes) + [
                             f"QC: kept {int(keep.sum())} of {len(work)} samples "
                             f"over {window[0]:.1f}–{window[1]:.1f} m"
                             + (", despiked" if do_despike else "")],
                         name=well.name, cases=list(well.cases),
                         active_case=well.active_case)
    set_well(updated, raw=raw, units=raw_units)
    st.success("Applied. The other pages now use the cleaned well.")
    st.rerun()

c2.download_button("Download QC flags (CSV)",
                   flags.assign(DEPTH=work["DEPTH"]).to_csv(index=False).encode(),
                   file_name=f"{well.name}_qc_flags.csv", mime="text/csv",
                   use_container_width=True)

with st.expander("Expected ranges used by the checks"):
    st.dataframe(
        pd.DataFrame([{"curve": k, "min": v[0], "max": v[1]}
                      for k, v in DEFAULT_RANGES.items()]),
        use_container_width=True, hide_index=True,
    )
