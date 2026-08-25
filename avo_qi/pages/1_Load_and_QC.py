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

import io as _stdlib_io  # noqa: E402

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
from avo_qi.core.depth import survey_from_table  # noqa: E402
from avo_qi.core.zones import zones_from_tops  # noqa: E402
from avo_qi.ui import (  # noqa: E402
    DEMO_WELL,
    DEPTH_REFERENCES,
    apply_depth_references,
    file_depth_curves,
    well_zones,
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

# ------------------------------------------------------- depth reference ----
st.divider()
st.subheader("3 · Depth reference")
st.caption(
    "Measured depth runs along the hole and starts at a rig floor, so it is "
    "not comparable between wells and is not the depth anything in the earth "
    "responds to. **TVDSS** — true vertical depth below mean sea level — is "
    "what makes two wells line up; **TVDBML**, below the mud line, is what "
    "compaction actually follows, and porosity and velocity trends with it. "
    "Both are needed to relate AVO class to depth rather than to hole length."
)

_header = st.session_state.get("las_header") or {}
_from_file = file_depth_curves()
if _from_file:
    st.success(
        "This file already carries " + ", ".join(f"**{c}**" for c in
                                                 sorted(_from_file))
        + ". Those curves are used as they are — they were made with the "
          "survey and the datum the well was actually drilled on.",
        icon=":material/check_circle:")

_c1, _c2, _c3 = st.columns(3)
_kb_default = (settings.kb_elevation if settings.kb_elevation is not None
               else _header.get("kb_elevation"))
_wd_default = (settings.water_depth if settings.water_depth is not None
               else _header.get("water_depth"))
_kb = _c1.number_input(
    "Drilling datum above MSL (m)", value=_kb_default, step=0.1, format="%.2f",
    key="kb_elevation_input", placeholder="KB / derrick floor elevation",
    help="Height of the depth reference — kelly bushing, derrick floor or "
         "rotary table — above mean sea level. TVDSS = TVD − this.")
_wd = _c2.number_input(
    "Water depth (m)", value=_wd_default, step=1.0, format="%.1f",
    key="water_depth_input", placeholder="sea level to seabed",
    help="TVDBML = TVDSS − this. Zero for a land well.")
_tvd_choice = _c3.radio(
    "True vertical depth", ["Not known", "Vertical well (TVD = MD)",
                            "From a deviation survey"],
    index=(1 if settings.vertical_well else
           2 if settings.deviation_survey else 0),
    key="tvd_source",
    help="A 30° hole at 4000 m MD is about 200 m shallower in TVD than in MD, "
         "so this is not a formality.")
if _header.get("kb_elevation") is not None or _header.get("water_depth") is not None:
    st.caption("Pre-filled from the LAS header; overwrite if it is wrong.")

if _tvd_choice == "From a deviation survey":
    _survey_file = st.file_uploader(
        "Deviation survey (CSV: measured depth, inclination, azimuth)",
        type=["csv", "txt"], key="survey_csv",
        help="Column names are matched loosely — MD/DEPTH, INC/DEVI/DRIFT and "
             "AZI/AZIM/HAZI. Inclination and azimuth in degrees.")
    if _survey_file is not None and st.session_state.get("_survey_file") != _survey_file.name:
        try:
            _table = pd.read_csv(_stdlib_io.BytesIO(_survey_file.getvalue()),
                                 sep=None, engine="python")
            _md, _inc, _azi = survey_from_table(_table)
            settings.deviation_survey = [
                {"md": float(a), "inc": float(b), "azi": float(c)}
                for a, b, c in zip(_md, _inc, _azi)]
            st.session_state["_survey_file"] = _survey_file.name
            st.success(f"Read {len(settings.deviation_survey)} survey stations "
                       f"from {_survey_file.name}")
        except Exception as exc:
            st.error(f"Could not read the survey: {exc}")
    if settings.deviation_survey:
        _survey_frame = pd.DataFrame(settings.deviation_survey)
        _s1, _s2, _s3 = st.columns(3)
        _s1.metric("Stations", len(_survey_frame))
        _s2.metric("Survey to", f"{_survey_frame['md'].max():,.0f} m MD")
        _s3.metric("Maximum inclination", f"{_survey_frame['inc'].max():.1f}°")
        with st.expander("Survey stations"):
            st.dataframe(_survey_frame, use_container_width=True, hide_index=True)

if st.button("Apply depth reference", type="primary", key="apply_depth"):
    settings.kb_elevation = None if _kb is None else float(_kb)
    settings.water_depth = None if _wd is None else float(_wd)
    settings.vertical_well = _tvd_choice == "Vertical well (TVD = MD)"
    if _tvd_choice != "From a deviation survey":
        settings.deviation_survey = []
    try:
        apply_depth_references(well, settings)
        st.rerun()
    except ValueError as exc:
        st.error(str(exc))

_resolved = [c for c in DEPTH_REFERENCES if c in well.df.columns]
if _resolved:
    _cols = st.columns(len(_resolved) + 1)
    _deepest = int(np.argmax(well.df["DEPTH"].to_numpy(float)))
    _cols[0].metric("MD at TD", f"{well.df['DEPTH'].iloc[_deepest]:,.1f} m")
    for _col, _name in zip(_cols[1:], _resolved):
        _col.metric(f"{_name} at TD", f"{well.df[_name].iloc[_deepest]:,.1f} m")
    st.caption(
        "Every reference increases downwards: TVDSS is positive below mean sea "
        "level, TVDBML positive below the seabed. The reflector table on the "
        "AVO Classification page carries both per event, which is what lets "
        "class be plotted against true depth rather than hole length."
    )
else:
    st.info(
        "No vertical reference yet. Without one, every depth in the toolkit is "
        "measured depth — fine within this well, wrong between wells.",
        icon=":material/info:")

# ---------------------------------------------------------------- zonation --
st.divider()
st.subheader("4 · Zonation")

_has_zone_curve = "ZONE" in well.df.columns
if _has_zone_curve:
    _intervals = well_zones(well, settings)
    st.success(
        f"This well carries a discrete zone curve — "
        f"{0 if _intervals is None else len(_intervals)} interval(s). Formation "
        "tops are not needed, and the curve is used in preference to them "
        "because it is per-sample and cannot be mistyped.",
        icon=":material/check_circle:")
    if _intervals is not None and len(_intervals):
        st.dataframe(_intervals.round(2), use_container_width=True,
                     hide_index=True, height=200)
else:
    st.caption(
        "Many wells carry no discrete zone curve — this one does not — which "
        "leaves the zone filter, the zone-boundary flag on each reflector and "
        "the **Zone summary** on the AVO Classification page — thickness, "
        "net-to-gross, log averages and the events inside each zone — with "
        "nothing to work from. Enter **formation tops** here instead: a name "
        "and the measured depth it starts at. Each zone runs down to the next "
        "top, and the deepest to the bottom of the well."
    )

    _depth_md = well.df["DEPTH"].to_numpy(float)
    _lo = float(np.nanmin(_depth_md)) if _depth_md.size else 0.0
    _hi = float(np.nanmax(_depth_md)) if _depth_md.size else 0.0
    st.caption(f"This well runs {_lo:,.1f} – {_hi:,.1f} m MD.")

    _upload = st.file_uploader(
        "Tops as CSV (a name column and a depth column)", type=["csv", "txt"],
        key="tops_csv",
        help="Column names are matched loosely: zone/name/formation/marker "
             "for the name, and top/depth/md/tvd for the depth.")
    if _upload is not None and st.session_state.get("_tops_file") != _upload.name:
        try:
            _read = pd.read_csv(_stdlib_io.BytesIO(_upload.getvalue()))
            settings.zone_tops = zones_from_tops(_read)[["zone", "top"]].to_dict("records")
            st.session_state["_tops_file"] = _upload.name
            st.session_state.pop("zone_cache", None)
            st.success(f"Read {len(settings.zone_tops)} top(s) from {_upload.name}.")
            st.rerun()
        except Exception as exc:              # a bad file must not kill the page
            st.error(f"Could not read {_upload.name}: {exc}")

    _seed = pd.DataFrame(settings.zone_tops or [{"zone": "", "top": None}])
    _edited = st.data_editor(
        _seed, num_rows="dynamic", use_container_width=True, key="tops_editor",
        column_config={
            "zone": st.column_config.TextColumn("Zone / formation", required=False),
            "top": st.column_config.NumberColumn("Top (m MD)", format="%.2f"),
        })

    _clean = [
        {"zone": str(r["zone"]).strip(), "top": float(r["top"])}
        for _, r in _edited.iterrows()
        if str(r.get("zone", "")).strip() and pd.notna(r.get("top"))
    ]
    if _clean != list(settings.zone_tops or []):
        settings.zone_tops = _clean
        st.session_state.pop("zone_cache", None)
        st.rerun()

    if settings.zone_tops:
        _intervals = well_zones(well, settings)
        # Test the *interval*, not the top. A top above the first sample still
        # labels everything below it, so comparing the top against the well's
        # range flags a zone that is working perfectly well.
        _counts = [
            int(((_depth_md >= float(r["top"])) &
                 (_depth_md < (float(r["base"]) if pd.notna(r["base"]) else np.inf))
                 ).sum())
            for _, r in _intervals.iterrows()
        ]
        _empty = [str(_intervals["zone"].iloc[i])
                  for i, n in enumerate(_counts) if n == 0]
        if _empty:
            st.warning(
                f"{len(_empty)} zone(s) contain no samples of this well "
                f"({', '.join(_empty)}) — their interval falls outside "
                f"{_lo:,.1f}–{_hi:,.1f} m MD, so they label nothing.",
                icon=":material/warning:")
        st.dataframe(_intervals.round(2), use_container_width=True,
                     hide_index=True, height=200)
        st.caption(
            f"{len(_intervals)} interval(s). Pick which to analyse in the "
            "sidebar; the reflector table will then carry a zone per event and "
            "flag the ones sitting on a boundary."
        )
    else:
        st.caption("No tops entered, so nothing is filtered by zone.")

# -------------------------------------------------------------------- QC ---
st.divider()
st.subheader("5 · Quality control")

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
st.subheader("6 · Analysis window")

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
