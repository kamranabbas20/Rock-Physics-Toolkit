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
    FLUID_CASES,
    detect_fluid_cases,
    guess_mnemonics,
    standardise,
)
from avo_qi.core.fluids import (  # noqa: E402
    BRINE_PRESETS,
    CONDITION_DEFAULTS,
    GAS_PRESETS,
    OIL_PRESETS,
    fluid_suite,
)
from avo_qi.core.depth import survey_from_table  # noqa: E402
from avo_qi.core.lithology import GR_METHODS  # noqa: E402
from avo_qi.core.zones import tops_from_table  # noqa: E402
from avo_qi.ui import (  # noqa: E402
    DEMO_WELL,
    DEPTH_REFERENCES,
    apply_depth_references,
    PETRO_CURVES,
    apply_petrophysics,
    file_depth_curves,
    model_fluid_cases,
    vsh_series,
    parameter_defaults,
    petro_sources,
    read_uploaded_table,
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
        "Deviation survey — LAS, CSV or Excel (measured depth, inclination, "
        "azimuth)", type=["las", "csv", "txt", "xlsx", "xls"], key="survey_csv",
        help="A survey is a depth-indexed set of curves, so a LAS is as "
             "natural a format for it as a spreadsheet. Column and mnemonic "
             "names are matched loosely — MD/DEPTH/DEPT, INC/DEVI/DRIFT and "
             "AZI/AZIM/HAZI. Inclination and azimuth in degrees.")
    if _survey_file is not None and st.session_state.get("_survey_file") != _survey_file.name:
        try:
            _table = read_uploaded_table(_survey_file)
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

# ----------------------------------------------------------- petrophysics ---
st.divider()
st.subheader("4 · Petrophysics")

_petro_present = {c: (c in well.df.columns
                      and bool(np.isfinite(well.df[c].to_numpy(float)).any()))
                  for c in PETRO_CURVES}
_petro_missing = [c for c, there in _petro_present.items() if not there]
_sources = petro_sources()

st.caption(
    "**Does this file already carry a petrophysical interpretation, or should "
    "the toolkit compute one?** VSH, PHI and SW drive the lithology classes, "
    "the zone summary's net-to-gross, the forward model and every fluid "
    "substitution, so where they come from decides a great deal. An "
    "interpretation that arrived with the well was made with core, pressures "
    "and local calibration — none of which is here — and should normally be "
    "kept. Computing them is for the well that carries only raw logs."
)

# What the file itself carried, snapshotted before anything was computed into
# the same columns; before the first apply, that is simply what is here now.
_originals = st.session_state.get("petro_original")
_in_file = (set(_originals) if _originals is not None
            else {c for c, there in _petro_present.items() if there})
_status = pd.DataFrame([
    {"curve": c,
     "in the file": "yes" if c in _in_file else "no",
     "coverage now": (f"{np.isfinite(well.df[c].to_numpy(float)).mean():.0%}"
                      if c in well.df.columns else "—"),
     "source": _sources.get(c, "file" if _petro_present[c] else "—")}
    for c in PETRO_CURVES
])
st.dataframe(_status, use_container_width=True, hide_index=True)

_mode_labels = {
    "file": "Use what the file carries",
    "fill": "Compute only what the file is missing",
    "all": "Compute all three here",
}
_default_mode = _petro_options.get("mode") if (
    _petro_options := dict(settings.petrophysics or {})) else (
    "fill" if _petro_missing else "file")
_mode = st.radio(
    "Where VSH, PHI and SW come from", list(_mode_labels),
    format_func=lambda k: _mode_labels[k],
    index=list(_mode_labels).index(_default_mode or "file"),
    horizontal=True, key="petro_mode",
    help="Nothing is recomputed until you apply it, and the file's own curves "
         "are kept, so this is reversible.")

if _petro_missing:
    st.caption("This file carries no " + ", ".join(f"**{c}**" for c in _petro_missing)
               + " — computing " + ("it" if len(_petro_missing) == 1 else "them")
               + " here is the only way to have "
               + ("it" if len(_petro_missing) == 1 else "them") + " at all.")

_defaults, _from_parameters = parameter_defaults(raw)
if _from_parameters:
    st.caption(
        "Defaults below come from this well's own parameter curves — "
        + ", ".join(f"`{v}`" for v in _from_parameters.values())
        + " — which is what its interpretation was actually made with, and a "
          "better starting point than any textbook constant."
    )

_params = dict(settings.petrophysics or {})
if _mode != "file":
    _v, _p, _s = st.tabs(["VSH from GR", "Porosity", "Saturation"])
    with _v:
        _c1, _c2, _c3 = st.columns(3)
        _gr = (well.df["GR"].to_numpy(float) if "GR" in well.df.columns
               else np.array([np.nan]))
        _gr_finite = _gr[np.isfinite(_gr)]
        _params["vsh_method"] = _c1.selectbox(
            "GR transform", list(GR_METHODS), key="petro_vsh_method",
            index=list(GR_METHODS).index(_params.get("vsh_method",
                                                     settings.gr_method)),
            help="The non-linear transforms all read *less* shale than the "
                 "linear index for the same gamma-ray value.")
        _params["gr_clean"] = _c2.number_input(
            "GR clean (API)", value=float(_params.get(
                "gr_clean", _defaults.get("gr_clean",
                                          float(np.percentile(_gr_finite, 5))
                                          if _gr_finite.size else 20.0))),
            key="petro_gr_clean")
        _params["gr_shale"] = _c3.number_input(
            "GR shale (API)", value=float(_params.get(
                "gr_shale", _defaults.get("gr_shale",
                                          float(np.percentile(_gr_finite, 95))
                                          if _gr_finite.size else 120.0))),
            key="petro_gr_shale")
    with _p:
        _c1, _c2, _c3, _c4 = st.columns(4)
        _phi_methods = ["density"] + (["density-neutron"]
                                      if "NPHI" in well.df.columns else [])
        _params["phi_method"] = _c1.selectbox(
            "Porosity from", _phi_methods, key="petro_phi_method",
            help="Density and neutron respond to gas in opposite directions, "
                 "so combining them both estimates porosity and shows gas.")
        _matrix_default = float(_params.get("rho_matrix",
                                            _defaults.get("rho_matrix", 2.65)))
        _params["rho_matrix"] = _c2.number_input(
            "Matrix density (g/cc)", value=_matrix_default, step=0.01,
            format="%.3f", key="petro_rho_matrix",
            help="Quartz 2.65, calcite 2.71, dolomite 2.87. Being 0.05 g/cc "
                 "out moves porosity by about 3 units everywhere.")
        _params["rho_fluid"] = _c3.number_input(
            "Fluid density (g/cc)", value=float(_params.get(
                "rho_fluid", _defaults.get("rho_fluid", 1.0))),
            step=0.01, format="%.3f", key="petro_rho_fluid",
            help="What fills the invaded zone the density tool reads — mud "
                 "filtrate, not necessarily the reservoir fluid.")
        _params["shale_correct"] = _c4.checkbox(
            "Shale-correct", value=bool(_params.get("shale_correct", False)),
            key="petro_shale_correct",
            help="Subtract the clay-bound part: φe = φt − Vsh·φsh. Gassmann "
                 "wants the connected porosity, not the total.")
        if _params["shale_correct"]:
            _params["phi_shale"] = st.number_input(
                "Shale porosity φsh", value=float(_params.get("phi_shale", 0.10)),
                step=0.01, format="%.2f", key="petro_phi_shale")
        if _params["phi_method"] == "density-neutron":
            _params["dn_method"] = st.radio(
                "Combination", ["rms", "average"], horizontal=True,
                key="petro_dn_method",
                help="RMS is the usual gas-bearing combination; the average is "
                     "right in a liquid-filled hole.")
    with _s:
        _c1, _c2, _c3, _c4 = st.columns(4)
        _sw_methods = ["archie"] + (["simandoux"] if "VSH" in well.df.columns
                                    else [])
        _params["sw_method"] = _c1.selectbox(
            "Saturation model", _sw_methods, key="petro_sw_method",
            help="Archie assumes only the pore water conducts. In a shaly sand "
                 "the clay conducts too, so Archie reads too much water — it "
                 "hides pay rather than inventing it.")
        _params["rw"] = _c2.number_input(
            "Rw (ohm·m)", value=float(_params.get("rw", _defaults.get("rw", 0.05))),
            step=0.005, format="%.4f", key="petro_rw")
        _params["m"] = _c3.number_input(
            "Cementation m", value=float(_params.get("m", _defaults.get("m", 2.0))),
            step=0.05, format="%.3f", key="petro_m")
        if _params["sw_method"] == "simandoux":
            _params["r_shale"] = _c4.number_input(
                "Shale resistivity (ohm·m)",
                value=float(_params.get("r_shale", _defaults.get("r_shale", 2.0))),
                step=0.1, format="%.2f", key="petro_r_shale")
        else:
            _params["n"] = _c4.number_input(
                "Saturation n",
                value=float(_params.get("n", _defaults.get("n", 2.0))),
                step=0.05, format="%.3f", key="petro_n")
        _params["a"] = st.number_input(
            "Tortuosity a", value=float(_params.get("a", 1.0)), step=0.05,
            format="%.2f", key="petro_a")
        if "RT" not in well.df.columns:
            st.warning("No resistivity curve here, so no saturation can be "
                       "computed. Assign one above if the file has one.",
                       icon=":material/warning:")

if st.button("Apply interpretation", type="primary", key="apply_petro"):
    _params["mode"] = _mode
    settings.petrophysics = _params
    _notes = apply_petrophysics(well, settings)
    for _note in _notes:
        st.session_state.setdefault("petro_notes", [])
    st.session_state["petro_notes"] = _notes
    st.rerun()

_notes = st.session_state.get("petro_notes") or []
if _notes and any(v == "computed" for v in _sources.values()):
    st.success("  \n".join(f"- {n}" for n in _notes),
               icon=":material/calculate:")
    _both = [c for c in PETRO_CURVES
             if _sources.get(c) == "computed"
             and c in (st.session_state.get("petro_original") or {})]
    if _both:
        _rows = []
        for _curve in _both:
            _theirs = np.asarray(st.session_state["petro_original"][_curve], float)
            _mine = well.df[_curve].to_numpy(float)
            _ok = np.isfinite(_theirs) & np.isfinite(_mine)
            _rows.append({
                "curve": _curve,
                "samples compared": int(_ok.sum()),
                "correlation": (float(np.corrcoef(_mine[_ok], _theirs[_ok])[0, 1])
                                if _ok.sum() > 2 else np.nan),
                "median difference": (float(np.median(_mine[_ok] - _theirs[_ok]))
                                      if _ok.any() else np.nan),
            })
        st.caption(
            "Computed here against what the file carried — the file's curves "
            "are kept, so this comparison stays available and the choice stays "
            "reversible."
        )
        st.dataframe(pd.DataFrame(_rows).round(4), use_container_width=True,
                     hide_index=True)
elif not _petro_missing and _mode == "file":
    st.caption("Nothing is being recomputed; the file's own interpretation is "
               "in use throughout.")

# -------------------------------------------------------------- fluid cases --
st.divider()
st.subheader("5 · Fluid cases")
st.caption(
    "A fluid case is the well as it would log with a different pore fluid, and "
    "comparing them is what separates a fluid response from a lithology one. "
    "Some wells arrive with the cases already computed — `VP_BR`, `VS_OIL`, "
    "`RHOB_GAS` — and those are used as they are. A well that carries only one "
    "set of logs can have them **modelled here** at Batzle-Wang reservoir "
    "conditions, from industry-default fluids you can overwrite."
)

_cases_now = list(well.cases) or ["in situ"]
_modelled = list(well.computed_cases)
_loaded = [c for c in _cases_now if c not in _modelled]
st.dataframe(pd.DataFrame([
    {"case": c,
     "source": "modelled here" if c in _modelled else "read from the file",
     "curves": ", ".join(
         col for col in (f"{n}_{c.upper().replace(' ', '')}" for n in
                         ("VP", "VS", "RHOB")) if col in well.df.columns)
     or "VP, VS, RHOB"}
    for c in _cases_now
]), use_container_width=True, hide_index=True)

_detected = detect_fluid_cases(raw.columns) if raw is not None else {}
_assignable = [c for c in FLUID_CASES]
if raw is not None and len(_detected) > 1:
    with st.expander("Assign the case curves by hand"):
        st.caption(
            "The detector reads the fluid off the mnemonic — `VP_BR` is brine, "
            "`RHOB_GAS` is gas. A file that names its cases `VP_1` and `VP_2`, "
            "or whose *oil* curves are really the in-situ ones, cannot be read "
            "that way and has to be told."
        )
        _columns = ["— none —"] + list(raw.columns)
        _assigned = {}
        _grid = st.columns(len(_assignable))
        for _col, _case in zip(_grid, _assignable):
            _col.markdown(f"**{_case}**")
            _current = (settings.case_mapping.get(_case)
                        or _detected.get(_case) or {})
            _picked = {}
            for _curve in ("VP", "VS", "RHOB"):
                _source = _current.get(_curve)
                _picked[_curve] = _col.selectbox(
                    f"{_curve} ({_case})", _columns,
                    index=_columns.index(_source) if _source in _columns else 0,
                    key=f"case_{_case}_{_curve}", label_visibility="collapsed",
                    placeholder=_curve)
            _kept = {k: v for k, v in _picked.items() if v != "— none —"}
            if len(_kept) == 3:
                _assigned[_case] = _kept
            elif _kept:
                _col.caption("needs all three")
        if st.button("Apply fluid cases", key="apply_cases"):
            if not _assigned:
                st.error("No case has all three of VP, VS and RHOB assigned.")
            else:
                settings.case_mapping = _assigned
                set_well(standardise(raw, mapping=well.mapping, units=raw_units,
                                     depth_unit=depth_unit, name=well.name,
                                     case_mapping=_assigned),
                         raw=raw, units=raw_units)
                settings.case_mapping = _assigned      # set_well clears it
                st.rerun()

_missing_cases = [c for c in ("brine", "oil", "gas") if c not in _cases_now]
if _missing_cases:
    st.caption(
        "This well carries no " + ", ".join(f"**{c}**" for c in _missing_cases)
        + " case. Modelling "
        + ("it" if len(_missing_cases) == 1 else "them")
        + " here writes ordinary fluid cases that flow through every page — the "
          "gather, the AVO classification, the crossplots — exactly as ones "
          "loaded from the LAS would, and each is labelled *(computed)* so a "
          "model is never mistaken for a measurement."
    )

_fluid = dict(settings.fluid_model or {})
if "PHI" not in well.df.columns:
    st.info("Modelling a fluid case needs a porosity curve — Gassmann divides "
            "by it. Assign or compute PHI above.", icon=":material/info:")
else:
    _t1, _t2, _t3 = st.tabs(["Fluids", "Conditions", "What is in the pores now"])
    with _t1:
        _c1, _c2, _c3 = st.columns(3)
        _brine_name = _c1.selectbox(
            "Brine", list(BRINE_PRESETS), key="fluid_brine_preset",
            index=list(BRINE_PRESETS.values()).index(0.035))
        _fluid["salinity"] = BRINE_PRESETS[_brine_name]
        _oil_name = _c2.selectbox("Oil", list(OIL_PRESETS), index=1,
                                  key="fluid_oil_preset")
        _fluid["api"], _fluid["gor"] = OIL_PRESETS[_oil_name]
        _gas_name = _c3.selectbox(
            "Gas", list(GAS_PRESETS), key="fluid_gas_preset",
            index=list(GAS_PRESETS.values()).index(0.65))
        _fluid["gas_gravity"] = GAS_PRESETS[_gas_name]
        st.caption(
            "Industry starting values, not this field's fluids: the salinity, "
            "the API and the GOR are a PVT report's answer where one exists. "
            "A dead oil and a live one are different rocks — a GOR of 100 can "
            "halve the modulus."
        )
    with _t2:
        _c1, _c2, _c3, _c4 = st.columns(4)
        _fluid["pressure_gradient"] = _c1.number_input(
            "Pressure gradient (MPa/km)",
            value=float(_fluid.get("pressure_gradient",
                                   CONDITION_DEFAULTS["pressure_gradient"])),
            step=0.1, format="%.2f", key="fluid_p_grad",
            help="10.0 is a normally pressured column; raise it for overpressure.")
        _fluid["temperature_surface"] = _c2.number_input(
            "Temperature at datum (°C)",
            value=float(_fluid.get("temperature_surface",
                                   CONDITION_DEFAULTS["temperature_surface"])),
            step=1.0, format="%.1f", key="fluid_t_surface")
        _fluid["temperature_gradient"] = _c3.number_input(
            "Geothermal gradient (°C/km)",
            value=float(_fluid.get("temperature_gradient",
                                   CONDITION_DEFAULTS["temperature_gradient"])),
            step=1.0, format="%.1f", key="fluid_t_grad")
        _fluid["datum"] = _c4.number_input(
            "Gradient datum (m MD)", value=float(_fluid.get("datum", 0.0)),
            step=10.0, format="%.1f", key="fluid_datum",
            help="Depth the gradients start from. An offshore well's MD starts "
                 "at a rig floor, not at the sea bed.")
        _preview = fluid_suite(well.df["DEPTH"].to_numpy(float),
                               parameters=_fluid, conditions=_fluid,
                               datum=float(_fluid.get("datum", 0.0)))
        _mid = len(well.df) // 2
        _p1, _p2, _p3 = st.columns(3)
        _p1.metric("Pressure at mid-well", f"{_preview['pressure'][_mid]:,.1f} MPa")
        _p2.metric("Temperature there", f"{_preview['temperature'][_mid]:,.1f} °C")
        _p3.metric("Gas K there", f"{_preview['gas'][0][_mid]:.3f} GPa",
                   help="A fixed table would call it 0.021 GPa at any depth.")
        for _case, _warning in _preview["warnings"].items():
            st.warning(f"{_case}: {_warning}", icon=":material/warning:")
    with _t3:
        _c1, _c2, _c3 = st.columns(3)
        _in_situ = _c1.selectbox(
            "Fluid in the pores now", ["brine", "oil", "gas"],
            index=["brine", "oil", "gas"].index(
                _fluid.get("in_situ_hydrocarbon") or "brine"),
            key="fluid_in_situ",
            help="A well logged in a gas leg is not brine-filled. Substituting "
                 "it as though it were tells Gassmann the rock is stiffer than "
                 "the logs say, and it refuses rather than answering.")
        _fluid["in_situ_hydrocarbon"] = None if _in_situ == "brine" else _in_situ
        _fluid["hydrocarbon_sw"] = _c2.slider(
            "Water left in the hydrocarbon cases", 0.0, 1.0,
            float(_fluid.get("hydrocarbon_sw", 0.2)), 0.05,
            key="fluid_hc_sw",
            help="A reservoir at residual water, not a pore of pure gas — "
                 "which is not a rock that exists.")
        _fluid["mixing"] = _c3.selectbox(
            "Mixing law", ["wood", "patchy", "brie"], key="fluid_mixing",
            help="How the brine and the hydrocarbon share the pore. Wood is "
                 "the finely-mixed limit, where a little gas dominates.")
        if _in_situ != "brine" and "SW" not in well.df.columns:
            st.caption("No SW curve, so the pores are taken as fully "
                       f"{_in_situ}-filled where the substitution runs.")

    _m1, _m2, _m3 = st.columns(3)
    _make = _m1.multiselect("Cases to model", ["brine", "oil", "gas"],
                            default=_missing_cases or ["brine", "oil", "gas"],
                            key="fluid_cases_to_make")
    _k_mineral = _m2.number_input(
        "Mineral K (GPa)", value=float(_fluid.get("k_mineral", 37.0)), step=1.0,
        format="%.1f", key="fluid_k_mineral",
        help="Quartz 37, calcite 76.8. The Rock Physics page mixes it from the "
             "mineral fractions if you need more than one.")
    _fluid["k_mineral"] = _k_mineral
    _limit = _m3.checkbox(
        "Reservoir only", value=bool(_fluid.get("reservoir_only", True)),
        key="fluid_reservoir_only",
        help="Gassmann assumes a connected, isotropic frame. A shale is "
             "neither, and substituting one returns a negative dry frame "
             "rather than an answer.")
    _fluid["reservoir_only"] = _limit

    _reservoir = None
    if _limit:
        _vsh_curve, _vsh_source = vsh_series(well.df, settings)
        if _vsh_source is None:
            st.caption("No VSH or GR curve, so every sample is offered to "
                       "Gassmann and the ones it refuses keep their own logs.")
        else:
            _cut = float(settings.vsh_cutoffs.get("silty sand", 0.35))
            _reservoir = np.isfinite(_vsh_curve) & (_vsh_curve <= _cut)
            st.caption(
                f"Substituting where VSH ≤ {_cut:.2f} — {_reservoir.mean():.0%} "
                "of the well. Everywhere else each case keeps the well's own "
                "curves, so the seal above a substituted sand is still there "
                "and the interface survives."
            )

    if st.button("Model the fluid cases", type="primary", key="model_cases"):
        settings.fluid_model = _fluid
        _overwrites = [c for c in _make if c in _loaded]
        if _overwrites:
            st.warning(
                "Replacing the file's own **" + "**, **".join(_overwrites)
                + "** case(s) with modelled curves for this session.",
                icon=":material/warning:")
        _results = model_fluid_cases(
            well, settings, well.df["PHI"].to_numpy(float), cases=_make,
            k_mineral=_k_mineral, reservoir=_reservoir)
        # Samples left alone for two quite different reasons: outside the
        # reservoir cutoff, which was asked for, and refused by Gassmann, which
        # was not. Reporting them together would hide the second.
        _outside = (np.zeros(len(well.df), bool) if _reservoir is None
                    else ~_reservoir)
        st.session_state["fluid_case_results"] = {
            case: {
                "substituted": int(r["valid"].sum()),
                "of": int(r["valid"].size),
                "outside": int(_outside.sum()),
                "reasons": [str(x) for x in pd.unique(
                    r["reasons"][~r["valid"] & ~_outside]) if str(x).strip()][:3],
            }
            for case, r in _results.items()}
        st.rerun()

    _outcome = st.session_state.get("fluid_case_results")
    if _outcome:
        st.success(
            "  \n".join(
                f"- **{case}**: substituted {row['substituted']:,} of "
                f"{row['of']:,} samples"
                + (f"; {row['outside']:,} outside the reservoir cutoff"
                   if row["outside"] else "")
                + (f"; refused elsewhere ({', '.join(row['reasons'])})"
                   if row["reasons"] else "")
                + ", all of them keeping the well's own curves."
                for case, row in _outcome.items()),
            icon=":material/water_drop:")
        st.caption(
            "Pick a case in the sidebar to carry it through the rest of the "
            "app. A refusal is not a failure to hide: a negative dry frame "
            "means the porosity, the mineral and the measured velocities "
            "disagree, and the rock cannot be what all three say it is."
        )

# ---------------------------------------------------------------- zonation --
st.divider()
st.subheader("6 · Zonation")

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
        "nothing to work from. Give it a zonation here instead — **upload** a "
        "LAS carrying a discrete zone curve, or a tops list as CSV or Excel, "
        "or type the **formation tops** straight into the table. Each zone "
        "runs down to the next top, and the deepest to the bottom of the well."
    )

    _depth_md = well.df["DEPTH"].to_numpy(float)
    _lo = float(np.nanmin(_depth_md)) if _depth_md.size else 0.0
    _hi = float(np.nanmax(_depth_md)) if _depth_md.size else 0.0
    st.caption(f"This well runs {_lo:,.1f} – {_hi:,.1f} m MD.")

    _upload = st.file_uploader(
        "Zonation — LAS, CSV or Excel", type=["las", "csv", "txt", "xlsx", "xls"],
        key="tops_csv",
        help="Two shapes are read. A tops list needs a name column "
             "(zone/name/formation/marker) and a depth column "
             "(top/depth/md/tvd). A LAS instead carries a discrete zone curve "
             "— a code per sample — because its data section is numeric and "
             "has nowhere to put a name; intervals are then built where the "
             "code changes, and the codes stand in as their own names.")
    if _upload is not None and st.session_state.get("_tops_file") != _upload.name:
        try:
            _read = read_uploaded_table(_upload)
            settings.zone_tops = tops_from_table(
                _read, names=settings.zone_names or None
            )[["zone", "top"]].to_dict("records")
            st.session_state["_tops_file"] = _upload.name
            st.session_state.pop("zone_cache", None)
            _codes = all(str(t["zone"]).strip().lstrip("-").isdigit()
                         for t in settings.zone_tops)
            st.success(
                f"Read {len(settings.zone_tops)} top(s) from {_upload.name}."
                + (" A LAS carries codes rather than names — type over them in "
                   "the table below to name the zones." if _codes else ""))
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
st.subheader("7 · Quality control")

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
st.subheader("8 · Analysis window")

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
