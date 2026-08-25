"""Shared Streamlit plumbing: session state, the common sidebar, and plots.

Everything Streamlit- or Plotly-flavoured lives here or in ``pages/`` so that
``core/`` stays importable without a UI stack.
"""

from __future__ import annotations

import copy
import functools
import io as _stdlib_io
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from avo_qi.core.attributes import acoustic_impedance, vpvs
from avo_qi.core.lithology import (
    DEFAULT_VSH_CUTOFFS,
    LITHOLOGIES,
    UNDEFINED,
    classify_lithology,
    vsh_from_gr,
)
from avo_qi.core.wavelet import bandpass_ormsby, load_wavelet, ricker
from avo_qi.core.zones import (UNZONED, assign_zones, zones_from_curve,
                               zones_from_tops)
from avo_qi.core.depth import depth_references, survey_from_table
from avo_qi.io.loader import (CANONICAL, add_standard_cases, depth_to_twt,
                              detect_fluid_cases, read_las_header, read_well,
                              resample_to_time, standardise)

DEMO_WELL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data", "demo_well.las")

#: Fixed colours per lithology, cleanest to shaliest.
LITHOLOGY_COLOURS = {
    "sand": "#f0c419",
    "silty sand": "#d99a3a",
    "silt": "#8c8060",
    "shale": "#5b6b73",
    UNDEFINED: "#c9c9c9",
}

#: Fixed colours per fluid case, shared by every plot in the app.
CASE_COLOURS = {
    "in situ": "#333333",
    "brine": "#1f77b4",
    "oil": "#2ca02c",
    "gas": "#d62728",
}

#: Fixed colours per AVO class, shared by every plot in the app.
CLASS_COLOURS = {
    "I": "#1f77b4",
    "IIp": "#17becf",
    "IIn": "#9467bd",
    "III": "#d62728",
    "IV": "#ff7f0e",
    "background/other": "#9e9e9e",
}


def add_repo_root_to_path():
    """Make ``import avo_qi`` work when Streamlit runs a script inside it."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)


@dataclass
class Settings:
    """The shared analysis settings carried across pages."""

    dt: float = 0.001
    t0: float = 1.6
    angle_min: float = 0.0
    angle_max: float = 40.0
    angle_step: float = 2.0
    method: str = "zoeppritz"
    a_tol: float = 0.02
    wavelet_kind: str = "Ricker"
    ricker_freq: float = 30.0
    ormsby: tuple = (5.0, 10.0, 60.0, 80.0)
    wavelet_length: float = 0.128
    threshold: float = 0.05
    case: str = None
    zones: list = field(default_factory=list)
    zone_names: dict = field(default_factory=dict)
    #: Formation tops entered by hand, for a well whose LAS carries no
    #: discrete zone curve: a list of ``{"zone": name, "top": depth_md}``.
    zone_tops: list = field(default_factory=list)
    #: Depth reference, all of it belonging to the loaded well: the height of
    #: the drilling datum above mean sea level, the water depth, a deviation
    #: survey as ``[{"md": .., "inc": .., "azi": ..}, ...]``, and whether the
    #: hole was declared vertical.
    kb_elevation: float = None
    water_depth: float = None
    deviation_survey: list = field(default_factory=list)
    vertical_well: bool = False
    #: How VSH, PHI and SW are obtained — ``mode`` is ``"file"``, ``"fill"`` or
    #: ``"all"`` — and the parameters of each transform.
    petrophysics: dict = field(default_factory=dict)
    #: Pore-fluid parameters and conditions for modelling the fluid cases a
    #: well does not carry: salinity, API, GOR, gas gravity, the pressure and
    #: temperature gradients, and what is in the pores now.
    fluid_model: dict = field(default_factory=dict)
    #: An explicit ``{case: {curve: column}}`` assignment, for a well whose
    #: fluid-case curves are not named the way the detector expects.
    case_mapping: dict = field(default_factory=dict)
    vsh_cutoffs: dict = field(default_factory=lambda: dict(DEFAULT_VSH_CUTOFFS))
    lithologies: list = field(default_factory=lambda: list(LITHOLOGIES) + [UNDEFINED])
    gr_method: str = "linear"
    near: tuple = (0.0, 12.0)
    mid: tuple = (13.0, 26.0)
    far: tuple = (27.0, 40.0)
    wavelet_meta: dict = field(default_factory=dict)

    @property
    def angles(self):
        step = max(float(self.angle_step), 0.1)
        return np.arange(float(self.angle_min), float(self.angle_max) + step / 2, step)


# -------------------------------------------------------------- brand -----
#: Where the SVG assets live.
ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

#: The chrome colour, matching ``.streamlit/config.toml``. It appears in none
#: of the semantic palettes in ``ui_colours.py`` on purpose: an accent that is
#: also a class colour makes a button look like a finding.
BRAND_PETROL = "#0E5A6B"


@functools.lru_cache(maxsize=8)
def brand(name="mark.svg"):
    """One of the brand assets, as SVG markup.

    The *markup* rather than the path: ``st.image`` resolves a relative path
    against the working directory, and the app is run from the repository root
    but the tests run each page from wherever pytest was started.
    """
    with open(os.path.join(ASSETS, name), "r", encoding="utf-8") as handle:
        return handle.read()


# --------------------------------------------------------------- state -----
def page_setup(title, icon=None):
    """Page config, the favicon, the sidebar logo and the page title.

    Every page calls this, so the identity is set in one place.  ``icon`` still
    accepts an emoji for a caller that wants one; left alone, the page takes
    the toolkit's own mark.
    """
    st.set_page_config(page_title=f"{title} | AVO & QI Toolkit",
                       page_icon=icon or brand("mark.svg"), layout="wide")
    # The lockup sits above the sidebar navigation; the mark stands in for it
    # when the sidebar is collapsed.
    st.logo(brand("logo.svg"), icon_image=brand("mark.svg"), size="large")
    st.title(title)


def get_settings():
    if "settings" not in st.session_state:
        st.session_state["settings"] = Settings()
    return st.session_state["settings"]


#: Settings fields that describe **one well** rather than the session. They
#: travel with the well: a rig floor height, a set of tops and a fluid model
#: are facts about the hole they were entered against, and applying one well's
#: to another puts it on the wrong datum.
WELL_SETTINGS = (
    "case", "zones", "zone_names", "zone_tops", "kb_elevation", "water_depth",
    "deviation_survey", "vertical_well", "petrophysics", "fluid_model",
    "case_mapping",
)

#: Session-state keys that likewise belong to the active well.
WELL_SESSION_KEYS = (
    "raw_df", "raw_units", "las_header", "file_depth_curves", "petro_source",
    "petro_original", "petro_notes", "fluid_case_results", "mc_result",
    "_uploaded_name", "_tops_file", "_survey_file",
)


def wells():
    """The well library: ``{name: WellData}``, in the order they were loaded."""
    return st.session_state.setdefault("wells", {})


def get_well():
    """The active well, or None.

    A well assigned straight into ``session_state["well"]`` — which the tests
    do, and which is the obvious thing for a caller to do — is adopted into the
    library here, so "the library holds the active well" is true however the
    well arrived rather than only when it came through :func:`set_well`.
    """
    well = st.session_state.get("well")
    if well is not None:
        library = wells()
        if library.get(well.name) is not well:
            library[well.name] = well
            st.session_state["active_well"] = well.name
    return well


def active_well_name():
    name = st.session_state.get("active_well")
    return name if name in wells() else None


def _capture_well_state():
    """Everything about the active well that lives outside ``WellData``."""
    settings = get_settings()
    return {
        "settings": {field: copy.deepcopy(getattr(settings, field))
                     for field in WELL_SETTINGS},
        "session": {key: st.session_state[key] for key in WELL_SESSION_KEYS
                    if key in st.session_state},
    }


def _restore_well_state(record):
    settings = get_settings()
    blank = Settings()
    for field in WELL_SETTINGS:
        stored = (record or {}).get("settings", {})
        setattr(settings, field,
                copy.deepcopy(stored[field]) if field in stored
                else copy.deepcopy(getattr(blank, field)))
    session = (record or {}).get("session", {})
    for key in WELL_SESSION_KEYS:
        if key in session:
            st.session_state[key] = session[key]
        else:
            st.session_state.pop(key, None)


def set_active_well(name):
    """Switch which well the pages work on, taking its own state with it.

    The zone tops, the datum, the petrophysics choice and the fluid model are
    stored per well and swapped here.  Without that, switching wells would
    leave well B standing on well A's rig floor and filtered by zone names that
    do not exist in it.
    """
    library = wells()
    if name not in library or name == active_well_name():
        return get_well()

    current = active_well_name()
    if current is not None:
        st.session_state.setdefault("well_state", {})[current] = _capture_well_state()

    st.session_state["active_well"] = name
    st.session_state["well"] = library[name]
    _restore_well_state(st.session_state.get("well_state", {}).get(name))
    st.session_state.pop("zone_cache", None)
    return library[name]


def set_well(well, raw=None, units=None):
    """Add a well to the library — replacing one of the same name — and make
    it active.  A newly loaded well starts with its own blank state."""
    current = active_well_name()
    if current is not None and (well is None or current != well.name):
        st.session_state.setdefault("well_state", {})[current] = _capture_well_state()

    if well is None:
        st.session_state["well"] = None
        st.session_state["active_well"] = None
    else:
        wells()[well.name] = well
        st.session_state["well"] = well
        st.session_state["active_well"] = well.name
        st.session_state.setdefault("well_state", {}).pop(well.name, None)

    _restore_well_state(None)                    # a fresh well, a fresh state
    st.session_state["raw_df"] = raw
    st.session_state["raw_units"] = units or {}
    st.session_state.pop("time_well_cache", None)
    st.session_state.pop("zone_cache", None)
    # Which depth references this file supplied, recorded before anything is
    # computed into the same columns.
    st.session_state["file_depth_curves"] = set() if well is None else {
        c for c in ("TVD", "TVDSS", "TVDBML") if c in well.df.columns}
    settings = st.session_state.get("settings")
    if settings is not None:
        settings.case = well.active_case if well is not None else None


def drop_well(name):
    """Forget one well, and fall back to whichever remains.

    Whether the dropped well was the active one has to be read *before* it
    leaves the library: ``active_well_name`` only reports a name the library
    still holds, so asking afterwards always says no — and ``get_well`` would
    then adopt the dropped well straight back in.
    """
    library = wells()
    was_active = active_well_name() == name
    library.pop(name, None)
    (st.session_state.get("well_state") or {}).pop(name, None)

    if was_active:
        st.session_state["well"] = None
        st.session_state["active_well"] = None
        _restore_well_state(None)
        st.session_state.pop("raw_df", None)
        st.session_state.pop("raw_units", None)
        if library:
            set_active_well(next(iter(library)))
    st.session_state.pop("time_well_cache", None)
    st.session_state.pop("zone_cache", None)


def load_demo_well():
    df, units = read_well(DEMO_WELL)
    well = standardise(df, units=units, name="DEMO-1")
    set_well(well, raw=df, units=units)
    # The demo well's zone codes come with a published mapping; a real well's
    # usually does not, and the codes then stand in as their own names.
    from avo_qi.sample_data.make_demo_well import ZONE_NAMES

    settings = st.session_state.get("settings")
    if settings is not None:
        settings.zone_names = dict(ZONE_NAMES)
        settings.zones = []
        st.session_state.pop("zone_cache", None)
    return well


def load_uploaded_well(uploaded, mapping=None, depth_unit="m"):
    """Read a Streamlit UploadedFile into a standardised well.

    The upload is parsed straight from memory.  Nothing is written to disk, so
    a well never lands in the working directory where it could be picked up by
    a later ``git add``.
    """
    suffix = os.path.splitext(uploaded.name)[1].lower()
    buffer = _stdlib_io.BytesIO(uploaded.getvalue())
    df, units = read_well(buffer, suffix=suffix)
    header = {}
    if suffix == ".las":
        try:
            header = read_las_header(_stdlib_io.BytesIO(uploaded.getvalue()))
        except Exception:      # a header is a convenience, never a blocker
            header = {}
    well = standardise(df, mapping=mapping, units=units, depth_unit=depth_unit,
                       name=os.path.splitext(uploaded.name)[0])
    set_well(well, raw=df, units=units)
    st.session_state["las_header"] = header
    return well


def read_uploaded_table(uploaded):
    """A supporting table — a deviation survey, a zonation — as a DataFrame.

    Routed by extension through the same reader the well itself uses, so LAS,
    CSV, TXT and Excel all arrive the same way.  A LAS is the natural format
    for both of these: a survey is a depth-indexed set of curves, and a
    zonation is a discrete curve on a depth axis.
    """
    suffix = os.path.splitext(uploaded.name)[1].lower()
    frame, _units = read_well(_stdlib_io.BytesIO(uploaded.getvalue()),
                              suffix=suffix)
    return frame


def time_well(well, settings, case=None):
    """Resample one fluid case onto a regular two-way-time grid.

    The time axis is always integrated from the **active** case's Vp, never
    from the requested case's.  Every case then lands on the same grid, so
    sample *i* is the same interface in all of them — which is what makes a
    cross-case comparison meaningful.  Re-timing the well per fluid scenario
    would silently misalign the reflectors.
    """
    case = case or well.active_case
    key = ("time_well", id(well), settings.dt, settings.t0, well.active_case, case)
    cache = st.session_state.setdefault("time_well_cache", {})
    if key in cache:
        return cache[key]

    reference = well.frame(well.active_case)
    keep = reference[["VP", "VS", "RHOB"]].notna().all(axis=1)
    reference = reference[keep].reset_index(drop=True)
    if reference.empty:
        raise ValueError("the well has no samples with Vp, Vs and RHOB all present")

    twt = depth_to_twt(reference["DEPTH"].to_numpy(float),
                       reference["VP"].to_numpy(float), t0=settings.t0)

    target = well.frame(case)[keep.to_numpy()].reset_index(drop=True)
    out = resample_to_time(target, twt, dt=settings.dt)

    if len(cache) > 24:                      # bounded: a handful of cases only
        cache.clear()
    cache[key] = out
    return out


def build_wavelet(settings):
    """Return ``(t, w)`` for the wavelet described by the sidebar settings."""
    if settings.wavelet_kind == "Ormsby":
        f1, f2, f3, f4 = settings.ormsby
        t, w = bandpass_ormsby(f1, f2, f3, f4, settings.dt, settings.wavelet_length)
    elif settings.wavelet_kind == "Uploaded" and "wavelet_upload" in st.session_state:
        t, w, meta = load_wavelet(st.session_state["wavelet_upload"], dt=settings.dt)
        settings.wavelet_meta = meta
        return t, w
    else:
        t, w = ricker(settings.ricker_freq, settings.dt, settings.wavelet_length)
    settings.wavelet_meta = {}
    return t, w


# ------------------------------------------------------------- sidebar -----
def sidebar(show_wavelet=True, show_angles=True, show_classifier=True):
    """Render the shared sidebar and return the updated :class:`Settings`."""
    s = get_settings()
    well = get_well()

    with st.sidebar:
        st.header("Well")
        library = wells()
        if len(library) > 1:
            # The library is the point of loading more than one: switching here
            # takes each well's own tops, datum and interpretation with it.
            names = list(library)
            chosen = st.selectbox(
                "Active well", names,
                index=names.index(active_well_name()) if active_well_name() in names else 0,
                key="active_well_picker",
                help=f"{len(names)} wells loaded. Every page works on the "
                     "active one; the Multi-well page compares them.")
            if chosen != active_well_name():
                set_active_well(chosen)
                st.rerun()
            well = get_well()
        if well is None:
            st.info("No well loaded.")
        else:
            st.success(f"**{well.name}** — {len(well.df)} samples")
            depth = well.depth
            st.caption(f"{depth.min():.1f} – {depth.max():.1f} m MD")
            # A computed interpretation is a model of the well, not a
            # measurement of it, and must say so wherever the well is named.
            _computed = [c for c, source in petro_sources().items()
                         if source == "computed"]
            if _computed:
                st.caption(f"{', '.join(_computed)} computed on the Load & QC "
                           "page, not read from the file.")
        if st.button("Load demo well", use_container_width=True):
            load_demo_well()
            st.rerun()

        if well is not None and well.has_fluid_cases:
            st.header("Fluid case")
            options = list(well.cases)
            current = s.case if s.case in options else well.active_case
            s.case = st.selectbox(
                "Substituted case", options, index=options.index(current),
                # A case computed here is a model of the well, not a
                # measurement of it, and must never look like one.
                format_func=lambda c: (f"{c} (computed)"
                                       if well.is_computed(c) else c),
                help="Fluid cases loaded with the well, or computed on the "
                     "Rock Physics page. The two-way-time axis always comes "
                     "from the well's in-situ case so the cases stay aligned "
                     "sample for sample.",
            )
        elif well is not None:
            s.case = well.active_case

        st.header("Time axis")
        s.dt = st.number_input("Sample rate dt (s)", 0.0002, 0.008, float(s.dt), 0.0002,
                               format="%.4f")
        s.t0 = st.number_input("TWT at first sample (s)", 0.0, 10.0, float(s.t0), 0.05)

        if show_angles:
            st.header("Angles")
            c1, c2 = st.columns(2)
            s.angle_min = c1.number_input("Min (deg)", 0.0, 60.0, float(s.angle_min), 1.0)
            s.angle_max = c2.number_input("Max (deg)", 1.0, 70.0, float(s.angle_max), 1.0)
            s.angle_step = st.number_input("Step (deg)", 0.5, 10.0, float(s.angle_step), 0.5)
            if s.angle_max <= s.angle_min:
                st.warning("Max angle must exceed min angle; using min + 10 deg.")
                s.angle_max = s.angle_min + 10.0
            st.caption(f"{s.angles.size} angles: {s.angles[0]:.0f}–{s.angles[-1]:.0f} deg")

            s.method = st.selectbox(
                "Reflectivity method", ["zoeppritz", "aki_richards"],
                index=["zoeppritz", "aki_richards"].index(s.method),
                format_func=lambda m: "Zoeppritz (exact)" if m == "zoeppritz"
                else "Aki-Richards (linear)",
            )

        if show_wavelet:
            st.header("Wavelet")
            kinds = ["Ricker", "Ormsby", "Uploaded"]
            s.wavelet_kind = st.selectbox("Type", kinds, index=kinds.index(s.wavelet_kind))
            if s.wavelet_kind == "Ricker":
                s.ricker_freq = st.slider("Peak frequency (Hz)", 5.0, 90.0,
                                          float(s.ricker_freq), 1.0)
            elif s.wavelet_kind == "Ormsby":
                f1, f2, f3, f4 = s.ormsby
                c1, c2 = st.columns(2)
                f1 = c1.number_input("f1", 1.0, 40.0, float(f1), 1.0)
                f2 = c2.number_input("f2", 2.0, 60.0, float(f2), 1.0)
                f3 = c1.number_input("f3", 10.0, 150.0, float(f3), 1.0)
                f4 = c2.number_input("f4", 12.0, 200.0, float(f4), 1.0)
                s.ormsby = (f1, f2, f3, f4)
            else:
                up = st.file_uploader("Wavelet CSV (amplitude, or time+amplitude)",
                                      type=["csv", "txt"], key="wavelet_file")
                if up is not None:
                    st.session_state["wavelet_upload"] = _uploaded_wavelet_array(up)
                if "wavelet_upload" not in st.session_state:
                    st.caption("Upload a wavelet, or the Ricker default is used.")
            s.wavelet_length = st.number_input("Wavelet length (s)", 0.032, 0.512,
                                               float(s.wavelet_length), 0.016, format="%.3f")

        if well is not None:
            table = well_zones(well, s)
            st.header("Zonation")
            if table is not None and len(table):
                available = list(dict.fromkeys(table["zone"]))
                current = [z for z in (s.zones or available) if z in available]
                s.zones = st.multiselect(
                    "Zones to analyse", available, default=current or available,
                    help="Restricts the crossplots, the reflectors and the rock "
                         "physics to the selected zones.",
                )
                source = ("the ZONE curve" if "ZONE" in well.df.columns
                          else "formation tops")
                st.caption(f"{len(table)} interval(s) from {source}.")
            else:
                # Shown rather than hidden: an absent control leaves a reader
                # guessing whether the well has no zones or the app forgot them.
                s.zones = []
                st.caption(
                    "This well carries no zonation, so nothing is filtered by "
                    "zone. Map a discrete ZONE, FORMATION or MARKER curve on "
                    "the **Load & QC** page, or enter **formation tops** "
                    "there, to enable it."
                )

        if well is not None:
            st.header("Lithology")
            if "VSH" in well.df.columns or "GR" in well.df.columns:
                if "VSH" not in well.df.columns:
                    st.caption("No VSH curve — deriving it from GR.")
                    s.gr_method = st.selectbox(
                        "GR to VSH", list(GR_METHOD_LABELS),
                        index=list(GR_METHOD_LABELS).index(s.gr_method),
                        format_func=lambda m: GR_METHOD_LABELS[m],
                    )
                cuts = dict(s.vsh_cutoffs)
                c1, c2, c3 = st.columns(3)
                cuts["sand"] = c1.number_input("Sand ≤", 0.01, 0.90,
                                               float(cuts["sand"]), 0.01,
                                               key="cut_sand")
                cuts["silty sand"] = c2.number_input("Silty ≤", 0.02, 0.95,
                                                     float(cuts["silty sand"]), 0.01,
                                                     key="cut_silty")
                cuts["silt"] = c3.number_input("Silt ≤", 0.03, 0.99,
                                               float(cuts["silt"]), 0.01,
                                               key="cut_silt")
                if cuts["sand"] < cuts["silty sand"] < cuts["silt"]:
                    s.vsh_cutoffs = cuts
                else:
                    st.warning("VSH cutoffs must increase; keeping the last valid set.")
                s.lithologies = st.multiselect(
                    "Show lithologies", LITHOLOGIES + [UNDEFINED],
                    default=[c for c in s.lithologies if c in LITHOLOGIES + [UNDEFINED]],
                    help="Filters the crossplots, the reflector table and the "
                         "A-B crossplot. A reflector is kept when either side "
                         "of it is a selected lithology.",
                )
            else:
                # Same reasoning as the zonation branch: drop any selection
                # carried in from another well, or it would filter out a well
                # whose every sample is undefined.
                s.lithologies = list(LITHOLOGIES) + [UNDEFINED]
                st.caption(
                    "This well carries neither VSH nor GR, so nothing is "
                    "filtered by lithology. Map one on the **Load & QC** page "
                    "to enable it."
                )

        if show_classifier:
            st.header("Classifier")
            s.a_tol = st.slider(
                "Intercept tolerance a_tol", 0.0, 0.10, float(s.a_tol), 0.005,
                help="Half-width of the near-zero intercept band separating "
                     "Class II from Classes I and III.",
            )
            s.threshold = st.slider(
                "Event amplitude threshold", 0.0, 0.50, float(s.threshold), 0.01,
                help="Minimum turning-point amplitude on the full stack for an "
                     "event to count as a reflector, as a fraction of the "
                     "strongest event on the trace. A fraction rather than an "
                     "absolute level, because trace amplitude scales with the "
                     "wavelet and has no fixed units.",
            )

    st.session_state["settings"] = s
    return s


def _uploaded_wavelet_array(uploaded):
    raw = uploaded.getvalue().decode("utf-8", errors="ignore")
    frame = pd.read_csv(_stdlib_io.StringIO(raw), sep=None, engine="python", header=None)
    frame = frame.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if frame.iloc[0].isna().any():          # a header row survived the coercion
        frame = frame.iloc[1:]
    return frame.dropna().to_numpy(dtype=float)


def require_well():
    """Stop the page with a friendly prompt when no well is loaded."""
    well = get_well()
    if well is None:
        st.warning("Load a well first — use **Load demo well** in the sidebar, "
                   "or upload one on the **Data & Crossplots** page.")
        st.stop()
    missing = [c for c in ("VP", "VS", "RHOB") if c not in well.df.columns]
    if missing:
        st.error(f"The loaded well is missing {', '.join(missing)}. "
                 "Remap the mnemonics on the **Data & Crossplots** page.")
        st.stop()
    return well


# --------------------------------------------------------------- plots -----
def log_track_figure(df, depth_col="DEPTH", curves=None, height=760):
    """Classic side-by-side log tracks with depth increasing downwards."""
    from plotly.subplots import make_subplots

    curves = curves or ["VP", "VS", "RHOB", "AI", "VPVS"]
    curves = [c for c in curves if c in df.columns]
    fig = make_subplots(rows=1, cols=len(curves), shared_yaxes=True,
                        horizontal_spacing=0.02, subplot_titles=curves)
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    for i, curve in enumerate(curves, start=1):
        fig.add_trace(
            go.Scatter(x=df[curve], y=df[depth_col], mode="lines",
                       line=dict(width=1.2, color=palette[(i - 1) % len(palette)]),
                       name=curve, showlegend=False),
            row=1, col=i,
        )
    fig.update_yaxes(autorange="reversed", title_text=depth_col, row=1, col=1)
    fig.update_layout(height=height, margin=dict(l=50, r=20, t=40, b=40), hovermode="y unified")
    return fig


def crossplot(df, x, y, colour=None, size=4, log_x=False, title=None, height=520):
    """Interactive crossplot, optionally coloured by a third curve."""
    fig = go.Figure()
    marker = dict(size=size, opacity=0.75)
    if colour is not None and colour in df.columns:
        marker.update(
            color=df[colour], colorscale="Viridis", showscale=True,
            colorbar=dict(title=colour),
        )
        if colour == "DEPTH":
            marker["colorscale"] = "Viridis_r"
    fig.add_trace(go.Scatter(
        x=df[x], y=df[y], mode="markers", marker=marker,
        text=[f"{colour}: {v:.3g}" for v in df[colour]] if colour in df.columns else None,
        hovertemplate=f"{x}: %{{x:.4g}}<br>{y}: %{{y:.4g}}<extra></extra>",
        name="",
    ))
    fig.update_layout(
        title=title or f"{y} vs {x}", xaxis_title=x, yaxis_title=y,
        height=height, margin=dict(l=60, r=20, t=50, b=50),
    )
    if log_x:
        fig.update_xaxes(type="log")
    return fig


def gather_figure(gather, angles, twt, mode="Variable density", height=720,
                  gain=1.0, markers=None):
    """Display an angle gather as a variable-density image or a wiggle panel."""
    gather = np.asarray(gather, dtype=float)
    fig = go.Figure()
    limit = float(np.nanmax(np.abs(gather))) or 1.0

    if mode == "Variable density":
        fig.add_trace(go.Heatmap(
            z=gather, x=angles, y=twt, colorscale="RdBu", zmid=0.0,
            zmin=-limit / gain, zmax=limit / gain,
            colorbar=dict(title="Amplitude"),
            hovertemplate="angle %{x:.0f} deg<br>TWT %{y:.3f} s<br>amp %{z:.4f}<extra></extra>",
        ))
    else:
        spacing = float(np.median(np.diff(angles))) if angles.size > 1 else 1.0
        scale = gain * spacing / (limit if limit else 1.0)
        for j, ang in enumerate(angles):
            trace = ang + gather[:, j] * scale
            # Variable area: the positive lobe is shaded, seismic convention.
            fig.add_trace(go.Scatter(
                x=np.full(twt.size, float(ang)), y=twt, mode="lines",
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=np.maximum(trace, ang), y=twt, mode="lines", line=dict(width=0),
                fill="tonextx", fillcolor="rgba(20,20,20,0.85)",
                showlegend=False, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=trace, y=twt, mode="lines", line=dict(color="#222", width=0.9),
                name=f"{ang:.0f} deg", showlegend=False,
                hovertemplate=f"angle {ang:.0f} deg<br>TWT %{{y:.3f}} s<extra></extra>",
            ))
        fig.update_xaxes(title_text="Incidence angle (deg)")

    if markers is not None and len(markers):
        # The class legend would otherwise sit on top of the amplitude colour bar.
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        for label, sub in markers.groupby("avo_class"):
            fig.add_trace(go.Scatter(
                x=np.full(len(sub), float(angles[0])), y=sub["twt"], mode="markers",
                marker=dict(color=CLASS_COLOURS.get(label, "#000"), size=9,
                            symbol="triangle-right", line=dict(width=1, color="#fff")),
                name=f"Class {label}",
            ))

    fig.update_yaxes(autorange="reversed", title_text="TWT (s)")
    fig.update_xaxes(title_text="Incidence angle (deg)")
    fig.update_layout(height=height, margin=dict(l=60, r=20, t=40, b=50))
    return fig


def ab_crossplot(table, trend=None, a_tol=0.02, height=620):
    """Intercept-gradient crossplot with shaded class regions."""
    fig = go.Figure()

    a_vals = table["A_shuey"].to_numpy(float)
    b_vals = table["B_shuey"].to_numpy(float)
    a_lim = max(float(np.nanmax(np.abs(a_vals))) * 1.25, 0.25) if a_vals.size else 0.25
    b_lim = max(float(np.nanmax(np.abs(b_vals))) * 1.25, 0.5) if b_vals.size else 0.5

    # Class regions, drawn as background rectangles.
    regions = [
        ("I", a_tol, a_lim, -b_lim, 0.0),
        ("IIp", 0.0, a_tol, -b_lim, 0.0),
        ("IIn", -a_tol, 0.0, -b_lim, 0.0),
        ("III", -a_lim, -a_tol, -b_lim, 0.0),
        ("IV", -a_lim, 0.0, 0.0, b_lim),
    ]
    for label, x0, x1, y0, y1 in regions:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, layer="below",
                      line=dict(width=0), fillcolor=CLASS_COLOURS[label], opacity=0.10)
        fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=label, showarrow=False,
                           font=dict(size=13, color=CLASS_COLOURS[label]), opacity=0.85)

    fig.add_hline(y=0, line=dict(color="#666", width=1))
    fig.add_vline(x=0, line=dict(color="#666", width=1))
    for edge in (-a_tol, a_tol):
        fig.add_vline(x=edge, line=dict(color="#999", width=1, dash="dot"))

    for label, sub in table.groupby("avo_class"):
        hover = "A %{x:.4f}<br>B %{y:.4f}<br>%{text}<extra></extra>"
        depth_col = "depth" if "depth" in sub.columns else "sample"
        fig.add_trace(go.Scatter(
            x=sub["A_shuey"], y=sub["B_shuey"], mode="markers",
            marker=dict(size=9, color=CLASS_COLOURS.get(label, "#000"),
                        line=dict(width=0.7, color="#fff")),
            name=f"Class {label}",
            text=[f"{depth_col} {v:.4g}" for v in sub[depth_col]],
            hovertemplate=hover,
        ))

    if trend is not None and np.isfinite(trend.slope):
        xs = np.linspace(-a_lim, a_lim, 50)
        fig.add_trace(go.Scatter(
            x=xs, y=trend.predict(xs), mode="lines",
            line=dict(color="#333", width=2, dash="dash"),
            name=f"Background (B = {trend.slope:.2f}A {trend.intercept:+.3f})",
        ))

    fig.update_layout(
        xaxis_title="Intercept A", yaxis_title="Gradient B",
        xaxis_range=[-a_lim, a_lim], yaxis_range=[-b_lim, b_lim],
        height=height, margin=dict(l=60, r=20, t=40, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def add_derived_curves(df, chi_deg=None):
    """Append the attributes the crossplots need to a well DataFrame."""
    from avo_qi.core.attributes import eei, lambda_rho, mu_rho, poisson, shear_impedance

    out = df.copy()
    vp, vs, rho = out["VP"].to_numpy(float), out["VS"].to_numpy(float), out["RHOB"].to_numpy(float)
    out["AI"] = acoustic_impedance(vp, rho)
    out["SI"] = shear_impedance(vs, rho)
    out["VPVS"] = vpvs(vp, vs)
    out["POISSON"] = poisson(vp, vs)
    out["LAMBDA_RHO"] = lambda_rho(vp, vs, rho)
    out["MU_RHO"] = mu_rho(vs, rho)
    if chi_deg is not None:
        out["EEI"] = eei(vp, vs, rho, chi_deg)
    return out


def case_colour(case):
    """Stable colour for a fluid case, falling back for unrecognised names."""
    return CASE_COLOURS.get(case, "#7f7f7f")


def fluid_vector_crossplot(comparison, reference, targets, a_tol=0.02, height=640):
    """A-B crossplot of every fluid case, with the fluid vectors drawn.

    Each reflector appears once per case; an arrow runs from the reference
    case to each target case, so the length and direction of the fluid effect
    are readable directly off the intercept-gradient plane.
    """
    cases = [reference] + [c for c in targets if c != reference]
    a_vals = np.concatenate([comparison[f"A_{c}"].to_numpy(float) for c in cases])
    b_vals = np.concatenate([comparison[f"B_{c}"].to_numpy(float) for c in cases])
    a_lim = max(float(np.nanmax(np.abs(a_vals))) * 1.3, 0.25)
    b_lim = max(float(np.nanmax(np.abs(b_vals))) * 1.3, 0.5)

    fig = go.Figure()
    regions = [
        ("I", a_tol, a_lim, -b_lim, 0.0),
        ("IIp", 0.0, a_tol, -b_lim, 0.0),
        ("IIn", -a_tol, 0.0, -b_lim, 0.0),
        ("III", -a_lim, -a_tol, -b_lim, 0.0),
        ("IV", -a_lim, 0.0, 0.0, b_lim),
    ]
    for label, x0, x1, y0, y1 in regions:
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, layer="below",
                      line=dict(width=0), fillcolor=CLASS_COLOURS[label], opacity=0.08)
        fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=label, showarrow=False,
                           font=dict(size=12, color=CLASS_COLOURS[label]), opacity=0.7)
    fig.add_hline(y=0, line=dict(color="#666", width=1))
    fig.add_vline(x=0, line=dict(color="#666", width=1))

    label_col = "depth" if "depth" in comparison.columns else "sample"
    for case in cases:
        fig.add_trace(go.Scatter(
            x=comparison[f"A_{case}"], y=comparison[f"B_{case}"], mode="markers",
            name=case,
            marker=dict(size=11 if case == reference else 9, color=case_colour(case),
                        line=dict(width=1, color="#fff"),
                        symbol="circle" if case == reference else "diamond"),
            text=[f"{label_col} {v:.4g} — class {c}"
                  for v, c in zip(comparison[label_col], comparison[f"class_{case}"])],
            hovertemplate="A %{x:.4f}<br>B %{y:.4f}<br>%{text}<extra></extra>",
        ))

    # One arrow per reflector per target case: the fluid vector itself.
    for case in cases[1:]:
        for _, row in comparison.iterrows():
            if not np.all(np.isfinite([row[f"A_{reference}"], row[f"B_{reference}"],
                                       row[f"A_{case}"], row[f"B_{case}"]])):
                continue
            fig.add_annotation(
                x=row[f"A_{case}"], y=row[f"B_{case}"],
                ax=row[f"A_{reference}"], ay=row[f"B_{reference}"],
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1.1, arrowwidth=1.4,
                arrowcolor=case_colour(case), opacity=0.75,
            )

    fig.update_layout(
        xaxis_title="Intercept A", yaxis_title="Gradient B",
        xaxis_range=[-a_lim, a_lim], yaxis_range=[-b_lim, b_lim],
        height=height, margin=dict(l=60, r=20, t=40, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


#: Variable-area fill either side of the zero line, in the usual seismic
#: convention: troughs (left of zero) red, peaks (right of it) blue.  Muted
#: enough that the class-coloured markers still read on top of them.
TRACE_FILL_NEGATIVE = "rgba(198,40,40,0.55)"
TRACE_FILL_POSITIVE = "rgba(21,101,192,0.55)"

#: Fills for the two halves of a reflector's lobe.  Deliberately not the class
#: palette: these say *which samples were averaged*, not what the answer was.
LOBE_COLOURS = {"upper": "rgba(31,119,180,0.16)", "lower": "rgba(214,124,26,0.16)"}
#: The same two windows as an outline, for the gather.  A fill there would sit
#: under the heatmap and vanish, and drawn over it would tint the amplitudes —
#: on a red-blue scale that is the polarity, so tinting would say something
#: about the data that is not true.
LOBE_EDGES = {"upper": "rgba(31,119,180,0.85)", "lower": "rgba(214,124,26,0.85)"}


def classified_trace_figure(trace, twt, table, extrema, selected=None, height=900,
                            gain=1.0, logs=None, trace_title="Trace", lobe=None,
                            gather=None, gather_angles=None):
    """Log tracks beside a trace, with each reflector marked and coloured by class.

    The trace is drawn variable-area, and every classified reflector gets a
    marker sitting on the amplitude extremum it produces, so the class reads
    off the wiggle itself rather than off a legend at the edge.  Markers carry
    their table row index as ``customdata`` so a click can be resolved back to
    a reflector.

    ``logs`` is an ordered ``{title: values}`` mapping — typically Vp, Vs and
    RHOB — drawn as tracks to the *left* of the trace on a shared two-way-time
    axis.  Reading the wiggle against the logs that produced it is the whole
    point of putting them side by side, and a shared axis is what makes the
    comparison trustworthy: the tracks cannot drift out of register.

    ``gather`` adds an angle gather to the *right* of the trace on that same
    axis, so the reflector picked out on the stack can be read against how it
    behaves with angle — which is the quantity being classified.

    ``lobe`` is a :func:`avo_qi.core.blocking.lobe_windows` result aligned with
    ``table``.  The selected reflector's two half-lobes are shaded across every
    panel: the upper half, whose log samples were averaged into the layer
    above, and the lower half, which gave the layer below.  Shading it over the
    logs as well as the trace is the point — it shows exactly which samples the
    intercept and gradient were computed from, instead of leaving the window an
    invisible assumption.

    Note the layout deliberately does **not** set ``dragmode``.  With
    ``dragmode="select"`` Plotly treats a click as the start of a box-selection
    drag, and a zero-area box selects nothing — which silently breaks
    click-to-inspect while leaving the chart looking perfectly normal.
    """
    from plotly.subplots import make_subplots

    trace = np.asarray(trace, dtype=float)
    twt = np.asarray(twt, dtype=float)
    limit = float(np.nanmax(np.abs(trace))) or 1.0

    logs = dict(logs or {})
    show_gather = gather is not None and gather_angles is not None
    titles = list(logs) + [trace_title]
    if show_gather:
        titles.append("Gather")
    n_cols = len(titles)
    trace_col = len(logs) + 1
    gather_col = trace_col + 1 if show_gather else None
    # The trace needs the room; the logs only have to be readable, and the
    # gather is read for its trend with angle rather than for detail.
    widths = [1.0] * len(logs) + [1.9 if logs else 1.0]
    if show_gather:
        widths.append(1.5)
    total = sum(widths)

    fig = make_subplots(
        rows=1, cols=n_cols, shared_yaxes=True, subplot_titles=titles,
        column_widths=[w / total for w in widths], horizontal_spacing=0.025,
    )

    for col, (title, values) in enumerate(logs.items(), start=1):
        values = np.asarray(values, dtype=float)
        fig.add_trace(go.Scatter(
            x=values, y=twt, mode="lines", name=title, showlegend=False,
            line=dict(color="#444", width=1.1),
            hovertemplate=f"{title} %{{x:.4g}}<br>TWT %{{y:.3f}} s<extra></extra>",
        ), row=1, col=col)

    if show_gather:
        panel = np.asarray(gather, dtype=float)
        gather_angles = np.asarray(gather_angles, dtype=float)
        span = float(np.nanmax(np.abs(panel))) or 1.0
        fig.add_trace(go.Heatmap(
            z=panel, x=gather_angles, y=twt, colorscale="RdBu", zmid=0.0,
            zmin=-span / gain, zmax=span / gain, showscale=False,
            hovertemplate="angle %{x:.0f} deg<br>TWT %{y:.3f} s"
                          "<br>amp %{z:.4f}<extra></extra>",
        ), row=1, col=gather_col)

    # Variable area, filled both ways: red left of the zero line, blue right.
    # Each half needs its own baseline trace because ``fill="tonextx"`` fills
    # to the *previous* trace, so one shared zero line cannot serve both.
    for side, colour in ((np.maximum, TRACE_FILL_POSITIVE),
                         (np.minimum, TRACE_FILL_NEGATIVE)):
        fig.add_trace(go.Scatter(
            x=np.zeros_like(twt), y=twt, mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=trace_col)
        fig.add_trace(go.Scatter(
            x=side(trace, 0.0), y=twt, mode="lines", line=dict(width=0),
            fill="tonextx", fillcolor=colour,
            showlegend=False, hoverinfo="skip",
        ), row=1, col=trace_col)
    fig.add_trace(go.Scatter(
        x=trace, y=twt, mode="lines", line=dict(color="#222", width=1.2),
        name="trace", showlegend=False,
        hovertemplate="TWT %{y:.3f} s<br>amp %{x:.4f}<extra></extra>",
    ), row=1, col=trace_col)
    fig.add_vline(x=0, line=dict(color="#999", width=1), row=1, col=trace_col)

    index = np.asarray(extrema["index"], dtype=int)
    amplitude = np.asarray(extrema["amplitude"], dtype=float)
    marker_twt = twt[np.clip(index, 0, twt.size - 1)]
    classes = table["avo_class"].to_numpy()
    depth_col = "depth" if "depth" in table.columns else "sample"
    depths = table[depth_col].to_numpy()

    resolved = np.asarray(extrema.get("is_extremum", np.ones(index.size, bool)), dtype=bool)

    # One marker trace, coloured per point, rather than a trace per class.
    # Reflectors close together often snap to the *same* turning point, so
    # their markers land on identical coordinates; when those duplicates are
    # spread across several traces a click on the pile resolves to nothing at
    # all and inspection silently stops working.  A single trace hit-tests
    # cleanly and returns the topmost point, which is a real reflector.
    fill = [CLASS_COLOURS.get(c, "#9e9e9e") if ok else "rgba(0,0,0,0)"
            for c, ok in zip(classes, resolved)]
    edge = ["#fff" if ok else CLASS_COLOURS.get(c, "#9e9e9e")
            for c, ok in zip(classes, resolved)]
    widths = [1.5 if ok else 2.5 for ok in resolved]
    notes = ["" if ok else " (no extremum)" for ok in resolved]
    fig.add_trace(go.Scatter(
        x=amplitude, y=marker_twt, mode="markers", showlegend=False,
        marker=dict(size=13, color=fill, symbol="circle",
                    line=dict(width=widths, color=edge)),
        customdata=np.arange(index.size),
        text=[f"{depth_col} {d:.4g} — class {c}{n}"
              for d, c, n in zip(depths, classes, notes)],
        hovertemplate="%{text}<br>TWT %{y:.3f} s<br>amp %{x:.4f}"
                      "<extra>click to inspect</extra>",
    ), row=1, col=trace_col)

    # Legend entries only: no points, so they cannot intercept a click.
    for label in [c for c in CLASS_COLOURS if c in set(classes)]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            name=f"Class {label}" if label != "background/other" else "Background",
            marker=dict(size=11, color=CLASS_COLOURS[label],
                        line=dict(width=1.5, color="#fff")),
            hoverinfo="skip",
        ), row=1, col=trace_col)
    if not resolved.all():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name="No extremum",
            marker=dict(size=11, color="rgba(0,0,0,0)",
                        line=dict(width=2.5, color="#666")),
            hoverinfo="skip",
        ), row=1, col=trace_col)

    if selected is not None and lobe is not None and 0 <= int(selected) < amplitude.size:
        i = int(selected)
        resolved_lobe = np.asarray(lobe.get(
            "resolved", np.ones(amplitude.size, bool)), dtype=bool)
        if i < resolved_lobe.size and resolved_lobe[i]:
            # Half a sample either end, so the band covers the samples that were
            # averaged rather than the interval between their centres — a
            # two-sample half would otherwise show as a hairline.
            step = float(np.median(np.diff(twt))) if twt.size > 1 else 0.0
            for half, colour in LOBE_COLOURS.items():
                start = int(np.clip(lobe[f"{half}_start"][i], 0, twt.size - 1))
                stop = int(np.clip(lobe[f"{half}_stop"][i] - 1, 0, twt.size - 1))
                if stop < start:
                    continue
                y0 = float(twt[start]) - step / 2.0
                y1 = float(twt[stop]) + step / 2.0
                for col in range(1, n_cols + 1):
                    # Spans the panel's full width whatever its x range, so the
                    # same window reads across logs, trace and gather alike.
                    if show_gather and col == gather_col:
                        fig.add_hrect(y0=y0, y1=y1, fillcolor="rgba(0,0,0,0)",
                                      line=dict(color=LOBE_EDGES[half], width=1.4,
                                                dash="dot"),
                                      layer="above", row=1, col=col)
                    else:
                        fig.add_hrect(y0=y0, y1=y1, fillcolor=colour,
                                      line_width=0, layer="below",
                                      row=1, col=col)

    if selected is not None and 0 <= int(selected) < amplitude.size:
        i = int(selected)
        # Drawn as *shapes*, never as an extra trace.  Plotly keys a selection
        # by curve number, so adding or removing a trace between reruns — which
        # is exactly what a moving highlight would do — invalidates the pending
        # selection and the next click silently returns nothing.  Keeping the
        # trace list identical on every rerun is what makes click-to-inspect
        # repeatable rather than working only on the first click.
        fig.add_hline(y=float(marker_twt[i]),
                      line=dict(color="#111", width=1, dash="dot"))
        span = 0.055 * (limit * 2.7 / gain)
        fig.add_shape(
            type="circle", xref="x%d" % trace_col if trace_col > 1 else "x",
            yref="y%d" % trace_col if trace_col > 1 else "y",
            x0=float(amplitude[i]) - span, x1=float(amplitude[i]) + span,
            y0=float(marker_twt[i]) - 0.006, y1=float(marker_twt[i]) + 0.006,
            line=dict(color="#111", width=2.5), fillcolor="rgba(0,0,0,0)",
            layer="above",
        )

    fig.update_yaxes(autorange="reversed")
    fig.update_yaxes(title_text="TWT (s)", row=1, col=1)
    fig.update_xaxes(title_text="Amplitude", row=1, col=trace_col,
                     range=[-limit * 1.35 / gain, limit * 1.35 / gain])
    for col in range(1, trace_col):
        fig.update_xaxes(nticks=4, row=1, col=col)
    if show_gather:
        fig.update_xaxes(title_text="Angle (deg)", nticks=5, row=1, col=gather_col)
    fig.update_layout(
        height=height, margin=dict(l=60, r=20, t=60, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0,
                    font=dict(size=11)),
        clickmode="event+select",
    )
    fig.update_annotations(font_size=12)
    return fig


def resolve_reflector_pick(stored, samples, anchor_sample=None, fallback=0):
    """Turn whatever a reflector dropdown left in session state into a row.

    A Streamlit selectbox does not round-trip its *option*; it round-trips the
    **formatted label**.  The browser sends the label string back, and
    ``SelectboxSerde.deserialize`` looks it up in a mapping rebuilt from this
    run's options — handing back the raw string when the lookup misses.  The
    reflector labels carry depth, class, A and B, so anything that moves those
    numbers (a different wavelet, angle range, blocking average, fluid case, or
    any of the zone, lithology and interface-pair filters) rewrites them, the
    lookup misses, and the stored "index" is suddenly a sentence.  Comparing
    that against a row count raises ``TypeError``, which is a crash rather than
    a lost selection.

    So a stored value is trusted only when it is an in-range integer.  Failing
    that, ``anchor_sample`` — the sample index of whatever was selected last
    run — is looked up in ``samples``, which keeps the *same reflector* chosen
    when a filter renumbers the table underneath it rather than jumping to
    whichever reflector now occupies that row.  Failing both, ``fallback``.
    """
    samples = np.asarray(samples)
    n = samples.size
    if n == 0:
        return 0

    if isinstance(stored, (bool, str, bytes)):
        stored = None                    # a stale label, not an index
    if stored is not None:
        try:
            row = int(stored)
        except (TypeError, ValueError):
            row = None
        if row is not None and 0 <= row < n:
            return row

    if anchor_sample is not None:
        match = np.flatnonzero(samples == anchor_sample)
        if match.size:
            return int(match[0])

    return int(fallback) if 0 <= int(fallback) < n else 0


def selected_reflector_index(selection, marker_twt):
    """Resolve a Plotly selection payload back to a reflector row index.

    Prefers the marker's ``customdata``; falls back to matching the point's
    TWT against the marker positions, since selection payloads differ a
    little between Streamlit versions.
    """
    if not selection:
        return None
    points = None
    if isinstance(selection, dict):
        points = (selection.get("selection") or {}).get("points") or selection.get("points")
    else:                                    # attribute-style selection state
        inner = getattr(selection, "selection", None)
        points = getattr(inner, "points", None) if inner is not None else None
    if not points:
        return None

    point = points[0]
    custom = point.get("customdata") if isinstance(point, dict) else None
    if isinstance(custom, (list, tuple)) and custom:
        custom = custom[0]
    if custom is not None:
        try:
            return int(custom)
        except (TypeError, ValueError):
            pass

    y = point.get("y") if isinstance(point, dict) else None
    if y is None:
        return None
    marker_twt = np.asarray(marker_twt, dtype=float)
    if marker_twt.size == 0:
        return None
    return int(np.argmin(np.abs(marker_twt - float(y))))


#: Human-readable names for the gamma-ray shale-volume transforms.
GR_METHOD_LABELS = {
    "linear": "Linear (gamma-ray index)",
    "larionov_tertiary": "Larionov — Tertiary",
    "larionov_older": "Larionov — older rocks",
    "steiber": "Steiber",
    "clavier": "Clavier",
}


def wavelet_spectrum_figure(w, dt, height=280, max_hz=None):
    """Amplitude spectrum of the wavelet, with its peak and -6 dB band marked.

    The bandwidth is what sets everything downstream — the lobe each reflector
    gets blocked on, the thickness at which a bed tunes, whether two interfaces
    are separable at all — so it is worth being able to see rather than infer
    from a peak-frequency number.
    """
    from avo_qi.core.wavelet import amplitude_spectrum, bandwidth, dominant_frequency

    freqs, amplitude = amplitude_spectrum(w, dt)
    fig = go.Figure()
    if freqs.size == 0:
        fig.update_layout(height=height)
        return fig

    peak = dominant_frequency(w, dt)
    low, high = bandwidth(w, dt)

    # Show to a little past the band rather than to Nyquist, which is mostly
    # empty axis on a wavelet this short.
    if max_hz is None:
        max_hz = min(float(freqs[-1]),
                     (high if np.isfinite(high) else peak * 3.0) * 1.6 or float(freqs[-1]))

    fig.add_trace(go.Scatter(
        x=freqs, y=amplitude, mode="lines", name="amplitude",
        line=dict(color="#1565c0", width=2), fill="tozeroy",
        fillcolor="rgba(21,101,192,0.16)",
        hovertemplate="%{x:.1f} Hz<br>%{y:.3f}<extra></extra>",
    ))
    if np.isfinite(low) and np.isfinite(high):
        fig.add_vrect(x0=low, x1=high, fillcolor="#1565c0", opacity=0.07,
                      line_width=0, layer="below")
        fig.add_hline(y=10 ** (-6 / 20), line=dict(color="#999", width=1, dash="dot"),
                      annotation_text="-6 dB", annotation_position="right",
                      annotation_font_size=10)
    if peak > 0:
        fig.add_vline(x=peak, line=dict(color="#c62828", width=1.5, dash="dash"),
                      annotation_text=f"{peak:.0f} Hz",
                      annotation_position="top right", annotation_font_size=11)

    fig.update_layout(
        xaxis_title="Frequency (Hz)", yaxis_title="Normalised amplitude",
        xaxis_range=[0, max_hz], yaxis_range=[0, 1.08], height=height,
        margin=dict(l=60, r=20, t=20, b=40), showlegend=False,
    )
    return fig


def vsh_series(frame, settings):
    """Shale volume per sample — the curve if there is one, else from GR.

    Returns ``(values, source)``, where ``source`` is ``"VSH"``, ``"GR"`` or
    ``None``.  A well carrying neither gets all-NaN rather than zeros, so a
    missing curve reads as *unknown* on a plot instead of as clean sand.
    """
    if "VSH" in frame.columns and np.isfinite(frame["VSH"].to_numpy(float)).any():
        return frame["VSH"].to_numpy(float), "VSH"
    if "GR" in frame.columns and np.isfinite(frame["GR"].to_numpy(float)).any():
        return (vsh_from_gr(frame["GR"].to_numpy(float), method=settings.gr_method),
                "GR")
    return np.full(len(frame), np.nan), None


def detail_log_tracks(frame, settings):
    """Ordered ``{title: values}`` for the tracks beside the classified trace.

    VSH leads, on the far left, because it is what the lithology pair and the
    sand/shale reading are cut from — the reflector table's ``shale over sand``
    is this curve, so it belongs next to the elastic logs rather than only
    behind them.  A well carrying neither VSH nor GR simply gets no such track
    instead of an empty one.
    """
    tracks = {}
    vsh, source = vsh_series(frame, settings)
    if source is not None:
        tracks["VSH (v/v)" if source == "VSH" else "VSH from GR (v/v)"] = vsh
    for title, name in (("Vp (m/s)", "VP"), ("Vs (m/s)", "VS"),
                        ("RHOB (g/cc)", "RHOB")):
        if name in frame.columns:
            tracks[title] = frame[name].to_numpy(float)
    return tracks


def lithology_labels(frame, settings):
    """Lithology per sample for a well frame, from VSH or derived from GR.

    Returns all-``undefined`` when the well carries neither curve, so a
    missing curve never silently reads as clean sand.
    """
    vsh, source = vsh_series(frame, settings)
    if source is None:
        return np.full(len(frame), UNDEFINED, dtype=object)
    return classify_lithology(vsh, cutoffs=settings.vsh_cutoffs)


def lithology_colour(label):
    """Stable colour for a lithology label."""
    return LITHOLOGY_COLOURS.get(label, "#c9c9c9")


def apply_lithology_filter(frame, labels, settings):
    """Boolean mask of samples whose lithology is currently selected.

    A well with neither VSH nor GR is never filtered: every sample reads as
    undefined, so a selection made against another well would match nothing
    and blank the page.  The same reasoning as :func:`apply_zone_filter`.
    """
    labels = np.asarray(labels, dtype=object)
    selected = set(settings.lithologies or (LITHOLOGIES + [UNDEFINED]))
    if labels.size and all(label == UNDEFINED for label in labels):
        return np.ones(labels.shape, dtype=bool)
    return np.array([label in selected for label in labels], dtype=bool)


def lithology_crossplot(frame, labels, x, y, title=None, height=520, size=4):
    """Crossplot coloured by lithology rather than by a continuous curve."""
    fig = go.Figure()
    for label in LITHOLOGIES + [UNDEFINED]:
        mask = np.asarray(labels, dtype=object) == label
        if not mask.any():
            continue
        fig.add_trace(go.Scatter(
            x=frame[x].to_numpy()[mask], y=frame[y].to_numpy()[mask],
            mode="markers", name=label,
            marker=dict(size=size, opacity=0.8, color=lithology_colour(label),
                        line=dict(width=0.3, color="#555")),
            hovertemplate=f"{x}: %{{x:.4g}}<br>{y}: %{{y:.4g}}<extra>{label}</extra>",
        ))
    # A horizontal legend above the plot would sit on the title, so this one
    # goes down the right-hand side; there are only ever a few lithologies.
    fig.update_layout(
        title=title or f"{y} vs {x}", xaxis_title=x, yaxis_title=y, height=height,
        margin=dict(l=60, r=130, t=50, b=50),
        legend=dict(orientation="v", yanchor="top", y=1.0, x=1.02),
    )
    return fig


#: The depth references, in the order they are resolved and displayed.
DEPTH_REFERENCES = ("TVD", "TVDSS", "TVDBML")

#: The interpreted curves a well either arrives with or has computed for it.
PETRO_CURVES = ("VSH", "PHI", "SW")

#: Curves that carry an interpretation *parameter* rather than a measurement.
#: A well that was interpreted properly often ships them, and they are the
#: right defaults — far better than a textbook constant, because they are what
#: this well's own porosity and saturation were actually made with.
PARAMETER_MNEMONICS = {
    "rho_matrix": ["RHOMA", "RHOGA", "RHOG", "RHOMAT"],
    "rho_fluid": ["RHOFL", "RHOF", "RHOFLU"],
    "rw": ["RW", "RWA", "RWAT"],
    "m": ["M", "MEXP", "CEMENT"],
    "n": ["N", "NEXP", "NSAT"],
    "gr_clean": ["GRMIN", "GRCLEAN", "GRSAND"],
    "gr_shale": ["GRMAX", "GRSHALE", "GRSH"],
    "r_shale": ["RSH", "RSHALE", "RCLAY"],
}


def parameter_defaults(frame):
    """Interpretation parameters read off the file's own parameter curves.

    Returns ``(values, from_file)``.  A curve is reduced to its median: these
    are per-zone constants written out per sample, not logs.
    """
    values, from_file = {}, {}
    if frame is None or not len(frame):
        return values, from_file

    lookup = {}
    for column in frame.columns:
        lookup.setdefault(str(column).strip().upper().replace(" ", "").replace("_", ""),
                          column)
    for role, candidates in PARAMETER_MNEMONICS.items():
        for candidate in candidates:
            column = lookup.get(candidate)
            if column is None:
                continue
            series = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            series = series[np.isfinite(series) & (series > -999.0)]
            if series.size:
                values[role] = float(np.median(series))
                from_file[role] = str(column)
                break
    return values, from_file


def petro_sources():
    """``{curve: "file" | "computed"}`` for VSH, PHI and SW.

    Which of these came out of the well and which came out of a transform is
    not a detail: a density porosity built on the wrong matrix density is
    wrong everywhere downstream, in the same direction, and silently.
    """
    return dict(st.session_state.get("petro_source") or {})


def model_fluid_cases(well, settings, porosity, cases=("brine", "oil", "gas"),
                      k_mineral=37.0, reservoir=None):
    """Model the standard fluid cases with the settings' fluid parameters.

    Thin by design: the physics is :func:`add_standard_cases`, so the one-click
    suite here and the single substitution on the Rock Physics page cannot
    drift apart.
    """
    options = dict(settings.fluid_model or {})
    results = add_standard_cases(
        well, porosity, k_mineral=k_mineral, cases=tuple(cases),
        in_situ_hydrocarbon=options.get("in_situ_hydrocarbon"),
        sw=(well.df["SW"].to_numpy(float)
            if options.get("in_situ_hydrocarbon") and "SW" in well.df.columns
            else None),
        hydrocarbon_sw=float(options.get("hydrocarbon_sw", 0.2)),
        parameters={k: options[k] for k in ("salinity", "api", "gor", "gas_gravity")
                    if k in options},
        conditions={k: options[k] for k in ("pressure_gradient",
                                            "temperature_surface",
                                            "temperature_gradient")
                    if k in options},
        datum=float(options.get("datum", 0.0)),
        mixing=options.get("mixing", "wood"),
        reservoir=reservoir,
    )
    # The cases are new curves on the same samples, so anything already
    # resampled into time is stale.
    st.session_state.pop("time_well_cache", None)
    return results


def apply_petrophysics(well, settings):
    """Compute VSH, PHI and SW per the settings and write them into the well.

    ``settings.petrophysics["mode"]`` decides how much is recomputed: keep the
    file's interpretation, fill only what the file is missing, or recompute
    everything here.  Returns a list of notes describing what was done.
    """
    from avo_qi.core.petrophysics import (effective_porosity,
                                          porosity_from_density,
                                          porosity_from_density_neutron,
                                          sw_archie, sw_simandoux)

    options = dict(settings.petrophysics or {})
    mode = options.get("mode", "file")

    # The file's own curves, kept from the first pass. Computing into the same
    # columns would destroy them, and then "use what the file carries" could
    # never be gone back to — the choice has to stay reversible.
    originals = st.session_state.get("petro_original")
    if originals is None:
        originals = {c: well.df[c].to_numpy(float).copy()
                     for c in PETRO_CURVES if c in well.df.columns}
        st.session_state["petro_original"] = originals

    frame = well.df
    for curve, values in originals.items():
        frame[curve] = values
    original = set(originals)
    notes = []
    sources = {c: ("file" if c in original else None) for c in PETRO_CURVES}

    def wanted(curve):
        if mode == "file":
            return False
        if mode == "fill":
            return curve not in original
        return True                                   # mode == "all"

    def column(name):
        return (frame[name].to_numpy(float) if name in frame.columns
                else np.full(len(frame), np.nan))

    # --- VSH, from gamma ray
    if wanted("VSH") and "GR" in frame.columns:
        vsh = vsh_from_gr(column("GR"),
                          gr_clean=options.get("gr_clean"),
                          gr_shale=options.get("gr_shale"),
                          method=options.get("vsh_method", settings.gr_method))
        frame["VSH"] = vsh
        sources["VSH"] = "computed"
        notes.append(f"VSH computed from GR ({options.get('vsh_method', 'linear')}).")
    elif wanted("VSH"):
        notes.append("VSH not computed: this well has no gamma-ray curve.")

    # --- porosity, from density alone or with the neutron
    if wanted("PHI") and "RHOB" in frame.columns:
        rho_matrix = float(options.get("rho_matrix", 2.65))
        rho_fluid = float(options.get("rho_fluid", 1.0))
        if options.get("phi_method") == "density-neutron" and "NPHI" in frame.columns:
            found = porosity_from_density_neutron(
                column("RHOB"), column("NPHI"), rho_matrix, rho_fluid,
                method=options.get("dn_method", "rms"))
            how = f"density-neutron ({options.get('dn_method', 'rms')})"
        else:
            found = porosity_from_density(column("RHOB"), rho_matrix, rho_fluid)
            how = "density"
        phi = found["phi"]
        if options.get("shale_correct") and "VSH" in frame.columns:
            phi = effective_porosity(phi, frame["VSH"].to_numpy(float),
                                     float(options.get("phi_shale", 0.10)))
            how += ", shale-corrected"
        frame["PHI"] = phi
        sources["PHI"] = "computed"
        notes.append(
            f"PHI computed from {how} at matrix {rho_matrix:.2f}, fluid "
            f"{rho_fluid:.2f} g/cc"
            + (f"; {found['clipped']:.0%} of samples clipped to 0–1."
               if found["clipped"] > 0.01 else "."))
    elif wanted("PHI"):
        notes.append("PHI not computed: this well has no bulk-density curve.")

    # --- saturation, from resistivity
    if wanted("SW") and "RT" in frame.columns and "PHI" in frame.columns:
        rw = float(options.get("rw", 0.05))
        a = float(options.get("a", 1.0))
        m = float(options.get("m", 2.0))
        if options.get("sw_method") == "simandoux" and "VSH" in frame.columns:
            frame["SW"] = sw_simandoux(column("RT"), column("PHI"),
                                       column("VSH"), rw=rw,
                                       r_shale=float(options.get("r_shale", 2.0)),
                                       a=a, m=m)
            notes.append(f"SW computed by Simandoux at Rw {rw:.4f}, a {a:.2f}, "
                         f"m {m:.2f}, Rshale "
                         f"{float(options.get('r_shale', 2.0)):.2f}.")
        else:
            n = float(options.get("n", 2.0))
            frame["SW"] = sw_archie(column("RT"), column("PHI"), rw=rw, a=a,
                                    m=m, n=n)
            notes.append(f"SW computed by Archie at Rw {rw:.4f}, a {a:.2f}, "
                         f"m {m:.2f}, n {n:.2f}.")
        sources["SW"] = "computed"
    elif wanted("SW"):
        notes.append("SW not computed: it needs a resistivity curve and a "
                     "porosity.")

    ordered = [c for c in CANONICAL if c in frame.columns]
    well.df = frame[ordered + [c for c in frame.columns if c not in ordered]]
    st.session_state["petro_source"] = {k: v for k, v in sources.items()
                                        if v is not None}
    st.session_state.pop("time_well_cache", None)
    return notes


def file_depth_curves():
    """Which depth references came out of the loaded file.

    Once a computed TVD has been written into the well it is indistinguishable
    from one the file carried, and :func:`depth_references` gives a file curve
    priority over everything.  Without this, a well first declared vertical
    would keep TVD = MD forever, silently ignoring a survey uploaded a minute
    later.
    """
    return set(st.session_state.get("file_depth_curves") or ())


def well_depth_references(well, settings):
    """Resolve MD, TVD, TVDSS and TVDBML for this well without writing them."""
    frame = well.df
    from_file = file_depth_curves()

    def supplied(name):
        return (frame[name].to_numpy(float)
                if name in from_file and name in frame.columns else None)

    survey = None
    if settings.deviation_survey:
        survey = survey_from_table(pd.DataFrame(settings.deviation_survey))

    return depth_references(
        frame["DEPTH"].to_numpy(float),
        tvd=supplied("TVD"), tvd_ss=supplied("TVDSS"),
        tvd_bml=supplied("TVDBML"), survey=survey,
        kb_elevation=settings.kb_elevation, water_depth=settings.water_depth,
        vertical=settings.vertical_well,
    )


def apply_depth_references(well, settings):
    """Write the resolved references into the well and invalidate the caches.

    A reference that cannot be resolved is *removed* rather than written as a
    column of nulls, so "no TVDSS here" reads as an absent curve everywhere
    downstream instead of as a curve that is somehow all blank.
    """
    resolved = well_depth_references(well, settings)
    from_file = file_depth_curves()
    for name in DEPTH_REFERENCES:
        values = resolved[name]
        if np.isfinite(values).any():
            well.df[name] = values
        elif name in well.df.columns and name not in from_file:
            well.df.drop(columns=[name], inplace=True)

    ordered = [c for c in CANONICAL if c in well.df.columns]
    well.df = well.df[ordered + [c for c in well.df.columns if c not in ordered]]
    # The time frame is interpolated from these columns, so it is now stale.
    st.session_state.pop("time_well_cache", None)
    return resolved


def well_zones(well, settings):
    """Zone intervals for a well: its ZONE curve, else hand-entered tops.

    The curve wins where there is one — it is per-sample and cannot be
    mistyped.  Tops are the fallback for the many wells whose LAS carries no
    discrete zonation at all, which would otherwise leave the zone filter, the
    zone-boundary flag and every per-zone summary permanently dead.

    Returns None when the well has neither.
    """
    has_curve = "ZONE" in well.df.columns
    tops = list(settings.zone_tops or [])
    if not has_curve and not tops:
        return None

    depth = well.df["DEPTH"].to_numpy(float)
    key = ("zones", id(well), tuple(sorted(settings.zone_names.items())),
           tuple((str(t.get("zone")), t.get("top")) for t in tops))
    cache = st.session_state.setdefault("zone_cache", {})
    if key in cache:
        return cache[key]

    if has_curve:
        table = zones_from_curve(
            depth, well.df["ZONE"].to_numpy(),
            names=settings.zone_names or None,
        )
    else:
        # Close the deepest interval just *past* the last sample. Assignment
        # is `top <= depth < base`, so closing it exactly at total depth would
        # leave the deepest sample of the well unzoned.
        if depth.size:
            step = float(np.median(np.diff(np.sort(depth)))) if depth.size > 1 else 0.0
            base_depth = float(np.nanmax(depth)) + (step if step > 0 else 1e-6)
        else:
            base_depth = None
        table = zones_from_tops(tops, base_depth=base_depth)
    if len(cache) > 8:
        cache.clear()
    cache[key] = table
    return table


def zone_labels(frame, well, settings):
    """Zone name per sample of ``frame``, or all-unzoned where there is none."""
    table = well_zones(well, settings) if well is not None else None
    if table is None or not len(table) or "DEPTH" not in frame.columns:
        return np.full(len(frame), UNZONED, dtype=object)
    return assign_zones(frame["DEPTH"].to_numpy(float), table)


def apply_zone_filter(labels, settings):
    """Mask of samples in the selected zones; everything when none is chosen.

    A well with no zonation at all is never filtered.  Every sample then reads
    as unzoned, so any selection carried over from another well would match
    nothing and blank the page — which looks like a broken app rather than an
    empty filter.  A selection that legitimately excludes everything in a well
    that *does* have zones is still honoured, because that is a real answer.
    """
    labels = np.asarray(labels, dtype=object)
    if not settings.zones:
        return np.ones(labels.shape, dtype=bool)
    if labels.size and all(label == UNZONED for label in labels):
        return np.ones(labels.shape, dtype=bool)
    selected = set(settings.zones)
    return np.array([label in selected for label in labels], dtype=bool)
