"""Shared Streamlit plumbing: session state, the common sidebar, and plots.

Everything Streamlit- or Plotly-flavoured lives here or in ``pages/`` so that
``core/`` stays importable without a UI stack.
"""

from __future__ import annotations

import io as _stdlib_io
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from avo_qi.core.attributes import acoustic_impedance, vpvs
from avo_qi.core.wavelet import bandpass_ormsby, load_wavelet, ricker
from avo_qi.io.loader import depth_to_twt, read_well, resample_to_time, standardise

DEMO_WELL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data", "demo_well.las")

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
    threshold: float = 0.01
    case: str = None
    near: tuple = (0.0, 12.0)
    mid: tuple = (13.0, 26.0)
    far: tuple = (27.0, 40.0)
    wavelet_meta: dict = field(default_factory=dict)

    @property
    def angles(self):
        step = max(float(self.angle_step), 0.1)
        return np.arange(float(self.angle_min), float(self.angle_max) + step / 2, step)


# --------------------------------------------------------------- state -----
def page_setup(title, icon="~"):
    st.set_page_config(page_title=f"{title} | AVO & QI Toolkit", page_icon=icon, layout="wide")
    st.title(title)


def get_settings():
    if "settings" not in st.session_state:
        st.session_state["settings"] = Settings()
    return st.session_state["settings"]


def get_well():
    return st.session_state.get("well")


def set_well(well, raw=None, units=None):
    st.session_state["well"] = well
    st.session_state["raw_df"] = raw
    st.session_state["raw_units"] = units or {}
    st.session_state.pop("time_well_cache", None)
    settings = st.session_state.get("settings")
    if settings is not None:
        settings.case = well.active_case if well is not None else None


def load_demo_well():
    df, units = read_well(DEMO_WELL)
    well = standardise(df, units=units, name="DEMO-1")
    set_well(well, raw=df, units=units)
    return well


def load_uploaded_well(uploaded, mapping=None, depth_unit="m"):
    """Read a Streamlit UploadedFile into a standardised well."""
    suffix = os.path.splitext(uploaded.name)[1].lower()
    tmp = os.path.join(st.session_state.get("_tmpdir", "."), f"_upload{suffix}")
    with open(tmp, "wb") as fh:
        fh.write(uploaded.getbuffer())
    try:
        df, units = read_well(tmp)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    well = standardise(df, mapping=mapping, units=units, depth_unit=depth_unit,
                       name=os.path.splitext(uploaded.name)[0])
    set_well(well, raw=df, units=units)
    return well


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
        if well is None:
            st.info("No well loaded.")
        else:
            st.success(f"**{well.name}** — {len(well.df)} samples")
            depth = well.depth
            st.caption(f"{depth.min():.1f} – {depth.max():.1f} m MD")
        if st.button("Load demo well", use_container_width=True):
            load_demo_well()
            st.rerun()

        if well is not None and well.has_fluid_cases:
            st.header("Fluid case")
            options = list(well.cases)
            current = s.case if s.case in options else well.active_case
            s.case = st.selectbox(
                "Substituted case", options, index=options.index(current),
                help="Logs substituted upstream. The two-way-time axis always "
                     "comes from the well's in-situ case so the cases stay "
                     "aligned sample for sample.",
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

        if show_classifier:
            st.header("Classifier")
            s.a_tol = st.slider(
                "Intercept tolerance a_tol", 0.0, 0.10, float(s.a_tol), 0.005,
                help="Half-width of the near-zero intercept band separating "
                     "Class II from Classes I and III.",
            )
            s.threshold = st.slider(
                "Reflector |R| threshold", 0.0, 0.20, float(s.threshold), 0.005,
                help="Minimum peak |R| across angles for an interface to count "
                     "as a reflector.",
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


def classified_trace_figure(trace, twt, table, extrema, selected=None, height=720,
                            gain=1.0):
    """A single trace with each reflector's extremum marked and coloured by class.

    The trace is drawn variable-area, and every classified reflector gets a
    marker sitting on the amplitude extremum it produces, so the class reads
    off the wiggle itself rather than off a legend at the edge.  Markers carry
    their table row index as ``customdata`` so a click can be resolved back to
    a reflector.
    """
    trace = np.asarray(trace, dtype=float)
    twt = np.asarray(twt, dtype=float)
    limit = float(np.nanmax(np.abs(trace))) or 1.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.zeros_like(twt), y=twt, mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=np.maximum(trace, 0.0), y=twt, mode="lines", line=dict(width=0),
        fill="tonextx", fillcolor="rgba(20,20,20,0.75)",
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=trace, y=twt, mode="lines", line=dict(color="#222", width=1.2),
        name="trace", showlegend=False,
        hovertemplate="TWT %{y:.3f} s<br>amp %{x:.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line=dict(color="#999", width=1))

    index = np.asarray(extrema["index"], dtype=int)
    amplitude = np.asarray(extrema["amplitude"], dtype=float)
    marker_twt = twt[np.clip(index, 0, twt.size - 1)]
    classes = table["avo_class"].to_numpy()
    depth_col = "depth" if "depth" in table.columns else "sample"
    depths = table[depth_col].to_numpy()

    resolved = np.asarray(extrema.get("is_extremum", np.ones(index.size, bool)), dtype=bool)
    for label in [c for c in CLASS_COLOURS if c in set(classes)]:
        for on_extremum in (True, False):
            mask = (classes == label) & (resolved == on_extremum)
            if not mask.any():
                continue
            note = "" if on_extremum else " (no extremum)"
            fig.add_trace(go.Scatter(
                x=amplitude[mask], y=marker_twt[mask], mode="markers",
                name=(f"Class {label}" if label != "background/other" else "Background")
                     + note,
                marker=dict(
                    size=13,
                    color=CLASS_COLOURS[label] if on_extremum else "rgba(0,0,0,0)",
                    symbol="circle",
                    line=dict(width=1.5 if on_extremum else 2.5,
                              color="#fff" if on_extremum else CLASS_COLOURS[label]),
                ),
                customdata=np.flatnonzero(mask),
                text=[f"{depth_col} {d:.4g} — class {label}{note}"
                      for d in depths[mask]],
                hovertemplate="%{text}<br>TWT %{y:.3f} s<br>amp %{x:.4f}"
                              "<extra>click to inspect</extra>",
            ))

    if selected is not None and 0 <= int(selected) < amplitude.size:
        i = int(selected)
        fig.add_trace(go.Scatter(
            x=[amplitude[i]], y=[marker_twt[i]], mode="markers",
            marker=dict(size=24, color="rgba(0,0,0,0)", symbol="circle",
                        line=dict(width=2.5, color="#111")),
            name="selected", showlegend=False, hoverinfo="skip",
        ))

    fig.update_yaxes(autorange="reversed", title_text="TWT (s)")
    fig.update_xaxes(title_text="Amplitude",
                     range=[-limit * 1.35 / gain, limit * 1.35 / gain])
    fig.update_layout(
        height=height, margin=dict(l=60, r=20, t=30, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0,
                    font=dict(size=11)),
        clickmode="event+select", dragmode="select",
    )
    return fig


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
