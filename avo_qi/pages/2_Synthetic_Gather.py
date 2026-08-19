"""Page 2 — build and export the synthetic angle gather."""

from __future__ import annotations

import os
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import io as _stdlib_io  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from avo_qi.core.synthetic import angle_stack, build_gather, full_stack  # noqa: E402
from avo_qi.ui import (  # noqa: E402
    add_derived_curves,
    build_wavelet,
    gather_figure,
    log_track_figure,
    page_setup,
    require_well,
    sidebar,
    time_well,
)

page_setup("Synthetic Gather", icon=":ocean:")
settings = sidebar(show_classifier=False)
well = require_well()

# ------------------------------------------------------------- wavelet -----
st.subheader("Wavelet")
wt, wavelet = build_wavelet(settings)
meta = settings.wavelet_meta or {}

c1, c2 = st.columns([2, 1])
with c1:
    fig = go.Figure(go.Scatter(x=wt, y=wavelet, mode="lines", line=dict(width=2)))
    fig.add_hline(y=0, line=dict(color="#999", width=1))
    fig.add_vline(x=0, line=dict(color="#999", width=1, dash="dot"))
    fig.update_layout(xaxis_title="Time (s)", yaxis_title="Amplitude", height=280,
                      margin=dict(l=60, r=20, t=20, b=40))
    st.plotly_chart(fig, use_container_width=True)
with c2:
    st.metric("Samples", wavelet.size)
    st.metric("Length", f"{(wavelet.size - 1) * settings.dt * 1000:.0f} ms")
    if meta:
        st.metric("Zero-phase", "yes" if meta.get("zero_phase") else "no")
        if not meta.get("zero_phase"):
            st.warning(
                f"Uploaded wavelet is not zero-phase (asymmetry "
                f"{meta.get('symmetry_error', float('nan')):.3f}); the synthetic will "
                "carry that phase."
            )
        if meta.get("resampled"):
            st.caption(f"Resampled to dt = {settings.dt * 1000:.1f} ms.")
    else:
        st.metric("Zero-phase", "yes")

# -------------------------------------------------------------- gather -----
st.divider()
st.subheader("Angle gather")

try:
    tw = time_well(well, settings)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

vp = tw["VP"].to_numpy(float)
vs = tw["VS"].to_numpy(float)
rho = tw["RHOB"].to_numpy(float)
twt = tw["TWT"].to_numpy(float)
angles = settings.angles

if vp.size < 2:
    st.error(
        f"Only {vp.size} time samples at dt = {settings.dt * 1000:.1f} ms. "
        "Lower dt, or check the depth range of the well."
    )
    st.stop()

gather = build_gather(vp, vs, rho, angles, wavelet, dt=settings.dt, method=settings.method)
st.session_state["gather"] = gather
st.session_state["gather_twt"] = twt

c1, c2, c3 = st.columns([1, 1, 2])
mode = c1.radio("Display", ["Variable density", "Wiggle"], horizontal=True)
gain = c2.slider("Gain", 0.2, 5.0, 1.0, 0.1)
c3.caption(
    f"{gather.shape[0]} samples × {gather.shape[1]} angles · "
    f"{settings.method.replace('_', '-')} · dt = {settings.dt * 1000:.1f} ms · "
    f"TWT {twt[0]:.3f}–{twt[-1]:.3f} s"
)

left, right = st.columns([1, 2])
with left:
    tw_derived = add_derived_curves(tw)
    st.plotly_chart(
        log_track_figure(tw_derived, depth_col="TWT", curves=["VP", "RHOB", "VPVS"],
                         height=720),
        use_container_width=True,
    )
with right:
    st.plotly_chart(gather_figure(gather, angles, twt, mode=mode, gain=gain),
                    use_container_width=True)

# -------------------------------------------------------------- stacks -----
st.divider()
st.subheader("Angle stacks")

c1, c2, c3 = st.columns(3)
near = c1.slider("Near (deg)", float(angles[0]), float(angles[-1]),
                 (float(angles[0]), min(float(angles[0]) + 12, float(angles[-1]))))
mid = c2.slider("Mid (deg)", float(angles[0]), float(angles[-1]),
                (min(float(angles[0]) + 13, float(angles[-1])),
                 min(float(angles[0]) + 26, float(angles[-1]))))
far = c3.slider("Far (deg)", float(angles[0]), float(angles[-1]),
                (min(float(angles[0]) + 27, float(angles[-1])), float(angles[-1])))
settings.near, settings.mid, settings.far = near, mid, far

stacks = {}
for label, band in (("Near", near), ("Mid", mid), ("Far", far)):
    try:
        stacks[label] = angle_stack(gather, band, angles)
    except ValueError:
        st.warning(f"No angles fall in the {label.lower()} band {band}; skipped.")
stacks["Full"] = full_stack(gather)

fig = go.Figure()
colours = {"Near": "#1f77b4", "Mid": "#2ca02c", "Far": "#d62728", "Full": "#333333"}
for label, trace in stacks.items():
    fig.add_trace(go.Scatter(x=trace, y=twt, mode="lines", name=label,
                             line=dict(width=1.6, color=colours[label])))
fig.add_vline(x=0, line=dict(color="#999", width=1))
fig.update_yaxes(autorange="reversed", title_text="TWT (s)")
fig.update_layout(xaxis_title="Amplitude", height=620,
                  margin=dict(l=60, r=20, t=30, b=40), hovermode="y unified")
st.plotly_chart(fig, use_container_width=True)

if "Near" in stacks and "Far" in stacks:
    diff = stacks["Far"] - stacks["Near"]
    idx = int(np.argmax(np.abs(diff)))
    st.caption(
        f"Largest far-minus-near difference is {diff[idx]:+.4f} at TWT {twt[idx]:.3f} s — "
        "a negative value there means the event brightens with offset."
    )

# -------------------------------------------------------------- export -----
st.divider()
st.subheader("Export")

frame = pd.DataFrame(gather, columns=[f"{a:.1f}" for a in angles])
frame.insert(0, "TWT", twt)

c1, c2, c3 = st.columns(3)
c1.download_button("CSV", frame.to_csv(index=False).encode(),
                   file_name=f"{well.name}_gather.csv", mime="text/csv",
                   use_container_width=True)

npy_buffer = _stdlib_io.BytesIO()
np.save(npy_buffer, gather)
c2.download_button("NPY", npy_buffer.getvalue(), file_name=f"{well.name}_gather.npy",
                   mime="application/octet-stream", use_container_width=True)


def _segy_bytes(traces, sample_interval_s):
    """Write the gather to a minimal SEG-Y, one trace per angle."""
    import segyio

    with tempfile.NamedTemporaryFile(suffix=".sgy", delete=False) as tmp:
        path = tmp.name
    try:
        spec = segyio.spec()
        spec.format = 5                      # IEEE float
        spec.samples = np.arange(traces.shape[0]) * sample_interval_s * 1000.0
        spec.tracecount = traces.shape[1]
        with segyio.create(path, spec) as dst:
            for j in range(traces.shape[1]):
                dst.header[j] = {
                    segyio.su.tracl: j + 1,
                    segyio.su.offset: int(angles[j] * 100),
                    segyio.su.ns: traces.shape[0],
                    segyio.su.dt: int(sample_interval_s * 1e6),
                }
                dst.trace[j] = traces[:, j].astype(np.float32)
        with open(path, "rb") as fh:
            return fh.read()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


try:
    c3.download_button("SEG-Y", _segy_bytes(gather, settings.dt),
                       file_name=f"{well.name}_gather.sgy",
                       mime="application/octet-stream", use_container_width=True)
except ImportError:
    c3.caption("SEG-Y export needs `segyio` — `pip install segyio`.")
except Exception as exc:
    c3.caption(f"SEG-Y export unavailable: {exc}")
