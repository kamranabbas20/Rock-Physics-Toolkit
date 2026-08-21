"""Run the whole analysis from a terminal, with no browser involved.

The Streamlit app renders in a browser, which means the values behind every
plot are sent to that browser to be drawn.  On a machine where that matters,
this module does the same analysis end to end and writes the results to disk:
no web server, no WebSocket, no renderer holding your logs.

    python -m avo_qi.cli WELL.las --out results/

Figures are drawn with Matplotlib's Agg backend — a file writer, not a
display — and are optional; without Matplotlib installed the CLI still writes
every table.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

from avo_qi.core.attributes import acoustic_impedance, poisson, shear_impedance, vpvs
from avo_qi.core.avo import background_trend, reflector_avo
from avo_qi.core.lithology import classify_lithology, interface_lithology, lithology_fractions
from avo_qi.core.qc import curve_summary, depth_qc, despike, qc_flags, replace_nulls
from avo_qi.core.reflectivity import reflectivity_series
from avo_qi.core.synthetic import angle_stack, build_gather, full_stack
from avo_qi.core.wavelet import bandpass_ormsby, ricker
from avo_qi.io.loader import depth_to_twt, read_well, resample_to_time, standardise

__all__ = ["main", "run"]


def _log(message, quiet=False):
    if not quiet:
        print(message)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m avo_qi.cli",
        description="AVO & QI well analysis, start to finish, without a browser.",
    )
    parser.add_argument("well", help="LAS, CSV or Excel well file")
    parser.add_argument("--out", default="avo_qi_results", help="output directory")

    g = parser.add_argument_group("well")
    g.add_argument("--depth-unit", default="m", choices=["m", "ft"])
    g.add_argument("--case", default=None,
                   help="fluid case to analyse (in situ, brine, oil, gas)")
    g.add_argument("--top", type=float, default=None, help="analyse from this depth")
    g.add_argument("--base", type=float, default=None, help="analyse to this depth")

    g = parser.add_argument_group("QC")
    g.add_argument("--keep-nulls", action="store_true",
                   help="do not convert null sentinels to blanks")
    g.add_argument("--despike", action="store_true", help="repair spikes in Vp, Vs, RHOB")
    g.add_argument("--spike-threshold", type=float, default=5.0)
    g.add_argument("--drop-flagged", action="store_true",
                   help="drop samples that failed a QC check")

    g = parser.add_argument_group("seismic")
    g.add_argument("--dt", type=float, default=0.001, help="sample rate, seconds")
    g.add_argument("--t0", type=float, default=1.6, help="TWT at the first sample")
    g.add_argument("--angles", nargs=3, type=float, metavar=("MIN", "MAX", "STEP"),
                   default=[0.0, 40.0, 2.0])
    g.add_argument("--method", default="zoeppritz", choices=["zoeppritz", "aki_richards"])
    g.add_argument("--wavelet", default="ricker", choices=["ricker", "ormsby"])
    g.add_argument("--freq", type=float, default=30.0, help="Ricker peak frequency")
    g.add_argument("--ormsby", nargs=4, type=float, metavar=("F1", "F2", "F3", "F4"),
                   default=[5.0, 10.0, 60.0, 80.0])

    g = parser.add_argument_group("AVO")
    g.add_argument("--threshold", type=float, default=0.01, help="minimum |R| for a reflector")
    g.add_argument("--a-tol", type=float, default=0.02, help="Class II intercept band")
    g.add_argument("--vsh-cutoffs", nargs=3, type=float, metavar=("SAND", "SILTY", "SILT"),
                   default=[0.15, 0.35, 0.60])

    parser.add_argument("--no-figures", action="store_true", help="tables only")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _figures(path, well_frame, tw, gather, angles, table, trend, twt, quiet):
    """Write a multi-page PDF with Matplotlib's Agg backend — no display."""
    try:
        import matplotlib
        matplotlib.use("Agg")                      # a file writer, not a window
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        _log("  matplotlib not installed — skipping figures (tables are unaffected)", quiet)
        return None

    from avo_qi.ui_colours import CLASS_COLOURS

    with PdfPages(path) as pdf:
        # 1. Log tracks
        curves = [c for c in ("VP", "VS", "RHOB", "AI", "VPVS") if c in well_frame.columns]
        fig, axes = plt.subplots(1, len(curves), figsize=(11, 8), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, curve in zip(axes, curves):
            ax.plot(well_frame[curve], well_frame["DEPTH"], lw=0.7)
            ax.set_title(curve, fontsize=9)
            ax.grid(alpha=0.3)
        axes[0].invert_yaxis()
        axes[0].set_ylabel("DEPTH (m)")
        fig.suptitle("Log tracks")
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

        # 2. QI crossplots
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        axes[0].scatter(well_frame["AI"], well_frame["VPVS"], s=4, alpha=0.6)
        axes[0].set_xlabel("AI"); axes[0].set_ylabel("Vp/Vs")
        axes[0].set_title("Vp/Vs vs acoustic impedance"); axes[0].grid(alpha=0.3)
        axes[1].scatter(well_frame["AI"], well_frame["POISSON"], s=4, alpha=0.6, color="#d62728")
        axes[1].set_xlabel("AI"); axes[1].set_ylabel("Poisson")
        axes[1].set_title("Poisson vs acoustic impedance"); axes[1].grid(alpha=0.3)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # 3. Angle gather
        fig, ax = plt.subplots(figsize=(8, 9))
        limit = float(np.nanmax(np.abs(gather))) or 1.0
        ax.imshow(gather, aspect="auto", cmap="RdBu", vmin=-limit, vmax=limit,
                  extent=[angles[0], angles[-1], twt[-1], twt[0]])
        ax.set_xlabel("Incidence angle (deg)"); ax.set_ylabel("TWT (s)")
        ax.set_title("Synthetic angle gather")
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # 4. Intercept-gradient crossplot
        fig, ax = plt.subplots(figsize=(8, 7))
        for label, sub in table.groupby("avo_class"):
            ax.scatter(sub["A_shuey"], sub["B_shuey"], s=60, label=f"Class {label}",
                       color=CLASS_COLOURS.get(label, "#777"), edgecolor="w")
        if np.isfinite(trend.slope):
            xs = np.linspace(table["A_shuey"].min(), table["A_shuey"].max(), 20)
            ax.plot(xs, trend.predict(xs), "k--", lw=1.5,
                    label=f"background B={trend.slope:.2f}A{trend.intercept:+.3f}")
        ax.axhline(0, color="#666", lw=1); ax.axvline(0, color="#666", lw=1)
        ax.set_xlabel("Intercept A"); ax.set_ylabel("Gradient B")
        ax.set_title("Intercept-gradient crossplot"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

    return path


def run(args):
    """Execute the workflow.  Returns the output directory."""
    quiet = args.quiet
    os.makedirs(args.out, exist_ok=True)

    _log(f"Reading {args.well}", quiet)
    raw, units = read_well(args.well)
    name = os.path.splitext(os.path.basename(args.well))[0]
    well = standardise(raw, units=units, depth_unit=args.depth_unit, name=name)
    _log(f"  {len(well.df):,} samples · curves {', '.join(well.df.columns)}", quiet)
    for note in well.notes:
        _log(f"  note: {note}", quiet)

    missing = [c for c in ("VP", "VS", "RHOB") if c not in well.df.columns]
    if missing:
        raise SystemExit(f"missing required curve(s): {', '.join(missing)}")
    if args.case and args.case not in well.cases:
        raise SystemExit(f"case {args.case!r} not in this well; have {well.cases}")

    frame = well.frame(args.case)

    # ---- QC
    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    if not args.keep_nulls:
        for column in numeric:
            frame[column] = replace_nulls(frame[column].to_numpy(float))
    if args.despike:
        for column in ("VP", "VS", "RHOB"):
            values, flags = despike(frame[column].to_numpy(float),
                                    threshold=args.spike_threshold)
            frame[column] = values
            _log(f"  despiked {column}: {int(flags.sum())} samples", quiet)

    summary = curve_summary(frame[numeric], units=well.units)
    flags = qc_flags(frame, spike_threshold=args.spike_threshold)
    depth_report = depth_qc(frame["DEPTH"].to_numpy(float))
    _log(f"  QC: {int(flags['any_flag'].sum()):,} of {len(frame):,} samples flagged", quiet)
    if not depth_report["monotonic_increasing"]:
        _log("  WARNING: depth is not strictly increasing", quiet)

    depth = frame["DEPTH"].to_numpy(float)
    keep = np.ones(len(frame), dtype=bool)
    if args.top is not None:
        keep &= depth >= args.top
    if args.base is not None:
        keep &= depth <= args.base
    if args.drop_flagged:
        keep &= ~flags["any_flag"].to_numpy(bool)
    frame = frame[keep].reset_index(drop=True)
    _log(f"  analysing {len(frame):,} samples", quiet)

    complete = frame.dropna(subset=["VP", "VS", "RHOB"]).reset_index(drop=True)
    if len(complete) < 2:
        raise SystemExit("fewer than two samples have Vp, Vs and RHOB — nothing to analyse")

    # ---- attributes
    vp = complete["VP"].to_numpy(float)
    vs = complete["VS"].to_numpy(float)
    rho = complete["RHOB"].to_numpy(float)
    complete["AI"] = acoustic_impedance(vp, rho)
    complete["SI"] = shear_impedance(vs, rho)
    complete["VPVS"] = vpvs(vp, vs)
    complete["POISSON"] = poisson(vp, vs)

    # ---- time conversion and gather
    twt_log = depth_to_twt(complete["DEPTH"].to_numpy(float), vp, t0=args.t0)
    tw = resample_to_time(complete, twt_log, dt=args.dt)
    twt = tw["TWT"].to_numpy(float)
    tvp, tvs, trho = (tw[c].to_numpy(float) for c in ("VP", "VS", "RHOB"))

    lo, hi, step = args.angles
    angles = np.arange(lo, hi + step / 2, step)
    if args.wavelet == "ormsby":
        _, wavelet = bandpass_ormsby(*args.ormsby, args.dt)
    else:
        _, wavelet = ricker(args.freq, args.dt)

    gather = build_gather(tvp, tvs, trho, angles, wavelet, dt=args.dt, method=args.method)
    _log(f"  gather {gather.shape[0]} samples x {gather.shape[1]} angles", quiet)

    # ---- AVO
    rc = reflectivity_series(tvp, tvs, trho, angles, method=args.method)
    table = reflector_avo(
        rc, tvp, tvs, trho, angles, method=args.method,
        depth=tw["DEPTH"].to_numpy(float) if "DEPTH" in tw.columns else None,
        twt=twt, threshold=args.threshold, a_tol=args.a_tol,
    )
    trend = background_trend(table["A_shuey"], table["B_shuey"])

    cutoffs = dict(zip(("sand", "silty sand", "silt"), args.vsh_cutoffs))
    if "VSH" in tw.columns and np.isfinite(tw["VSH"].to_numpy(float)).any():
        litho = classify_lithology(tw["VSH"].to_numpy(float), cutoffs=cutoffs)
        if len(table):
            pairs = interface_lithology(litho, table["sample"].to_numpy())
            table = table.assign(litho_upper=pairs["upper"], litho_lower=pairs["lower"],
                                 litho_pair=pairs["pair"])
    else:
        litho = None

    _log(f"  {len(table)} reflectors above |R| > {args.threshold}", quiet)
    if len(table):
        for label, count in table["avo_class"].value_counts().items():
            _log(f"    Class {label}: {count}", quiet)

    # ---- write everything
    out = args.out
    summary.to_csv(os.path.join(out, "qc_curve_summary.csv"), index=False)
    flags.assign(DEPTH=depth).to_csv(os.path.join(out, "qc_flags.csv"), index=False)
    pd.DataFrame([depth_report]).to_csv(os.path.join(out, "qc_depth.csv"), index=False)
    complete.to_csv(os.path.join(out, "well_standardised.csv"), index=False)
    tw.to_csv(os.path.join(out, "well_time.csv"), index=False)
    table.to_csv(os.path.join(out, "reflectors.csv"), index=False)

    gather_frame = pd.DataFrame(gather, columns=[f"{a:.1f}" for a in angles])
    gather_frame.insert(0, "TWT", twt)
    gather_frame.to_csv(os.path.join(out, "gather.csv"), index=False)
    np.save(os.path.join(out, "gather.npy"), gather)

    stacks = {"TWT": twt, "full": full_stack(gather)}
    for label, band in (("near", (lo, lo + 12)), ("mid", (lo + 13, lo + 26)),
                        ("far", (max(hi - 13, lo), hi))):
        try:
            stacks[label] = angle_stack(gather, band, angles)
        except ValueError:
            pass
    pd.DataFrame(stacks).to_csv(os.path.join(out, "angle_stacks.csv"), index=False)

    report = [
        f"well                : {well.name}",
        f"source              : {os.path.abspath(args.well)}",
        f"fluid case          : {args.case or well.active_case}",
        f"cases available     : {', '.join(well.cases) or 'none'}",
        f"samples analysed    : {len(complete):,}",
        f"depth range         : {complete['DEPTH'].min():.1f} - {complete['DEPTH'].max():.1f} m",
        f"TWT range           : {twt[0]:.3f} - {twt[-1]:.3f} s",
        f"QC flagged samples  : {int(flags['any_flag'].sum()):,}",
        f"reflectivity method : {args.method}",
        f"angles              : {angles[0]:.0f} - {angles[-1]:.0f} deg, step {step:g}",
        f"wavelet             : {args.wavelet}"
        + (f" {args.freq:g} Hz" if args.wavelet == "ricker" else ""),
        f"reflectors          : {len(table)}",
    ]
    if np.isfinite(trend.slope):
        report.append(f"background trend    : B = {trend.slope:.3f} A {trend.intercept:+.4f}")
    if litho is not None:
        fractions = lithology_fractions(litho)
        report.append("lithology           : " + ", ".join(
            f"{k} {v:.0%}" for k, v in fractions.items()))
    if len(table):
        report.append("class counts        : " + ", ".join(
            f"{k} {v}" for k, v in table["avo_class"].value_counts().items()))
    with open(os.path.join(out, "report.txt"), "w") as fh:
        fh.write("\n".join(report) + "\n")

    if not args.no_figures:
        made = _figures(os.path.join(out, "figures.pdf"), complete, tw, gather,
                        angles, table, trend, twt, quiet)
        if made:
            _log("  wrote figures.pdf", quiet)

    _log(f"\nWritten to {os.path.abspath(out)} — no browser was involved.", quiet)
    return out


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        run(args)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
