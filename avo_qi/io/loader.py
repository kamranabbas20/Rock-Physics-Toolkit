"""Well-log ingest, mnemonic mapping and unit standardisation.

Every unit conversion in the toolkit happens here (or in a Streamlit
widget).  Downstream, ``core/`` assumes the conventions of SPEC.md section 3:
velocities in m/s, densities in g/cc, depth in m, two-way time in seconds,
angles in degrees.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "CANONICAL",
    "MNEMONIC_MAP",
    "guess_mnemonics",
    "read_las",
    "read_table",
    "read_well",
    "sonic_to_velocity",
    "velocity_to_si",
    "density_to_gcc",
    "standardise",
    "depth_to_twt",
    "resample_to_time",
    "read_las_header",
    "ELEVATION_MNEMONICS",
    "WellData",
    "FLUID_CASES",
    "FLUID_CASE_ALIASES",
    "detect_fluid_cases",
    "case_column",
    "add_substituted_case",
]

#: The curves ``core/`` needs, plus the optional ones the crossplots colour by.
#: ``DEPTH`` is measured depth; the three vertical references beside it are
#: what make one well comparable with another (see ``core/depth.py``).
CANONICAL = ["DEPTH", "TVD", "TVDSS", "TVDBML", "VP", "VS", "RHOB", "GR",
             "NPHI", "RT", "VSH", "PHI", "SW", "FACIES", "ZONE"]

#: Candidate source mnemonics for each canonical curve, in priority order.
#: Sonic mnemonics are listed too: they are converted to velocity on load.
#: ``TVDSS`` and ``TVDBML`` are resolved before ``TVD``: matching is by prefix
#: as well as exactly, so a bare ``TVD`` entry would otherwise swallow every
#: subsea curve in the file.
MNEMONIC_MAP = {
    "DEPTH": ["DEPT", "DEPTH", "MD", "TVD", "TVDSS"],
    "TVDSS": ["TVDSS", "TVDMSL", "SSTVD", "SUBSEA", "TVD_SS"],
    "TVDBML": ["TVDBML", "TVDML", "TVDSF", "TVD_BML"],
    "TVD": ["TVD", "TVDKB", "TVDRT", "TVDDF"],
    "VP": ["VP", "P_VEL", "PVEL", "VEL", "VELP", "DT", "DTC", "DTCO", "AC", "SONIC"],
    "VS": ["VS", "S_VEL", "SVEL", "VELS", "DTS", "DTSM", "DTSH", "ACS"],
    "RHOB": ["RHOB", "RHO", "DEN", "DENS", "RHOZ", "DENB"],
    "GR": ["GR", "GRD", "SGR", "CGR", "GAMM"],
    # The two raw curves a first-pass interpretation needs where the file
    # carries no PHI or SW of its own.
    "NPHI": ["NPHI", "TNPH", "NPOR", "CNC", "NEUT"],
    "RT": ["RT", "RDEP", "RESD", "ILD", "LLD", "AT90", "RD", "RES"],
    "VSH": ["VSH", "VSHALE", "VCL", "VCLAY"],
    "PHI": ["PHI", "PHIE", "PHIT", "POR", "NPHI", "PHID"],
    "SW": ["SW", "SWE", "SWT", "SUW"],
    "FACIES": ["FACIES", "LITH", "LITHOLOGY", "FACIE"],
    "ZONE": ["ZONE", "ZONES", "ZONELOG", "FORMATION", "FORM", "FM", "MARKER",
             "MARKERS", "UNIT", "STRAT", "HORIZON"],
}

#: Curves measured in depth, and so converted from feet with ``depth_unit``.
_DEPTH_CURVES = {"DEPTH", "TVD", "TVDSS", "TVDBML"}

#: Mnemonics that carry slowness rather than velocity.
SONIC_MNEMONICS = {"DT", "DTC", "DTCO", "AC", "SONIC", "DTS", "DTSM", "DTSH", "ACS"}

#: Canonical fluid-case names, in the order they are offered in the UI.
FLUID_CASES = ["in situ", "brine", "oil", "gas"]

#: Suffixes seen on fluid-substituted curves, e.g. ``VP_BR``, ``VS_GAS``,
#: ``RHOB_OIL``.  A well usually arrives with these already computed upstream;
#: :func:`add_substituted_case` can also produce them here, and marks anything
#: it computes in :attr:`WellData.computed_cases` so the two never blur.
FLUID_CASE_ALIASES = {
    "in situ": ["INSITU", "IN_SITU", "IS", "INSIT", "ORIG", "ORIGINAL", "INSITU1"],
    "brine": ["BR", "BRINE", "BRI", "WET", "W", "WATER", "SW100", "SW1", "B"],
    "oil": ["OIL", "OI", "O"],
    "gas": ["GAS", "GS", "G"],
}

#: The three curves that together make up one fluid case.
_CASE_CURVES = ("VP", "VS", "RHOB")

#: Prefixes accepted for each case curve when splitting off a fluid suffix.
#: Longest first, so ``RHOB_GAS`` matches ``RHOB`` before ``RHO``.
_CASE_PREFIXES = {
    "VP": ["VP", "DTCO", "DTC", "DT", "VELP", "PVEL"],
    "VS": ["VS", "DTSM", "DTS", "VELS", "SVEL"],
    "RHOB": ["RHOB", "RHOZ", "RHO", "DENS", "DEN"],
}

_ALIAS_TO_CASE = {
    alias: case for case, aliases in FLUID_CASE_ALIASES.items() for alias in aliases
}

_FT_PER_M = 3.280839895013123

#: Unit spellings that already are the core unit, so no note is worth raising.
_NATIVE_VELOCITY = {"m/s", "ms", "m/sec", ""}
_NATIVE_DENSITY = {"g/cc", "g/cm3", "gcc", "g/c3", ""}


def _clean_unit(unit):
    return str(unit or "").strip().lower().replace(" ", "")


def _norm(name):
    return str(name).strip().upper()


def _split_fluid_suffix(name):
    """Split a mnemonic into ``(curve, case)``, or ``(None, None)``.

    ``VP_BR`` -> ``("VP", "brine")``; ``VP`` -> ``("VP", None)``.  A suffix is
    only accepted when it is a *known* fluid token, which is what keeps
    ``VSH`` from being read as a shear log for some fluid called "H".
    """
    upper = _norm(name)
    for curve, prefixes in _CASE_PREFIXES.items():
        for prefix in prefixes:
            if upper == prefix:
                return curve, None
            if not upper.startswith(prefix):
                continue
            rest = upper[len(prefix):].lstrip("_-. ")
            if not rest or rest == upper[len(prefix):]:
                # No delimiter at all: only accept a bare, known fluid token
                # (VPGAS), never an arbitrary tail (VSH).
                rest = upper[len(prefix):]
            case = _ALIAS_TO_CASE.get(rest)
            if case is not None:
                return curve, case
    return None, None


def case_column(curve, case):
    """Column name a fluid case's curve is stored under in a standardised well.

    The active case always occupies the plain ``VP``/``VS``/``RHOB`` columns;
    every case additionally gets its own suffixed column, e.g. ``VP_GAS``.
    """
    return f"{curve}_{str(case).upper().replace(' ', '')}"


def detect_fluid_cases(columns):
    """Group fluid-substituted curves by case.

    Returns
    -------
    dict
        ``{case_name: {"VP": col, "VS": col, "RHOB": col}}``, keeping only
        cases that carry all three curves.  Curves with no fluid suffix are
        collected under ``"in situ"``.
    """
    found = {}
    for col in columns:
        curve, case = _split_fluid_suffix(col)
        if curve is None:
            continue
        case = case or "in situ"
        slot = found.setdefault(case, {})
        slot.setdefault(curve, col)      # first match wins, per priority order

    complete = {c: v for c, v in found.items() if all(k in v for k in _CASE_CURVES)}
    order = {name: i for i, name in enumerate(FLUID_CASES)}
    return dict(sorted(complete.items(), key=lambda kv: order.get(kv[0], 99)))


def guess_mnemonics(columns):
    """Best-guess mapping from canonical curve name to a column in ``columns``.

    Exact matches win over prefix matches; a column is never assigned twice.
    """
    cols = list(columns)
    upper = {_norm(c): c for c in cols}
    taken = set()
    mapping = {}

    for canonical, candidates in MNEMONIC_MAP.items():
        chosen = None
        for cand in candidates:                      # exact match first
            col = upper.get(cand)
            if col is not None and col not in taken:
                chosen = col
                break
        if chosen is None:                           # then a prefix match
            for cand in candidates:
                for key, col in upper.items():
                    if key.startswith(cand) and col not in taken:
                        chosen = col
                        break
                if chosen is not None:
                    break
        if chosen is not None:
            mapping[canonical] = chosen
            taken.add(chosen)
    return mapping


def _is_path(source):
    return isinstance(source, (str, bytes, os.PathLike))


def _as_text(source):
    """A LAS source as text, whether it arrived as a path, bytes or a buffer."""
    if _is_path(source):
        with open(source, "r", errors="replace") as fh:
            return fh.read()
    data = source.read() if hasattr(source, "read") else source
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data


def read_las(source):
    """Read a LAS into a DataFrame with the depth index as a column.

    Accepts a path, raw bytes, or a file-like object — an upload can be read
    straight from memory rather than being spilled to disk first.
    """
    import lasio

    las = lasio.read(io.StringIO(_as_text(source)))
    df = las.df().reset_index()
    units = {c.mnemonic: (c.unit or "") for c in las.curves}
    return df, units


#: LAS header mnemonics for the two heights that turn TVD into a datum depth.
#: ``EKB``/``KB`` is the drilling datum above mean sea level; ``WD`` is the
#: water depth.  Wells carry these under half a dozen names and often not at
#: all, which is why the UI asks rather than relies on them.
ELEVATION_MNEMONICS = {
    "kb_elevation": ["EKB", "KB", "EDF", "DF", "APD", "EREF", "ELEV", "EPD"],
    "ground_level": ["EGL", "GL", "GLE"],
    "water_depth": ["WD", "WATERDEPTH", "WDEP", "EWD", "WATER"],
}


def read_las_header(source):
    """Elevations and water depth from a LAS ``~Well``/``~Params`` section.

    Returns ``{"kb_elevation": .., "ground_level": .., "water_depth": ..}``
    with ``None`` where the header is silent — which it very often is: these
    fields are optional in the standard and routinely left blank, as in the
    bundled 15/9-19-A, whose ``EKB`` and ``GL`` entries exist but are empty.
    """
    import lasio

    las = lasio.read(io.StringIO(_as_text(source)))
    found = {}
    for section in (las.params, las.well):
        for item in section:
            key = _norm(getattr(item, "mnemonic", ""))
            value = getattr(item, "value", None)
            if key and key not in found:
                found[key] = value

    out = {}
    for role, candidates in ELEVATION_MNEMONICS.items():
        out[role] = None
        for candidate in candidates:
            if candidate not in found:
                continue
            try:
                value = float(str(found[candidate]).strip())
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                out[role] = value
                break
    return out


def read_table(source, suffix=None):
    """Read a CSV or Excel well table.  ``suffix`` names the format when
    ``source`` is a buffer rather than a path."""
    ext = (suffix or (os.path.splitext(str(source))[1] if _is_path(source) else "")).lower()
    if ext in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(source)
    else:
        if not _is_path(source) and hasattr(source, "seek"):
            source.seek(0)
        df = pd.read_csv(source, sep=None, engine="python")
    return df, {c: "" for c in df.columns}


def read_well(source, suffix=None):
    """Read a well from LAS, CSV or Excel.  Returns ``(DataFrame, units)``.

    ``source`` may be a path or an in-memory buffer; pass ``suffix`` (such as
    ``".las"``) for a buffer, since there is no filename to infer it from.
    """
    ext = (suffix or (os.path.splitext(str(source))[1] if _is_path(source) else "")).lower()
    if ext == ".las":
        return read_las(source)
    return read_table(source, suffix=ext)


def sonic_to_velocity(dt, unit="us/ft"):
    """Convert a sonic slowness log to velocity in m/s.

    ``us/ft``: velocity in ft/s is ``1e6 / DT``, converted to m/s.
    ``us/m``:  velocity in m/s is ``1e6 / DT`` directly.
    """
    dt = np.asarray(dt, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = 1e6 / dt
    u = str(unit).strip().lower().replace("µ", "u").replace(" ", "")
    if u in ("us/ft", "usec/ft", "us/f", "usft"):
        v = v / _FT_PER_M
    elif u in ("us/m", "usec/m", "usm"):
        pass
    else:
        raise ValueError(f"unknown sonic unit {unit!r}; expected 'us/ft' or 'us/m'")
    return np.where(np.isfinite(v), v, np.nan)


def velocity_to_si(v, unit="m/s"):
    """Convert a velocity log to m/s from m/s or ft/s."""
    v = np.asarray(v, dtype=float)
    u = str(unit).strip().lower().replace(" ", "")
    if u in ("m/s", "ms", "m/sec", ""):
        return v
    if u in ("ft/s", "fts", "ft/sec", "f/s"):
        return v / _FT_PER_M
    if u in ("km/s", "kms"):
        return v * 1000.0
    raise ValueError(f"unknown velocity unit {unit!r}")


def density_to_gcc(rho, unit="g/cc"):
    """Convert a density log to g/cc from g/cc or kg/m3."""
    rho = np.asarray(rho, dtype=float)
    u = str(unit).strip().lower().replace(" ", "")
    if u in ("g/cc", "g/cm3", "gcc", "g/c3", ""):
        return rho
    if u in ("kg/m3", "kgm3", "kg/m**3"):
        return rho / 1000.0
    raise ValueError(f"unknown density unit {unit!r}")


def _auto_velocity_unit(values, declared):
    """Fall back to a magnitude sniff when the header unit is missing."""
    if declared:
        return declared
    med = float(np.nanmedian(values)) if np.any(np.isfinite(values)) else np.nan
    if not np.isfinite(med):
        return "m/s"
    if med < 20:
        return "km/s"
    if med > 9000:
        return "ft/s"
    return "m/s"


def _auto_density_unit(values, declared):
    if declared:
        return declared
    med = float(np.nanmedian(values)) if np.any(np.isfinite(values)) else np.nan
    return "kg/m3" if np.isfinite(med) and med > 100 else "g/cc"


@dataclass
class WellData:
    """A standardised well: canonical curve names, canonical units.

    A well may carry several **fluid cases** — in situ plus brine, oil and gas
    logs substituted upstream.  The active case occupies the plain
    ``VP``/``VS``/``RHOB`` columns; every case also keeps its own suffixed
    columns, so :meth:`frame` can swap between them without re-reading.
    """

    df: pd.DataFrame
    mapping: dict = field(default_factory=dict)
    units: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    name: str = "well"
    cases: list = field(default_factory=list)
    active_case: str = "in situ"
    #: Cases this session computed rather than read from the file.  Kept apart
    #: so a reader is never left guessing which curves came out of the well and
    #: which came out of a model.
    computed_cases: list = field(default_factory=list)

    @property
    def depth(self):
        return self.df["DEPTH"].to_numpy(dtype=float)

    @property
    def has_fluid_cases(self):
        """True when the well carries more than one fluid case."""
        return len(self.cases) > 1

    def _resolve(self, case):
        if case is None:
            return self.active_case
        if self.cases and case not in self.cases:
            raise ValueError(f"case {case!r} is not in this well; have {self.cases}")
        return case

    def frame(self, case=None):
        """The well table with ``VP``/``VS``/``RHOB`` set to one fluid case."""
        case = self._resolve(case)
        out = self.df.copy()
        for curve in _CASE_CURVES:
            col = case_column(curve, case)
            if col in out.columns:
                out[curve] = out[col]
        return out

    def logs(self, case=None):
        """``(vp, vs, rho)`` as float arrays in core units, for one case."""
        frame = self.frame(case)
        return (
            frame["VP"].to_numpy(dtype=float),
            frame["VS"].to_numpy(dtype=float),
            frame["RHOB"].to_numpy(dtype=float),
        )

    def complete(self, case=None):
        """Rows where Vp, Vs and RHOB are all present, for one case."""
        return self.frame(case).dropna(subset=["VP", "VS", "RHOB"])

    def is_computed(self, case):
        """True when a case was modelled here rather than read from the file."""
        return case in self.computed_cases

    def add_case(self, case, vp, vs, rho, note=None, computed=True):
        """Attach a fluid case, stored the same way a loaded one would be.

        Writing computed curves into the ordinary ``VP_GAS``-style columns
        means every page picks them up through the machinery that already
        exists — nothing downstream needs to know where they came from.  What
        *is* recorded is that they were computed, in :attr:`computed_cases`,
        so the UI can label them and nobody mistakes a model for a measurement.
        """
        if case not in FLUID_CASES:
            raise ValueError(f"case {case!r} is not one of {FLUID_CASES}")
        length = len(self.df)
        for name, values in (("VP", vp), ("VS", vs), ("RHOB", rho)):
            values = np.asarray(values, dtype=float).ravel()
            if values.size != length:
                raise ValueError(
                    f"{name} has {values.size} samples but the well has {length}")
            self.df[case_column(name, case)] = values

        if case not in self.cases:
            order = {name: i for i, name in enumerate(FLUID_CASES)}
            self.cases = sorted(self.cases + [case], key=lambda c: order.get(c, 99))
        if computed and case not in self.computed_cases:
            self.computed_cases.append(case)
        elif not computed and case in self.computed_cases:
            self.computed_cases.remove(case)
        if note:
            self.notes.append(note)
        return self


def add_substituted_case(well, target, fluid_in, fluid_out, porosity, k_mineral,
                         source_case=None):
    """Gassmann-substitute one of a well's cases and attach the result.

    Parameters
    ----------
    well : WellData
        Modified in place, and returned for chaining.
    target : str
        Which of :data:`FLUID_CASES` the substituted curves become.
    fluid_in, fluid_out : tuple
        ``(K in GPa, rho in g/cc)`` of the fluid in the rock now and the one
        replacing it.  Either element may be an array to follow a
        depth-varying profile.
    porosity : array_like
        Pore volume fraction, one value per sample of the well.
    k_mineral : array_like
        Mineral bulk modulus in GPa, scalar or per sample.
    source_case : str, optional
        Which case to substitute *from*; the active case by default.

    Returns
    -------
    dict
        The result from :func:`avo_qi.core.gassmann.substitute`, including the
        ``valid`` mask and per-sample ``reasons`` — a caller that ignores those
        is presenting a partly empty case as a complete one.
    """
    from avo_qi.core.gassmann import substitute

    source_case = well._resolve(source_case)
    vp, vs, rho = well.logs(source_case)
    result = substitute(vp, vs, rho, porosity, k_mineral, fluid_in, fluid_out)

    n_bad = int((~result["valid"]).sum())
    note = (f"{target}: Gassmann-substituted here from the {source_case} case"
            + (f"; {n_bad} of {len(vp)} samples could not be substituted"
               if n_bad else ""))
    well.add_case(target, result["VP"], result["VS"], result["RHOB"], note=note)
    return result


def standardise(df, mapping=None, units=None, depth_unit="m", name="well", case=None):
    """Rename to canonical mnemonics and convert every curve to core units.

    Parameters
    ----------
    df : DataFrame
        Raw well table.
    mapping : dict, optional
        ``{canonical: source_column}``; guessed from the columns if omitted.
    units : dict, optional
        ``{source_column: unit_string}``, e.g. from the LAS header.  Units
        are sniffed from the values where the header is silent.
    depth_unit : str
        ``'m'`` or ``'ft'``.

    Returns
    -------
    WellData
    """
    units = {_norm(k): v for k, v in (units or {}).items()}
    mapping = dict(mapping) if mapping else guess_mnemonics(df.columns)

    out = pd.DataFrame(index=df.index)
    notes = []
    out_units = {}

    def convert(canonical, source, label=None):
        """Read one source column and put it in core units, noting any change."""
        label = label or canonical
        values = pd.to_numeric(df[source], errors="coerce").to_numpy(dtype=float)
        declared = str(units.get(_norm(source), "") or "").strip()
        src_upper = _norm(source)

        if canonical in ("VP", "VS"):
            is_sonic = any(src_upper.startswith(m) for m in SONIC_MNEMONICS) or (
                "us/" in declared.lower().replace("\u00b5", "u")
            )
            if is_sonic:
                unit = declared if "/" in declared else "us/ft"
                unit = unit.replace("\u00b5", "u").replace("USEC", "us")
                values = sonic_to_velocity(values, unit=unit)
                notes.append(f"{label}: converted sonic {source} ({unit}) to m/s")
            else:
                unit = _auto_velocity_unit(values, declared)
                values = velocity_to_si(values, unit=unit)
                if _clean_unit(unit) not in _NATIVE_VELOCITY:
                    notes.append(f"{label}: converted {source} from {unit} to m/s")
            return values, "m/s"

        if canonical == "RHOB":
            unit = _auto_density_unit(values, declared)
            values = density_to_gcc(values, unit=unit)
            if _clean_unit(unit) not in _NATIVE_DENSITY:
                notes.append(f"{label}: converted {source} from {unit} to g/cc")
            return values, "g/cc"

        if canonical in _DEPTH_CURVES:
            if str(depth_unit).lower().startswith("f"):
                values = values / _FT_PER_M
                notes.append(f"{canonical}: converted from ft to m")
            return values, "m"

        return values, declared

    for canonical, source in mapping.items():
        if source is None or source not in df.columns:
            continue
        out[canonical], out_units[canonical] = convert(canonical, source)

    # Fluid-substituted cases (VP_BR, VS_OIL, RHOB_GAS, ...).  Substitution is
    # done upstream; all that happens here is recognising and standardising the
    # results so every page can switch between them.
    detected = detect_fluid_cases(df.columns)
    # A single case is already in the plain VP/VS/RHOB columns; materialising
    # a suffixed copy of it would only duplicate every curve in the QC tables.
    if len(detected) > 1:
        for case_name, curves in detected.items():
            for curve, source in curves.items():
                col = case_column(curve, case_name)
                out[col], out_units[col] = convert(curve, source,
                                                   label=f"{curve} [{case_name}]")

    if detected:
        if case is not None and case not in detected:
            raise ValueError(
                f"case {case!r} is not in this well; found {sorted(detected)}"
            )
        active = case or ("in situ" if "in situ" in detected else next(iter(detected)))
        for curve in _CASE_CURVES:
            col = case_column(curve, active)
            if col in out.columns:
                out[curve] = out[col]
                out_units[curve] = out_units.get(col, "")
        if len(detected) > 1:
            notes.append(
                f"fluid cases found: {', '.join(detected)} — active case is {active!r}"
            )
    else:
        active = "in situ"

    missing = [c for c in ("VP", "VS", "RHOB") if c not in out.columns]
    if missing:
        notes.append(f"missing required curve(s): {', '.join(missing)}")
    if "DEPTH" not in out.columns:
        out.insert(0, "DEPTH", np.arange(len(out), dtype=float))
        notes.append("no depth curve found; using sample index")

    # A single in-situ case re-converts the same source curves as the canonical
    # mapping, so the notes would say everything twice.
    notes = list(dict.fromkeys(notes))

    ordered = [c for c in CANONICAL if c in out.columns]
    out = out[ordered + [c for c in out.columns if c not in ordered]]
    return WellData(df=out.reset_index(drop=True), mapping=mapping, units=out_units,
                    notes=notes, name=name, cases=list(detected), active_case=active)


def depth_to_twt(depth, vp, t0=0.0):
    """Integrate a Vp log to two-way time (seconds), checkshot-free.

    ``t(i) = t0 + 2 * sum(dz / vp)`` using the interval velocity between
    successive samples.  ``t0`` is the two-way time at the first sample.
    """
    depth = np.asarray(depth, dtype=float)
    vp = np.asarray(vp, dtype=float)
    if depth.shape != vp.shape:
        raise ValueError("depth and vp must have the same shape")
    if depth.size == 0:
        return np.array([], dtype=float)

    dz = np.diff(depth)
    v_int = 0.5 * (vp[:-1] + vp[1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        dt = 2.0 * dz / v_int
    dt = np.where(np.isfinite(dt), dt, 0.0)
    return float(t0) + np.concatenate([[0.0], np.cumsum(dt)])


def resample_to_time(df, twt, dt=0.001, columns=None, t0=None, t1=None):
    """Resample depth-sampled logs onto a regular two-way-time grid.

    Returns a DataFrame with a ``TWT`` column plus the interpolated curves;
    this is the sampling that ``core/synthetic`` expects.
    """
    twt = np.asarray(twt, dtype=float)
    if twt.size < 2:
        raise ValueError("need at least two samples to resample")

    columns = list(columns) if columns else [
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
    ]

    order = np.argsort(twt)
    twt_sorted = twt[order]
    t_start = float(twt_sorted[0]) if t0 is None else float(t0)
    t_end = float(twt_sorted[-1]) if t1 is None else float(t1)
    n = max(int(np.floor((t_end - t_start) / float(dt))) + 1, 1)
    grid = t_start + np.arange(n) * float(dt)

    out = {"TWT": grid}
    for c in columns:
        values = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)[order]
        good = np.isfinite(values) & np.isfinite(twt_sorted)
        if good.sum() < 2:
            out[c] = np.full(n, np.nan)
        else:
            out[c] = np.interp(grid, twt_sorted[good], values[good])
    return pd.DataFrame(out)
