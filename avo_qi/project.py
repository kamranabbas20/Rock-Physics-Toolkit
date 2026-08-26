"""The setup file: everything you decided about a well, and none of its data.

Close the browser today and the library goes with it — the tops, the datum, the
petrophysics choice, the shear-sonic model, the fluid parameters, the cutoffs.
On a real well that is twenty minutes of re-entry every session, and there is
no way to hand a setup to a colleague or keep one beside the project.

A setup file fixes that, and the important decision is what it is *not*. It
carries **your interpretation, not your logs**: names, depths, parameters and
choices — a few kilobytes of JSON you could read in a text editor, mail to
someone, or commit next to the project. The LAS stays the system of record and
the setup is what you decided about it, which is also why applying one needs
the well loaded first.

Two things keep it honest.

:data:`SAVED` and :data:`EXCLUDED` between them name **every** field of
``Settings``, and a test fails if a new field belongs to neither. A setup file
that silently stopped carrying something would be worse than none, because it
would restore *most* of a session and look complete.

Every setup records a :func:`fingerprint` of the well it was built from —
name, sample count, depth range and curve names. Applying a setup to a
different well is then something the app can notice and ask about, rather than
quietly standing well B on well A's rig floor.
"""

from __future__ import annotations

import datetime as _datetime
import json

__all__ = [
    "SCHEMA",
    "default_folder",
    "SUFFIX",
    "SAVED",
    "EXCLUDED",
    "PER_WELL",
    "fingerprint",
    "compare_fingerprint",
    "build_setup",
    "dumps",
    "loads",
    "apply_setup",
    "describe",
    "suggested_name",
    "WIDGET_KEYS",
]

#: Bumped whenever the meaning of a field changes.  A file from the future is
#: refused rather than half-read.
SCHEMA = 1

#: The extension the app suggests.  Doubly suffixed so it is obvious in a
#: folder what the file is and that it is readable text.
SUFFIX = ".avoqi-setup.json"

#: Settings that describe **one well** rather than the session.  The same list
#: ``ui.WELL_SETTINGS`` swaps when you change the active well; it is asserted
#: equal to it in the tests rather than imported, so ``project`` stays free of
#: Streamlit.
PER_WELL = (
    "case", "zones", "zone_names", "zone_tops", "kb_elevation", "water_depth",
    "deviation_survey", "vertical_well", "petrophysics", "vs_prediction",
    "fluid_model", "case_mapping",
)

#: Settings shared by every well — how you are looking, rather than what you
#: are looking at.
SHARED = (
    "dt", "t0", "angle_min", "angle_max", "angle_step", "method", "a_tol",
    "wavelet_kind", "ricker_freq", "ormsby", "wavelet_length", "threshold",
    "vsh_cutoffs", "net_cutoffs", "net_pay", "lithologies", "gr_method",
    "near", "mid", "far",
)

#: Everything a setup file carries.
SAVED = PER_WELL + SHARED

#: Fields deliberately left out, with the reason.  Anything not here and not in
#: :data:`SAVED` is a field somebody forgot, and the tests say so.
EXCLUDED = {
    "wavelet_meta": "holds the samples of an uploaded wavelet — data rather "
                    "than a choice, and potentially large. Re-upload the "
                    "wavelet file instead.",
}

#: Widget keys that mirror a saved setting.  Streamlit gives a keyed widget's
#: own state priority over its ``value=`` argument, so writing a setting is not
#: enough to move the control that shows it — the stale widget simply writes
#: its old value back on the next run, and applying a setup appears to do
#: nothing.  Forgetting these keys makes each widget re-read the setting it is
#: derived from.  Listed by setting so the reason for each is visible, and a
#: test asserts every key still exists in the app.
WIDGET_KEYS = {
    "kb_elevation": ("kb_elevation_input",),
    "water_depth": ("water_depth_input",),
    "vertical_well": ("tvd_source",),
    "petrophysics": ("petro_mode", "petro_vsh_method", "petro_phi_method",
                     "petro_sw_method", "petro_dn_method", "petro_gr_clean",
                     "petro_gr_shale", "petro_rho_matrix", "petro_rho_fluid",
                     "petro_rw", "petro_m", "petro_n", "petro_a",
                     "petro_r_shale", "petro_shale_correct", "petro_phi_shale"),
    "vs_prediction": ("vs_model", "vs_degree"),
    "vsh_cutoffs": ("cut_sand", "cut_silty", "cut_silt"),
    "net_cutoffs": ("net_vsh_cut", "net_phi_cut", "net_sw_cut"),
    "net_pay": ("net_pay_toggle",),
    "fluid_model": ("fluid_brine_preset", "fluid_oil_preset",
                    "fluid_gas_preset", "fluid_in_situ", "fluid_hc_sw",
                    "fluid_mixing", "fluid_k_mineral", "fluid_datum",
                    "fluid_p_grad", "fluid_t_grad", "fluid_t_surface",
                    "fluid_reservoir_only"),
}


def stale_widget_keys(changed):
    """Widget keys to forget after applying a setup, for the settings that
    actually changed."""
    keys = []
    for name in changed or ():
        keys.extend(WIDGET_KEYS.get(name, ()))
    return keys


#: Curves whose *names* are recorded in the fingerprint. Never their values.
_FINGERPRINT_KEYS = ("well", "samples", "depth_min", "depth_max", "curves")


#: Envelope for a dict whose keys are not strings.  JSON has no integer keys,
#: and ``zone_names`` is keyed by the numeric code in the ZONE curve — so a
#: plain ``json.dumps`` silently turns ``{1: "Shale"}`` into ``{"1": "Shale"}``
#: and every zone lookup by code misses, leaving the app showing bare codes and
#: the wrong interval count.  Pairs survive the trip with their types intact.
PAIRS = "__pairs__"


def _plain(value):
    """JSON-safe copy: tuples become lists, numpy scalars become Python ones,
    and a dict with non-string keys becomes a :data:`PAIRS` envelope."""
    if isinstance(value, dict):
        if value and not all(isinstance(k, str) for k in value):
            return {PAIRS: [[_plain(k), _plain(v)] for k, v in value.items()]}
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (bool, str)):
        return value
    if hasattr(value, "item"):                    # numpy scalar
        try:
            return _plain(value.item())
        except Exception:                          # pragma: no cover
            return str(value)
    if isinstance(value, (int, float)):
        return value
    return str(value)


def _restore(value):
    """Undo :func:`_plain` where it enveloped something."""
    if isinstance(value, dict):
        if set(value) == {PAIRS}:
            return {_restore(k): _restore(v) for k, v in value[PAIRS]}
        return {k: _restore(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_restore(v) for v in value]
    return value


def fingerprint(well):
    """Enough of a well to recognise it again, and nothing anyone could mine.

    Sample count, depth range and curve *names* — no curve values.  It is
    deliberately not a hash of the data: a hash would tie a setup to one export
    of a well and break on a harmless re-export, and it would also be a
    fingerprint of proprietary data sitting in a file meant to be shareable.
    """
    frame = getattr(well, "df", None)
    if frame is None or not len(frame):
        return {"well": getattr(well, "name", None), "samples": 0,
                "depth_min": None, "depth_max": None, "curves": []}
    depth = frame["DEPTH"] if "DEPTH" in frame.columns else None
    return {
        "well": getattr(well, "name", None),
        "samples": int(len(frame)),
        "depth_min": None if depth is None else round(float(depth.min()), 3),
        "depth_max": None if depth is None else round(float(depth.max()), 3),
        "curves": sorted(str(c) for c in frame.columns),
    }


def compare_fingerprint(setup, well):
    """What differs between the well a setup was built from and this one.

    Returns ``{"matches": bool, "differences": [str, ...]}``.  A difference is
    reported, never enforced: re-running a setup against a re-exported well
    with two extra curves is a perfectly ordinary thing to want, and only the
    person doing it knows whether it is the same well.
    """
    stored = dict((setup or {}).get("well") or {})
    current = fingerprint(well)
    differences = []

    if stored.get("well") != current["well"]:
        differences.append("built for %r, this well is %r"
                           % (stored.get("well"), current["well"]))
    if stored.get("samples") != current["samples"]:
        differences.append("%s samples then, %s now"
                           % (stored.get("samples"), current["samples"]))
    for key, label in (("depth_min", "top"), ("depth_max", "bottom")):
        then, now = stored.get(key), current.get(key)
        if then is not None and now is not None and abs(then - now) > 0.5:
            differences.append("%s at %.1f m then, %.1f m now" % (label, then, now))
    missing = sorted(set(stored.get("curves") or []) - set(current["curves"]))
    if missing:
        differences.append("curves no longer present: " + ", ".join(missing))

    return {"matches": not differences, "differences": differences}


def build_setup(well, settings, note=None):
    """Capture the setup as a plain dict, ready to be written as JSON."""
    return {
        "schema": SCHEMA,
        "kind": "avo-qi-setup",
        "created": _datetime.datetime.now().replace(microsecond=0).isoformat(),
        "note": note or "",
        "well": fingerprint(well),
        "settings": {name: _plain(getattr(settings, name, None))
                     for name in SAVED},
    }


def dumps(setup):
    """The file's bytes.  Indented, because it is meant to be readable."""
    return json.dumps(setup, indent=2, sort_keys=False).encode("utf-8")


def loads(payload):
    """Parse a setup file, refusing anything that is not one.

    Raises
    ------
    ValueError
        If the text is not JSON, is not a setup file, or was written by a
        newer schema than this build understands.  An older schema is
        accepted: fields it lacks simply keep their current values.
    """
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8", errors="replace")
    try:
        setup = json.loads(payload)
    except Exception as exc:
        raise ValueError("this is not a setup file: %s" % exc) from None
    if not isinstance(setup, dict) or setup.get("kind") != "avo-qi-setup":
        raise ValueError("this is not an AVO & QI setup file")

    schema = setup.get("schema")
    if not isinstance(schema, int):
        raise ValueError("the setup file does not say which schema it uses")
    if schema > SCHEMA:
        raise ValueError(
            "this setup was written by a newer version (schema %d; this build "
            "understands %d). Update the toolkit rather than let it read half "
            "the file." % (schema, SCHEMA))
    return setup


def apply_setup(setup, settings, fields=None):
    """Write a setup's values onto a ``Settings``, and say what changed.

    Only fields the file actually carries are touched, so an older setup
    leaves newer settings at their defaults instead of blanking them.

    Returns a list of the field names that changed value.
    """
    stored = dict((setup or {}).get("settings") or {})
    wanted = tuple(fields) if fields is not None else SAVED
    changed = []
    for name in wanted:
        if name not in stored:
            continue
        before = getattr(settings, name, None)
        value = _restore(stored[name])
        if name in ("ormsby", "near", "mid", "far") and isinstance(value, list):
            value = tuple(value)
        setattr(settings, name, value)
        # Compared as they really are, not as two JSON-safe copies: comparing
        # plainified values would call {1: "Shale"} and {"1": "Shale"} equal,
        # which is precisely the difference that breaks a zone lookup.
        if before != value:
            changed.append(name)
    return changed


def describe(setup):
    """A short human summary of what a setup carries, for showing before it is
    applied.  Reading a file and then being asked is the whole point."""
    stored = dict((setup or {}).get("settings") or {})
    well = dict((setup or {}).get("well") or {})
    tops = stored.get("zone_tops") or []
    survey = stored.get("deviation_survey") or []
    return {
        "well": well.get("well"),
        "created": (setup or {}).get("created"),
        "note": (setup or {}).get("note") or "",
        "fields": sum(1 for name in SAVED if name in stored),
        "tops": len(tops),
        "survey_stations": len(survey),
        "datum": stored.get("kb_elevation"),
        "water_depth": stored.get("water_depth"),
        "petrophysics": (stored.get("petrophysics") or {}).get("mode"),
        "vs_model": (stored.get("vs_prediction") or {}).get("model"),
        "case": stored.get("case"),
    }


def default_folder():
    """Where a "save a copy here" box should start.

    The Desktop if there is one, else the home directory.  This is the folder
    of the machine **running the app**, which is your own only because the
    toolkit binds to localhost — deploy it for a team and this is the server's
    Desktop, which is why the box is a deliberate action rather than the
    default way to save.
    """
    import os

    home = os.path.expanduser("~")
    desktop = os.path.join(home, "Desktop")
    return desktop if os.path.isdir(desktop) else home


def suggested_name(well):
    """A filename that says which well it belongs to."""
    name = (getattr(well, "name", None) or "well")
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(name))
    return (safe.strip("_") or "well") + SUFFIX
