"""Depth references: measured depth, true vertical depth, and the two datums.

A well log is sampled along the **hole**, not down the earth.  Measured depth
runs along the borehole from the drilling datum; true vertical depth is the
vertical part of it; and neither is what a geologist compares wells on.  Two
further references matter:

**TVDSS** — true vertical depth subsea, measured from mean sea level.  It is
what makes two wells comparable at all: the drilling datum is a rig floor whose
height above the sea is an accident of the rig, so two wells with the same
formation at 3800 m MD are not at the same depth in the earth.

**TVDBML** — below the mud line, measured from the seabed.  Compaction, and
therefore porosity and velocity, run from the sediment surface, not from the
sea surface; a trend fitted against TVDSS in 90 m of water and 300 m of water
is fitting two different things.

Both are simple subtractions once TVD exists, and the whole difficulty is
getting TVD:

* If the well is vertical, TVD is MD and there is nothing to do — but *say* it
  is an assumption rather than making it silently, because a 30° hole at
  4000 m MD is 200 m off.
* If a deviation survey exists, TVD comes from it by **minimum curvature**,
  which fits a circular arc between two stations rather than pretending the
  hole is straight between them.  It is the industry-standard method and the
  only one worth implementing.
* If the file already carries a TVD or TVDSS curve, that wins over anything
  computed here.

Sign convention, stated once and used everywhere: **all four depths increase
downwards**.  TVDSS is positive below mean sea level and negative above it;
TVDBML is positive below the seabed.  Software that plots subsea depth as a
negative axis is using the opposite sign — negate on the way out, not here.

Streamlit-free and dependency-light, like the rest of ``core/``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "minimum_curvature",
    "tvd_from_survey",
    "survey_from_table",
    "depth_references",
    "TVD_MNEMONICS",
    "TVDSS_MNEMONICS",
    "TVDBML_MNEMONICS",
    "SURVEY_MNEMONICS",
]

#: Mnemonics that usually hold each depth reference.  ``TVDSS`` is checked
#: before ``TVD`` wherever a name is matched by prefix, or every subsea curve
#: would answer to ``TVD``.
TVDSS_MNEMONICS = ["TVDSS", "TVDMSL", "TVDSSL", "SUBSEA", "SSTVD", "TVD_SS"]
TVDBML_MNEMONICS = ["TVDBML", "TVDML", "TVDSF", "BML", "TVD_BML", "TVDLAT"]
TVD_MNEMONICS = ["TVD", "TVDKB", "TVDRT", "TVDDF", "TRUEVERT"]

#: Column names a deviation survey arrives with: measured depth, inclination
#: from vertical in degrees, and azimuth in degrees.
SURVEY_MNEMONICS = {
    "md": ["MD", "DEPTH", "DEPT", "MEASUREDDEPTH", "MDEPTH", "SURVEYMD"],
    "inc": ["INC", "INCL", "INCLINATION", "DEVI", "DEV", "ANGLE", "DRIFT"],
    "azi": ["AZI", "AZIM", "AZIMUTH", "HAZI", "AZ", "HAZIM"],
}


def minimum_curvature(md, inc, azi):
    """True vertical depth and horizontal position at each survey station.

    Minimum curvature fits a circular arc through each pair of stations rather
    than treating the hole as straight between them.  For two stations with
    inclinations :math:`I_1, I_2` and azimuths :math:`A_1, A_2` the dogleg
    angle is

    .. math::
        \\cos\\beta = \\cos(I_2 - I_1)
                    - \\sin I_1 \\sin I_2 \\,[1 - \\cos(A_2 - A_1)]

    and each increment is scaled by the ratio factor
    :math:`(2/\\beta)\\tan(\\beta/2)`, which tends to 1 as the arc straightens
    out — so a straight hole reduces to the balanced-tangential answer instead
    of dividing by zero.

    Parameters
    ----------
    md, inc, azi : array_like
        Station measured depth, inclination from vertical (degrees) and
        azimuth (degrees).  ``md`` must increase.

    Returns
    -------
    dict
        ``tvd``, ``north``, ``east`` and ``dogleg`` (degrees per station),
        each the same length as ``md``.
    """
    md = np.asarray(md, dtype=float)
    inc = np.radians(np.asarray(inc, dtype=float))
    azi = np.radians(np.asarray(azi, dtype=float))
    if not (md.shape == inc.shape == azi.shape):
        raise ValueError("md, inc and azi must have the same length")
    if md.size == 0:
        empty = np.array([], dtype=float)
        return {"tvd": empty, "north": empty.copy(), "east": empty.copy(),
                "dogleg": empty.copy()}
    if md.size > 1 and np.any(np.diff(md) < 0):
        raise ValueError("survey measured depth must increase")

    tvd = np.zeros(md.size)
    north = np.zeros(md.size)
    east = np.zeros(md.size)
    dogleg = np.zeros(md.size)
    # The first station carries its own MD as TVD: above the shallowest survey
    # the hole is taken as vertical, which is what a survey that starts at
    # 300 m is implicitly saying.
    tvd[0] = md[0]

    for k in range(1, md.size):
        d_md = md[k] - md[k - 1]
        i1, i2 = inc[k - 1], inc[k]
        a1, a2 = azi[k - 1], azi[k]
        cos_beta = np.cos(i2 - i1) - np.sin(i1) * np.sin(i2) * (1.0 - np.cos(a2 - a1))
        beta = np.arccos(np.clip(cos_beta, -1.0, 1.0))
        # The ratio factor is 1 in the limit of a straight segment; taking the
        # limit explicitly avoids 0/0 on the (very common) vertical section.
        rf = 1.0 if beta < 1e-9 else (2.0 / beta) * np.tan(beta / 2.0)
        half = 0.5 * d_md * rf
        tvd[k] = tvd[k - 1] + half * (np.cos(i1) + np.cos(i2))
        north[k] = north[k - 1] + half * (np.sin(i1) * np.cos(a1)
                                          + np.sin(i2) * np.cos(a2))
        east[k] = east[k - 1] + half * (np.sin(i1) * np.sin(a1)
                                        + np.sin(i2) * np.sin(a2))
        dogleg[k] = np.degrees(beta)

    return {"tvd": tvd, "north": north, "east": east, "dogleg": dogleg}


def tvd_from_survey(md, survey_md, survey_inc, survey_azi):
    """True vertical depth at each log sample, from a deviation survey.

    Survey stations are tens of metres apart and log samples are centimetres,
    so the log depths have to be placed *within* a survey interval.  Rather
    than interpolating TVD linearly between stations — which straightens the
    arc that minimum curvature just took the trouble to fit — inclination and
    azimuth are interpolated to the sample and minimum curvature is applied
    from the station above it.

    Samples above the shallowest station are taken as vertical; below the
    deepest, the last station's attitude is held.
    """
    md = np.asarray(md, dtype=float)
    survey_md = np.asarray(survey_md, dtype=float)
    survey_inc = np.asarray(survey_inc, dtype=float)
    survey_azi = np.asarray(survey_azi, dtype=float)

    good = (np.isfinite(survey_md) & np.isfinite(survey_inc)
            & np.isfinite(survey_azi))
    survey_md, survey_inc, survey_azi = (survey_md[good], survey_inc[good],
                                         survey_azi[good])
    order = np.argsort(survey_md)
    survey_md, survey_inc, survey_azi = (survey_md[order], survey_inc[order],
                                         survey_azi[order])
    if survey_md.size == 0:
        raise ValueError("the survey has no usable stations")
    if survey_md.size == 1:
        # One station says nothing about curvature; hold its inclination.
        drop = np.cos(np.radians(survey_inc[0]))
        return survey_md[0] + (md - survey_md[0]) * drop

    stations = minimum_curvature(survey_md, survey_inc, survey_azi)
    inc_at = np.radians(np.interp(md, survey_md, survey_inc))
    azi_at = np.radians(np.interp(md, survey_md, survey_azi))
    # Which station each sample hangs from: the deepest one at or above it.
    above = np.clip(np.searchsorted(survey_md, md, side="right") - 1, 0,
                    survey_md.size - 1)

    # One minimum-curvature step from that station down to the sample, done
    # for every sample at once rather than in a Python loop.
    i1 = np.radians(survey_inc)[above]
    a1 = np.radians(survey_azi)[above]
    d_md = md - survey_md[above]
    cos_beta = (np.cos(inc_at - i1)
                - np.sin(i1) * np.sin(inc_at) * (1.0 - np.cos(azi_at - a1)))
    beta = np.arccos(np.clip(cos_beta, -1.0, 1.0))
    rf = np.where(beta < 1e-9, 1.0,
                  (2.0 / np.where(beta < 1e-9, 1.0, beta)) * np.tan(beta / 2.0))
    tvd = stations["tvd"][above] + 0.5 * d_md * rf * (np.cos(i1) + np.cos(inc_at))

    # Above the shallowest station the hole is taken as vertical, so the
    # sample's TVD is that station's less the MD between them.
    shallow = md <= survey_md[0]
    tvd[shallow] = stations["tvd"][0] - (survey_md[0] - md[shallow])
    return tvd


def survey_from_table(frame):
    """``(md, inc, azi)`` from a table whose columns are named loosely.

    A survey arrives as a CSV with whatever headers the surveying company
    used, so the columns are matched on a list of the usual names rather than
    demanded exactly.
    """
    import pandas as pd

    frame = pd.DataFrame(frame).copy()
    lookup = {str(c).strip().upper().replace(" ", "").replace("_", ""): c
              for c in frame.columns}
    picked = {}
    for role, candidates in SURVEY_MNEMONICS.items():
        for candidate in candidates:
            key = candidate.replace("_", "")
            match = next((lookup[k] for k in lookup
                          if k == key or k.startswith(key)), None)
            if match is not None and match not in picked.values():
                picked[role] = match
                break
    missing = [role for role in ("md", "inc", "azi") if role not in picked]
    if missing:
        raise ValueError(
            "the survey needs measured depth, inclination and azimuth columns; "
            f"could not find {', '.join(missing)} in {list(frame.columns)}")

    values = [pd.to_numeric(frame[picked[role]], errors="coerce").to_numpy(float)
              for role in ("md", "inc", "azi")]
    keep = np.isfinite(values[0]) & np.isfinite(values[1]) & np.isfinite(values[2])
    return tuple(v[keep] for v in values)


def _given(values, shape):
    """A curve that was actually supplied, or None."""
    if values is None:
        return None
    values = np.asarray(values, dtype=float)
    if values.shape != shape or not np.isfinite(values).any():
        return None
    return values


def depth_references(md, tvd=None, tvd_ss=None, tvd_bml=None, survey=None,
                     kb_elevation=None, water_depth=None, vertical=False):
    """Resolve MD, TVD, TVDSS and TVDBML from whatever is known.

    Nothing here is guessed silently: every reference that could not be
    resolved comes back as all-NaN with a note saying what is missing, because
    a plausible-looking subsea depth built on an assumed rig-floor height is
    worse than no subsea depth at all.

    Parameters
    ----------
    md : array_like
        Measured depth per sample.
    tvd, tvd_ss, tvd_bml : array_like, optional
        Curves the file already carries.  Each wins over anything computed
        here — a file's own TVDSS was made with the survey and the datum the
        people who drilled the well had, which beats a reconstruction.
    survey : tuple or DataFrame, optional
        ``(md, inc, azi)`` or a table :func:`survey_from_table` can read.
    kb_elevation : float, optional
        Height of the drilling datum above mean sea level, in metres.  This is
        what turns TVD into TVDSS.
    water_depth : float, optional
        Sea level to seabed, in metres.  This is what turns TVDSS into TVDBML.
    vertical : bool
        Take TVD = MD.  An assumption, and recorded as one in the notes.

    Returns
    -------
    dict
        ``MD``, ``TVD``, ``TVDSS``, ``TVDBML`` arrays, a ``notes`` list, and
        ``source`` naming where TVD came from.
    """
    md = np.asarray(md, dtype=float)
    blank = np.full(md.shape, np.nan)
    notes = []
    tvd = _given(tvd, md.shape)
    tvd_ss = _given(tvd_ss, md.shape)
    tvd_bml = _given(tvd_bml, md.shape)
    kb_known = kb_elevation is not None and np.isfinite(float(kb_elevation))

    if tvd is not None:
        source = "curve"
        notes.append("TVD read from the file.")
    elif survey is not None:
        if not isinstance(survey, (tuple, list)) or len(survey) != 3:
            survey = survey_from_table(survey)
        tvd = tvd_from_survey(md, *survey)
        source = "survey"
        notes.append(f"TVD computed by minimum curvature from "
                     f"{np.size(survey[0])} survey stations.")
    elif vertical:
        tvd = md.copy()
        source = "vertical"
        notes.append("TVD assumed equal to MD — the well is taken as vertical. "
                     "A 30° hole at 4000 m MD would be ~200 m shallower than "
                     "this.")
    elif tvd_ss is not None and kb_known:
        tvd = tvd_ss + float(kb_elevation)
        source = "subsea curve"
        notes.append("TVD reconstructed from the file's TVDSS and the drilling "
                     "datum height.")
    else:
        tvd = blank.copy()
        source = None
        notes.append("No TVD: the file carries no TVD curve, no deviation "
                     "survey was given, and the well was not declared vertical.")

    if tvd_ss is not None:
        notes.append("TVDSS read from the file.")
    # `if not kb_elevation` would read a rig floor at 0 m — a land well on the
    # coast — as an unknown one, so the check is explicitly against None.
    elif kb_known:
        tvd_ss = tvd - float(kb_elevation)
    else:
        tvd_ss = blank.copy()
        notes.append("No TVDSS: the height of the drilling datum above mean "
                     "sea level is not known.")

    if tvd_bml is not None:
        notes.append("TVDBML read from the file.")
    elif water_depth is not None and np.isfinite(float(water_depth)):
        tvd_bml = tvd_ss - float(water_depth)
    else:
        tvd_bml = blank.copy()
        notes.append("No TVDBML: the water depth is not known.")

    return {"MD": md, "TVD": tvd, "TVDSS": tvd_ss, "TVDBML": tvd_bml,
            "notes": notes, "source": source}
