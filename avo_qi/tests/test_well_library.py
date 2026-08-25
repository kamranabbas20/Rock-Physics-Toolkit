"""Several wells at once, each keeping its own state.

The toolkit held exactly one well until now, and everything that describes a
well rather than the session — the tops, the rig-floor height, the water depth,
the petrophysics choice, the fluid model — lived on the shared ``Settings``.
With more than one well open that is a trap: switching wells would leave the
second one standing on the first one's datum and filtered by zone names that do
not exist in it. So the state travels with the well.
"""

from __future__ import annotations

import os
import textwrap

import pytest

streamlit = pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SECOND = os.path.join(ROOT, "avo_qi", "tests", "data", "15_9_19_A.las")

HEADER = f'''
import sys
sys.path.insert(0, {ROOT!r})
import streamlit as st
from avo_qi.io.loader import read_well, standardise
from avo_qi.ui import (active_well_name, drop_well, get_settings, get_well,
                       load_demo_well, set_active_well, set_well, wells)


def second():
    raw, units = read_well({SECOND!r})
    return standardise(raw, units=units, name="15/9-19-A"), raw, units


def report():
    s = get_settings()
    st.session_state["report"] = {{
        "library": list(wells()),
        "active": None if get_well() is None else get_well().name,
        "kb": s.kb_elevation,
        "water": s.water_depth,
        "tops": [t["zone"] for t in (s.zone_tops or [])],
        "zones": list(s.zones or []),
        "case": s.case,
    }}


step = st.session_state.get("step", 0)
'''


def run(steps, rounds=None):
    """Run a script whose body is a numbered sequence of steps."""
    body = "\n".join(
        f"if step == {i}:\n"
        + "\n".join("    " + line for line in
                    textwrap.dedent(code).strip().split("\n"))
        + f"\n    st.session_state['step'] = {i + 1}"
        for i, code in enumerate(steps))
    at = AppTest.from_string(HEADER + body + "\nreport()\n", default_timeout=120)
    for _ in range(rounds or len(steps) + 1):
        at.run()
        assert not at.exception, at.exception
    return at


class TestTheLibrary:
    def test_a_second_well_joins_rather_than_replaces(self):
        at = run(["load_demo_well()",
                  "w, raw, units = second(); set_well(w, raw=raw, units=units)"])
        report = at.session_state["report"]
        assert report["library"] == ["DEMO-1", "15/9-19-A"]
        assert report["active"] == "15/9-19-A"

    def test_reloading_the_same_name_replaces_it(self):
        at = run(["load_demo_well()", "load_demo_well()"])
        assert at.session_state["report"]["library"] == ["DEMO-1"]

    def test_a_well_set_directly_is_adopted(self):
        """The obvious thing for a caller to do — and what the other tests do —
        is assign session_state['well']. The library has to notice."""
        at = run(["w, raw, units = second(); st.session_state['well'] = w"])
        report = at.session_state["report"]
        assert report["library"] == ["15/9-19-A"]
        assert report["active"] == "15/9-19-A"

    def test_switching_is_by_name_and_is_a_no_op_for_a_stranger(self):
        at = run(["load_demo_well()",
                  "w, raw, units = second(); set_well(w, raw=raw, units=units)",
                  "set_active_well('nobody')"])
        assert at.session_state["report"]["active"] == "15/9-19-A"


class TestStateTravelsWithTheWell:
    STEPS = [
        # A well with a datum, tops and a zone selection of its own.
        """
        load_demo_well()
        s = get_settings()
        s.kb_elevation = 25.0
        s.water_depth = 90.0
        s.zone_tops = [{"zone": "Alpha", "top": 2000.0}]
        s.zones = ["Gas Sand"]
        """,
        # A second well, which must start clean.
        """
        w, raw, units = second()
        set_well(w, raw=raw, units=units)
        get_settings().kb_elevation = 40.0
        """,
        "set_active_well('DEMO-1')",
        "set_active_well('15/9-19-A')",
    ]

    def _at(self, upto):
        return run(self.STEPS[:upto])

    def test_a_new_well_starts_on_no_datum_at_all(self):
        """The bug this prevents: well B standing on well A's rig floor."""
        report = self._at(2).session_state["report"]
        assert report["active"] == "15/9-19-A"
        assert report["water"] is None
        assert report["tops"] == []
        assert report["zones"] == []

    def test_switching_back_restores_the_first_wells_state(self):
        report = self._at(3).session_state["report"]
        assert report["active"] == "DEMO-1"
        assert report["kb"] == 25.0
        assert report["water"] == 90.0
        assert report["tops"] == ["Alpha"]
        assert report["zones"] == ["Gas Sand"]

    def test_and_switching_away_again_restores_the_second(self):
        report = self._at(4).session_state["report"]
        assert report["active"] == "15/9-19-A"
        assert report["kb"] == 40.0
        assert report["water"] is None
        assert report["tops"] == []

    def test_the_fluid_case_follows_its_own_well(self):
        """The demo well has four cases and 15/9-19-A has one; a case name from
        one is meaningless in the other."""
        at = run(self.STEPS[:2] + ["set_active_well('DEMO-1')"])
        assert at.session_state["report"]["case"] == "in situ"


class TestDroppingAWell:
    def test_removing_the_active_well_falls_back_to_another(self):
        at = run(["load_demo_well()",
                  "w, raw, units = second(); set_well(w, raw=raw, units=units)",
                  "drop_well('15/9-19-A')"])
        report = at.session_state["report"]
        assert report["library"] == ["DEMO-1"]
        assert report["active"] == "DEMO-1"

    def test_removing_the_last_well_leaves_nothing_loaded(self):
        at = run(["load_demo_well()", "drop_well('DEMO-1')"])
        report = at.session_state["report"]
        assert report["library"] == []
        assert report["active"] is None

    def test_a_dropped_wells_state_does_not_haunt_the_next_one(self):
        at = run([
            """
            load_demo_well()
            get_settings().kb_elevation = 25.0
            """,
            "drop_well('DEMO-1')",
            "w, raw, units = second(); set_well(w, raw=raw, units=units)",
        ])
        assert at.session_state["report"]["kb"] is None
