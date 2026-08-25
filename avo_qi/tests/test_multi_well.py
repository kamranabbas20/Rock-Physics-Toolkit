"""The cross-well page, and the property that makes it honest.

Every other page works on the active well, so "which tops apply" is never a
question there — there is only one answer.  Comparing wells makes it one: the
demo well is zoned by a ZONE curve, the North Sea well by hand-entered tops,
and each has to be zoned by *its own*.  A page that ran the analysis with the
active well's settings would label one well's reflectors with another well's
formation names and the table would look perfectly plausible.

So most of what is tested here is provenance rather than arithmetic: that each
well arrives with its own tops, its own datum and its own fluid case, and that
a well which cannot be analysed is named rather than allowed to stop the rest.
"""

from __future__ import annotations

import os

import pytest

streamlit = pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest  # noqa: E402

from avo_qi.io.loader import read_well, standardise  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(HERE, "pages", "6_Multi_well.py")
SECOND = os.path.join(HERE, "tests", "data", "15_9_19_A.las")

#: Tops for the North Sea well, well inside its 3500–4095 m interval so both
#: sit either side of reflectors the trace actually picks.
SECOND_TOPS = [{"zone": "Heather", "top": 3600.0},
               {"zone": "Brent", "top": 3800.0},
               {"zone": "Dunlin", "top": 3980.0}]


def second_well():
    raw, units = read_well(SECOND)
    return standardise(raw, units=units, name="15/9-19-A"), raw, units


def library_page(second=True, tops=None, timeout=300):
    """Page 6 with the demo well active and, by default, a second well beside
    it carrying its own tops in the per-well record."""
    from avo_qi.sample_data.make_demo_well import ZONE_NAMES
    from avo_qi.tests.test_app_smoke import demo_well
    from avo_qi.ui import Settings

    demo, raw, units = demo_well()
    settings = Settings()
    settings.zone_names = dict(ZONE_NAMES)

    at = AppTest.from_file(PAGE, default_timeout=timeout)
    at.session_state["settings"] = settings
    at.session_state["well"] = demo
    at.session_state["active_well"] = demo.name
    at.session_state["raw_df"] = raw
    at.session_state["raw_units"] = units

    holdings = {demo.name: demo}
    records = {}
    if second:
        other, other_raw, other_units = second_well()
        holdings[other.name] = other
        # The state well B would have left behind when it was last active.
        records[other.name] = {
            "settings": {"zone_tops": list(SECOND_TOPS if tops is None else tops),
                         "kb_elevation": 25.0, "water_depth": 120.0},
            "session": {"raw_df": other_raw, "raw_units": other_units},
        }
    at.session_state["wells"] = holdings
    at.session_state["well_state"] = records
    at.run()
    return at


def compare(at):
    """Press *Compare wells* and return the page as it stands afterwards."""
    for button in at.button:
        if "Compare" in button.label:
            button.click().run()
            return at
    raise AssertionError("no Compare wells button on the page")


def frames(at):
    return [element.value for element in at.dataframe]


@pytest.fixture(scope="module")
def compared():
    return compare(library_page())


class TestItNeedsMoreThanOneWell:
    def test_one_well_is_sent_back_to_load_and_qc(self):
        at = library_page(second=False)
        assert not at.exception
        assert at.warning
        assert "Load & QC" in at.warning[0].value

    def test_two_wells_offer_the_comparison(self):
        at = library_page()
        assert not at.exception
        assert not at.warning
        assert any("Compare" in b.label for b in at.button)

    def test_nothing_is_computed_until_it_is_asked_for(self):
        """The analysis is minutes of work on real wells, so it must not run on
        every rerun of the page."""
        at = library_page()
        assert at.info
        assert "multiwell_results" not in [k for k in at.session_state.filtered_state]


class TestTheComparison:
    def test_runs_clean_on_two_wells(self, compared):
        assert not compared.exception

    def test_every_well_is_analysed(self, compared):
        stored = compared.session_state["multiwell_results"]
        assert set(stored["results"]) == {"DEMO-1", "15/9-19-A"}
        assert not stored["failed"]

    def test_the_overview_has_a_row_per_well(self, compared):
        overview = frames(compared)[0]
        assert list(overview["well"]) == ["DEMO-1", "15/9-19-A"]
        assert (overview["events"] > 0).all()

    def test_the_shared_trend_is_fitted_through_every_event(self, compared):
        """The point of the page: a background trend fitted across wells rather
        than in one hole."""
        trends = frames(compared)[1]
        assert trends["well"].iloc[-1] == "— all wells —"
        per_well = trends["events"].iloc[:-1].sum()
        # The shared fit uses every event both wells contributed.
        assert trends["events"].iloc[-1] == per_well

    def test_each_well_keeps_its_own_events(self, compared):
        stored = compared.session_state["multiwell_results"]["results"]
        demo = stored["DEMO-1"]["table"]
        other = stored["15/9-19-A"]["table"]
        assert len(demo) and len(other)
        # Different wells, different depths — the two tables are not one table
        # computed twice.
        assert demo["depth"].max() < other["depth"].min()


class TestEachWellIsZonedByItsOwnTops:
    def test_the_zone_table_names_both_wells_own_zones(self, compared):
        """The demo well is zoned by its ZONE curve, the North Sea well by the
        tops stored against it. Neither may borrow the other's."""
        zoned = [f for f in frames(compared)
                 if "well" in getattr(f, "columns", []) and "zone" in f.columns]
        assert zoned, "no per-zone event table was rendered"
        table = zoned[0]
        demo_zones = set(table.loc[table["well"] == "DEMO-1", "zone"])
        other_zones = set(table.loc[table["well"] == "15/9-19-A", "zone"])
        assert {"Heather", "Brent", "Dunlin"} & other_zones
        assert not ({"Heather", "Brent", "Dunlin"} & demo_zones)
        assert demo_zones - {"unzoned"}

    def test_a_well_without_tops_is_simply_unzoned(self):
        at = compare(library_page(tops=[]))
        table = at.session_state["multiwell_results"]["results"]["15/9-19-A"]["table"]
        assert set(table["zone"]) == {"unzoned"}


class TestTheAnalysisCarriesEachWellsOwnSettings:
    def test_settings_are_read_per_well_not_from_the_active_one(self):
        """``well_settings_for`` is what keeps the tops with their well; this
        pins the behaviour the page depends on."""
        at = library_page()
        at.session_state["settings"].zone_tops = [{"zone": "WRONG", "top": 2000.0}]
        at.run()
        at = compare(at)
        table = at.session_state["multiwell_results"]["results"]["15/9-19-A"]["table"]
        assert "WRONG" not in set(table["zone"])
        assert {"Heather", "Brent", "Dunlin"} & set(table["zone"])
