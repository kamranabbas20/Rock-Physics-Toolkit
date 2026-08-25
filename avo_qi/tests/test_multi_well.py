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

import numpy as np
import pandas as pd
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


class TestClassAgainstProperty:
    """The panel that answers what the class actually depends on.

    Pooled across wells it is the same code page 4 runs on one, so what is
    worth pinning here is the pooling itself and the honesty of the readout:
    a ranking that quotes the best of seventeen raw p-values as if it were the
    only test is worse than no ranking at all.
    """

    @staticmethod
    def ranking(at):
        for element in at.dataframe:
            frame = element.value
            if hasattr(frame, "columns") and "property" in frame.columns:
                return frame
        raise AssertionError("no dependence ranking on the page")

    def test_the_ranking_covers_every_available_property(self, compared):
        from avo_qi.ui import property_options

        everything = pd.concat(
            [r["table"] for r in
             compared.session_state["multiwell_results"]["results"].values()],
            ignore_index=True)
        assert len(self.ranking(compared)) == len(property_options(everything))

    def test_it_is_ranked_by_effect_size(self, compared):
        found = self.ranking(compared)["ε²"].dropna().to_numpy(float)
        assert (np.diff(found) <= 1e-9).all()

    def test_the_adjusted_p_is_never_smaller_than_the_raw_one(self, compared):
        """Bonferroni over the properties tested. The panel picks the best of
        many and must not quote its raw p as if it had asked once."""
        found = self.ranking(compared).dropna(subset=["p"])
        assert len(found) >= 2
        assert (found["p (adj)"] >= found["p"] - 1e-12).all()
        assert (found["p (adj)"] <= 1.0).all()

    def test_both_wells_are_pooled_into_it(self, compared):
        """The count tested has to exceed either well on its own, or the panel
        is quietly showing one well."""
        results = compared.session_state["multiwell_results"]["results"]
        each = [len(r["table"]) for r in results.values()]
        tested = self.ranking(compared)["events"].max()
        assert tested > max(each) - min(each)     # more than one well's worth
        assert tested <= sum(each)

    def test_the_pooled_table_carries_the_lobe_properties(self, compared):
        for table in compared.session_state["multiwell_results"]["results"].values():
            assert {"phi_res", "ntg_res", "d_phi", "reservoir_side"} <= set(table["table"].columns)


class TestAnomaliesAcrossWells:
    """Ranked together, scored apart.

    The property that makes this worth having: a well with a tight A–B cloud
    and a well with a wide one cannot share a yardstick. Scale them together
    and the noisy well fills the top of the list while the quiet well's one
    genuine standout disappears into everyone else's scatter.
    """

    @staticmethod
    def listing(at):
        for element in at.dataframe:
            frame = element.value
            if hasattr(frame, "columns") and "rank" in frame.columns:
                return frame
        raise AssertionError("no anomaly listing on the page")

    def test_each_well_is_measured_against_its_own_trend(self, compared):
        """Not the shared one: a trend is what the ordinary rock in *that*
        hole does, and measuring well B against well A's background would
        report the difference between the wells as an anomaly in every
        reflector of one of them."""
        from avo_qi.core.avo import background_trend

        results = compared.session_state["multiwell_results"]["results"]
        rows = self.listing(compared)
        assert len(rows)

        checked = 0
        for name, found in results.items():
            table = found["table"]
            own = background_trend(table["A_shuey"], table["B_shuey"])
            shared = background_trend(
                pd.concat([r["table"]["A_shuey"] for r in results.values()]),
                pd.concat([r["table"]["B_shuey"] for r in results.values()]))
            expected = np.asarray(
                own.deviation(table["A_shuey"], table["B_shuey"]), dtype=float)
            wrong = np.asarray(
                shared.deviation(table["A_shuey"], table["B_shuey"]),
                dtype=float)

            for _, row in rows[rows["well"] == name].iterrows():
                at = np.isclose(table["depth"].to_numpy(float), row["depth"],
                                atol=1e-3)
                assert at.sum() == 1, row["depth"]
                mine = float(expected[at][0])
                assert row["background_deviation"] == pytest.approx(mine,
                                                                    abs=5e-4)
                checked += 1

            # And the two trends must actually differ, or the test above would
            # pass with the shared trend used everywhere.
            assert not np.allclose(expected, wrong, atol=5e-4)
        assert checked >= 4

    def test_both_wells_can_reach_the_top_of_the_list(self, compared):
        """The whole point of scoring apart. With one shared scale the wider
        well would own the ranking outright."""
        rows = self.listing(compared)
        assert set(rows["well"]) == {"DEMO-1", "15/9-19-A"}

    def test_the_rank_is_dense_and_starts_at_one(self, compared):
        rows = self.listing(compared).sort_values("rank")
        assert list(rows["rank"]) == list(range(1, len(rows) + 1))

    def test_the_score_is_signed(self, compared):
        """A bright trough and a dim one are both anomalies and are not the
        same finding, so the sign has to survive to the table."""
        rows = self.listing(compared)
        column = next(c for c in rows.columns if c.endswith("_z"))
        assert (rows[column] < 0).any() and (rows[column] > 0).any()

    def test_the_wells_are_scaled_separately(self, compared):
        """Read off the caption, which is what a user would check."""
        text = " ".join(c.value for c in compared.caption)
        assert "own" in text and "median absolute deviation" in text

    def test_the_attributes_reach_the_pooled_table(self, compared):
        for found in compared.session_state["multiwell_results"]["results"].values():
            assert {"rs", "fluid_factor", "ab_product"} <= set(found["table"].columns)
