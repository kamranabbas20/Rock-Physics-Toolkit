"""End-to-end checks against a real well rather than the synthetic demo.

The bundled demo well is a clean three-layer model: 132 samples in time, every
curve complete, every boundary a step.  It is the right fixture for pinning a
known answer, and the wrong one for finding out what breaks on real data.
Every defect found in the blocking and reflector-picking work came from this
well and none of them from the demo — a reflector split on a neighbour's lobe,
a NaN polarity cast to ``INT_MIN``, and 42% of the reflector table describing
interfaces the seismic could not separate.

15/9-19-A is a North Sea well: 3905 samples over 3500–4095 m MD, with the gaps
real logs have — VSH on 2812 of 3905 samples, SW on 1965, RHOB carrying null
sentinels.  What it is for here is coverage of the messy path, so most of these
tests assert *invariants* rather than numbers.  The two that do pin numbers say
so, and exist to make a change in the defaults visible instead of silent.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

streamlit = pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest  # noqa: E402

from avo_qi.io.loader import read_well, standardise  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGES = os.path.join(HERE, "pages")
WELL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "data", "15_9_19_A.las")


def load():
    raw, units = read_well(WELL)
    return standardise(raw, units=units, name="15/9-19-A"), raw, units


def run_page(path, timeout=300):
    from avo_qi.ui import Settings

    well, raw, units = load()
    at = AppTest.from_file(path, default_timeout=timeout)
    at.session_state["well"] = well
    at.session_state["raw_df"] = raw
    at.session_state["raw_units"] = units
    at.session_state["settings"] = Settings()
    at.run()
    return at


def reflector_table(page):
    for element in page.dataframe:
        frame = element.value
        if hasattr(frame, "columns") and {"avo_class", "A_shuey"} <= set(frame.columns):
            return frame
    raise AssertionError("no reflector table on the page")


@pytest.fixture(scope="module")
def avo():
    return run_page(os.path.join(PAGES, "4_AVO_Classification.py"))


@pytest.fixture(scope="module")
def table(avo):
    assert not avo.exception
    return reflector_table(avo)


class TestTheWellLoads:
    def test_it_reads_with_its_gaps_intact(self):
        well, _, _ = load()
        df = well.df
        assert len(df) == 3905
        assert {"VP", "VS", "RHOB", "GR", "VSH", "SW", "DEPTH"} <= set(df.columns)

        # The null sentinel became NaN rather than surviving as -999.25, which
        # would sail through every finite check and poison the averages.
        for name in ("RHOB", "VSH", "SW"):
            values = df[name].to_numpy(float)
            assert not (values < -900).any(), f"{name} still carries a sentinel"
            assert np.isfinite(values).sum() < len(values), \
                f"{name} is complete here; this fixture is meant to have gaps"

    def test_the_sonics_are_physical(self):
        df = load()[0].df
        vp = df["VP"].to_numpy(float)
        vs = df["VS"].to_numpy(float)
        good = np.isfinite(vp) & np.isfinite(vs)
        assert (vs[good] < vp[good]).all()
        assert (vp[good] / vs[good] > np.sqrt(2)).all()


class TestEveryPageSurvivesIt:
    """The demo well is 132 samples of clean step boundaries. A page that only
    ever runs on that has not been run."""

    @pytest.mark.parametrize("page", [
        "1_Load_and_QC.py",
        "2_Data_and_Crossplots.py",
        "3_Synthetic_Gather.py",
        "4_AVO_Classification.py",
        "5_Rock_Physics.py",
    ])
    def test_it_runs_clean(self, page):
        at = run_page(os.path.join(PAGES, page))
        assert not at.exception, f"{page}: {at.exception}"


class TestTracePicking:
    """The invariant the AVO page rests on: the trace says where reflectors are."""

    @staticmethod
    def _full_stack(at):
        from avo_qi.core.synthetic import build_gather, full_stack
        from avo_qi.ui import build_wavelet, time_well

        settings = at.session_state["settings"]
        tw = time_well(at.session_state["well"], settings, settings.case)
        _, wavelet = build_wavelet(settings)
        return full_stack(build_gather(
            tw["VP"].to_numpy(float), tw["VS"].to_numpy(float),
            tw["RHOB"].to_numpy(float), settings.angles, wavelet,
            dt=settings.dt, method=settings.method))

    def test_every_reflector_is_a_turning_point(self, avo, table):
        from avo_qi.core.synthetic import _local_extrema

        stack = self._full_stack(avo)
        assert set(table["sample"].astype(int)) <= set(_local_extrema(stack).tolist())

    def test_the_polarity_reported_is_the_traces_own(self, avo, table):
        stack = self._full_stack(avo)
        for _, row in table.iterrows():
            expected = "peak" if stack[int(row["sample"])] > 0 else "trough"
            assert row["polarity"] == expected

    def test_every_event_gets_a_lobe_of_its_own(self, table):
        """This is the whole point of picking on the trace. Under log picking
        114 of 269 reflectors on this well had no turning point of their own
        and were blocked on a window belonging to a neighbour."""
        assert (table["blocking"] == "lobe").all()
        assert (table["lobe_samples"] > 0).all()

    def test_the_event_count_is_pinned(self, table):
        """A deliberate pin, not an invariant. 25 events at the default 5% cut
        and a 30 Hz Ricker; if a default moves, this should say so rather than
        the number changing quietly."""
        assert len(table) == 25

    def test_the_seismic_merges_many_log_interfaces_into_each_event(self, avo, table):
        """3905 log samples over ~316 samples of two-way time cannot yield 269
        separable events at 30 Hz, which is what a log-side threshold claimed."""
        assert len(table) < 40
        span = table["lobe_samples"].to_numpy(float)
        assert span.min() >= 2
        assert np.median(span) > 8      # a real lobe, not a couple of samples


class TestNoSilentNumericalDamage:
    def test_the_pipeline_raises_no_runtime_warnings(self):
        """A NaN cast through ``np.sign(...).astype(int)`` lands on INT_MIN and
        matches nothing, with a RuntimeWarning as the only clue. Gaps in this
        well's curves are what make that reachable."""
        from avo_qi.core.blocking import (blocked_reflectivity,
                                          half_cycle_samples, lobe_windows)
        from avo_qi.core.reflectivity import reflectivity_series
        from avo_qi.core.synthetic import build_gather, full_stack, trace_events
        from avo_qi.core.tuning import apparent_period
        from avo_qi.ui import Settings, build_wavelet, time_well

        settings = Settings()
        tw = time_well(load()[0], settings, settings.case)
        vp = tw["VP"].to_numpy(float)
        vs = tw["VS"].to_numpy(float)
        rho = tw["RHOB"].to_numpy(float)
        _, wavelet = build_wavelet(settings)
        window = half_cycle_samples(apparent_period(wavelet, settings.dt), settings.dt)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            stack = full_stack(build_gather(vp, vs, rho, settings.angles, wavelet,
                                            dt=settings.dt, method=settings.method))
            events = trace_events(stack, relative=settings.threshold)
            bounds = lobe_windows(stack, events["index"],
                                  polarity=events["polarity"].astype(float),
                                  max_half_width=2 * window)
            blocked = blocked_reflectivity(
                vp, vs, rho, events["index"], settings.angles, window=window,
                method="backus", guard=2, reflectivity_method=settings.method,
                bounds=bounds)
            reflectivity_series(vp, vs, rho, settings.angles, method=settings.method)

        for key in ("vp_upper", "vs_upper", "rho_upper",
                    "vp_lower", "vs_lower", "rho_lower"):
            assert np.isfinite(blocked[key]).all(), f"{key} came out non-finite"

    def test_the_fitted_intercepts_and_gradients_are_all_finite(self, table):
        for column in ("A_shuey", "B_shuey", "A_ar", "B_ar"):
            assert np.isfinite(table[column].to_numpy(float)).all()

    def test_shuey_and_aki_richards_agree(self, table):
        """Two independent fits to the same layers. A real well is where a
        disagreement would show up if the fit were being fed rubbish."""
        assert np.abs(table[["dA", "dB"]].to_numpy(float)).max() < 1e-6


class TestZonationFromTops:
    """15/9-19-A has no ZONE curve, so without tops the whole zone machinery —
    the filter, the boundary flag, the per-zone summaries — is dead on it.

    The depths below are invented for the test, not this field's real
    stratigraphy; what is being checked is the wiring, not the geology.
    """

    TOPS = [{"zone": "Upper", "top": 3500.0},
            {"zone": "Middle", "top": 3700.0},
            {"zone": "Reservoir", "top": 3900.0}]

    def test_the_well_has_no_zone_curve_to_begin_with(self):
        assert "ZONE" not in load()[0].df.columns

    def test_tops_produce_intervals_closed_at_the_bottom_of_the_well(self):
        from avo_qi.ui import Settings, well_zones

        well = load()[0]
        settings = Settings()
        assert well_zones(well, settings) is None      # nothing without tops

        settings.zone_tops = list(self.TOPS)
        intervals = well_zones(well, settings)
        assert list(intervals["zone"]) == ["Upper", "Middle", "Reservoir"]
        assert list(intervals["top"]) == [3500.0, 3700.0, 3900.0]
        # Each runs to the next, and the deepest is closed just *past* the last
        # sample rather than on it: assignment is `top <= depth < base`, so a
        # base exactly at total depth leaves the bottom sample unzoned.
        assert list(intervals["base"])[:2] == [3700.0, 3900.0]
        deepest = float(well.df["DEPTH"].max())
        step = float(np.median(np.diff(np.sort(well.df["DEPTH"].to_numpy(float)))))
        assert intervals["base"].iloc[-1] == pytest.approx(deepest + step)
        assert intervals["base"].iloc[-1] > deepest

    def test_every_sample_gets_a_zone(self):
        from avo_qi.core.zones import UNZONED
        from avo_qi.ui import Settings, zone_labels

        well = load()[0]
        settings = Settings()
        settings.zone_tops = list(self.TOPS)
        labels = zone_labels(well.df, well, settings)
        assert set(labels) == {"Upper", "Middle", "Reservoir"}
        assert UNZONED not in set(labels)

    def test_the_avo_page_carries_a_zone_per_event(self):
        at = run_page(os.path.join(PAGES, "4_AVO_Classification.py"))
        assert set(reflector_table(at)["zone"]) == {"unzoned"}

        at.session_state["settings"].zone_tops = list(self.TOPS)
        at.run()
        assert not at.exception
        table = reflector_table(at)
        assert set(table["zone"]) <= {"Upper", "Middle", "Reservoir"}
        assert len(set(table["zone"])) > 1

    def test_events_on_a_top_are_flagged_as_zone_boundaries(self):
        at = run_page(os.path.join(PAGES, "4_AVO_Classification.py"))
        at.session_state["settings"].zone_tops = list(self.TOPS)
        at.run()
        table = reflector_table(at)
        flagged = table[table["is_zone_boundary"]]
        assert len(flagged), "no event straddles a top"
        # A flagged event has different zones above and below it.
        for _, row in flagged.iterrows():
            assert row["zone"] != row["zone_below"]

    def test_a_top_at_the_very_start_of_the_well_still_labels_samples(self):
        """The warning on the load page tested the *top* against the well's
        range and called this empty: the shallowest top is 3500.0 while the
        first sample is at 3500.0183, and `depth >= top` matches it fine."""
        from avo_qi.core.zones import assign_zones
        from avo_qi.ui import Settings, well_zones

        well = load()[0]
        settings = Settings()
        settings.zone_tops = list(self.TOPS)
        intervals = well_zones(well, settings)

        depth = well.df["DEPTH"].to_numpy(float)
        assert depth.min() > 3500.0, "the fixture must start below the top"
        labels = assign_zones(depth, intervals)
        assert (labels == "Upper").sum() > 0

    def test_the_zone_filter_narrows_the_events(self):
        at = run_page(os.path.join(PAGES, "4_AVO_Classification.py"))
        at.session_state["settings"].zone_tops = list(self.TOPS)
        at.run()
        everything = len(reflector_table(at))

        at.session_state["settings"].zones = ["Reservoir"]
        at.run()
        assert not at.exception
        narrowed = reflector_table(at)
        assert 0 < len(narrowed) < everything
        # Kept when *either* side is in the zone, so the top of the interval
        # survives along with everything inside it.
        assert {"Reservoir"} <= set(narrowed["zone"]) | set(narrowed["zone_below"])

    def test_tops_do_not_leak_into_the_next_well(self):
        """The same failure the zone *selection* had once: a top is a depth in
        the well it was typed against and means nothing in another.

        Driven by actually loading a second well through the page's own button,
        so it is `set_well` doing the clearing and not the test.
        """
        at = run_page(os.path.join(PAGES, "1_Load_and_QC.py"))
        at.session_state["settings"].zone_tops = list(self.TOPS)
        at.run()
        assert at.session_state["settings"].zone_tops
        assert at.session_state["well"].name == "15/9-19-A"

        next(b for b in at.button if b.label == "Load demo well").click().run()
        assert not at.exception
        assert at.session_state["well"].name == "DEMO-1"
        assert at.session_state["settings"].zone_tops == []


class TestLithologyOnRealCurves:
    def test_pairs_come_from_the_wells_own_vsh(self, table):
        from avo_qi.core.lithology import LITHOLOGIES

        pairs = set(table["litho_pair"])
        assert pairs, "no lithology pairs at all"
        # Real VSH here spans 0.02–0.80, so more than one class must appear.
        named = {p for p in pairs if "undefined" not in p}
        assert len(named) > 1
        for pair in named:
            upper, lower = pair.split(" over ")
            assert upper in LITHOLOGIES and lower in LITHOLOGIES

    def test_the_pair_filter_narrows_the_table(self):
        # Its own page run: setting a widget mutates the AppTest, and the
        # module-scoped `avo` fixture is shared with every test above.
        at = run_page(os.path.join(PAGES, "4_AVO_Classification.py"))
        picker = next(m for m in at.multiselect
                      if m.label == "Interface pairs to keep")
        assert len(picker.options) > 1
        target = picker.options[0]
        picker.set_value([target]).run()
        assert not at.exception
        assert set(reflector_table(at)["litho_pair"]) == {target}
