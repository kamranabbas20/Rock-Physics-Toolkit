"""Rock properties per reflector, and whether the class depends on them.

Two things here are easy to get wrong and quiet when they are. A net-to-gross
computed over a half-lobe with a gap in the VSH must not count the gap as
non-net — that reads as a worse reservoir rather than a shorter interval. And
the reservoir side of a reflector is *not* always the one below: a sand base
carries its reservoir above, and getting that backwards mixes seal in with
reservoir and washes out the trend the panel exists to show.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from avo_qi.core.blocking import lobe_windows
from avo_qi.core.properties import (NET_CUTOFFS, class_dependence,
                                    class_property_summary, lobe_property,
                                    net_flag, net_to_gross, pick_side,
                                    reservoir_side)


def windows(count=1, upper=(0, 5), lower=(5, 10), resolved=True):
    """A lobe_windows-shaped result, built by hand so a test can say exactly
    which samples a property is averaged over."""
    return {
        "upper_start": np.array([upper[0]] * count),
        "upper_stop": np.array([upper[1]] * count),
        "lower_start": np.array([lower[0]] * count),
        "lower_stop": np.array([lower[1]] * count),
        "resolved": np.array([resolved] * count),
    }


class TestLobeAveraging:
    def test_it_averages_over_the_window_it_is_given(self):
        values = np.concatenate([np.full(5, 0.10), np.full(5, 0.30)])
        found = lobe_property(values, windows())
        assert found["upper"][0] == pytest.approx(0.10)
        assert found["lower"][0] == pytest.approx(0.30)
        assert found["contrast"][0] == pytest.approx(0.20)

    def test_the_contrast_is_signed_downwards(self):
        """Lower minus upper, so a porosity increase downwards is positive —
        the same direction the reflection coefficient is taken in."""
        values = np.concatenate([np.full(5, 0.30), np.full(5, 0.10)])
        assert lobe_property(values, windows())["contrast"][0] < 0

    def test_a_gap_is_skipped_not_counted_as_zero(self):
        values = np.array([0.2, np.nan, 0.2, np.nan, 0.2] + [0.1] * 5)
        assert lobe_property(values, windows())["upper"][0] == pytest.approx(0.2)

    def test_an_empty_window_is_nan_rather_than_zero(self):
        values = np.concatenate([np.full(5, np.nan), np.full(5, 0.3)])
        found = lobe_property(values, windows())
        assert np.isnan(found["upper"][0])
        assert np.isnan(found["contrast"][0])

    def test_the_median_is_offered_for_spiky_logs(self):
        values = np.array([0.2, 0.2, 0.2, 0.2, 9.9] + [0.1] * 5)
        assert lobe_property(values, windows())["upper"][0] > 1.0
        assert lobe_property(values, windows(),
                             statistic="median")["upper"][0] == pytest.approx(0.2)

    def test_an_unresolved_reflector_falls_back_to_the_interface(self):
        """No lobe of its own means no halves to average; the two samples
        either side are what is left."""
        values = np.array([0.10] * 5 + [0.30] * 5)
        bounds = windows(resolved=False)
        found = lobe_property(values, bounds, samples=np.array([4]))
        assert found["upper"][0] == pytest.approx(0.10)
        assert found["lower"][0] == pytest.approx(0.30)

    def test_it_uses_the_windows_the_blocking_actually_produced(self):
        """Not a hand-built dict: the same call the pipeline makes."""
        trace = np.array([0.0, -0.2, -0.6, -1.0, -0.6, -0.2, 0.0, 0.3, 0.1, 0.0])
        bounds = lobe_windows(trace, np.array([3]), polarity=np.array([-1.0]))
        assert bounds["resolved"][0]
        # A ramp, so each half-lobe average reports the samples it covered.
        values = np.arange(10, dtype=float)
        found = lobe_property(values, bounds)
        upper = values[bounds["upper_start"][0]:bounds["upper_stop"][0]]
        lower = values[bounds["lower_start"][0]:bounds["lower_stop"][0]]
        assert upper.size and lower.size
        assert found["upper"][0] == pytest.approx(upper.mean())
        assert found["lower"][0] == pytest.approx(lower.mean())
        # The trough runs 1..5 around its extremum at 3, so the halves sit
        # either side of it rather than both landing on the same rock.
        assert found["upper"][0] < 3 <= found["lower"][0]


class TestNetToGross:
    def test_a_clean_sand_under_a_shale_reads_one_and_zero(self):
        vsh = np.concatenate([np.full(5, 0.8), np.full(5, 0.1)])
        found = net_to_gross(vsh, windows())
        assert found["upper"][0] == pytest.approx(0.0)
        assert found["lower"][0] == pytest.approx(1.0)

    def test_it_is_the_fraction_of_the_interval_that_is_net(self):
        vsh = np.array([0.8] * 5 + [0.1, 0.1, 0.1, 0.9, 0.9])
        assert net_to_gross(vsh, windows())["lower"][0] == pytest.approx(0.6)

    def test_a_gap_shortens_the_interval_rather_than_lowering_the_ratio(self):
        """The trap this function exists to avoid: three net samples out of
        three logged is a net-to-gross of 1, not 0.6, even where two samples
        carry no VSH at all."""
        vsh = np.array([0.8] * 5 + [0.1, 0.1, 0.1, np.nan, np.nan])
        assert net_to_gross(vsh, windows())["lower"][0] == pytest.approx(1.0)

    def test_no_vsh_at_all_is_unknown_rather_than_barren(self):
        vsh = np.array([0.8] * 5 + [np.nan] * 5)
        assert np.isnan(net_to_gross(vsh, windows())["lower"][0])

    def test_net_pay_is_stricter_than_net_sand(self):
        vsh = np.full(10, 0.1)                       # all clean
        phi = np.concatenate([np.full(5, 0.25), np.full(5, 0.02)])
        sw = np.full(10, 0.3)
        sand = net_to_gross(vsh, windows())
        pay = net_to_gross(vsh, windows(), phi=phi, sw=sw)
        assert sand["lower"][0] == pytest.approx(1.0)
        assert pay["lower"][0] == pytest.approx(0.0)   # clean but tight
        assert pay["upper"][0] == pytest.approx(1.0)

    def test_water_leg_is_clean_sand_but_not_pay(self):
        vsh = np.full(10, 0.1)
        phi = np.full(10, 0.25)
        sw = np.concatenate([np.full(5, 0.3), np.full(5, 0.95)])
        pay = net_to_gross(vsh, windows(), phi=phi, sw=sw)
        assert pay["upper"][0] == pytest.approx(1.0)
        assert pay["lower"][0] == pytest.approx(0.0)

    def test_the_flag_is_three_valued(self):
        flag = net_flag(np.array([0.1, 0.9, np.nan]))
        assert flag[0] == 1.0 and flag[1] == 0.0 and np.isnan(flag[2])

    def test_a_criterion_of_the_wrong_length_is_refused(self):
        with pytest.raises(ValueError, match="one value per sample"):
            net_flag(np.zeros(10), phi=np.zeros(3))

    def test_the_defaults_are_the_documented_ones(self):
        assert set(NET_CUTOFFS) == {"vsh", "phi", "sw"}
        assert net_flag(np.array([NET_CUTOFFS["vsh"] - 1e-6]))[0] == 1.0
        assert net_flag(np.array([NET_CUTOFFS["vsh"] + 1e-6]))[0] == 0.0


class TestTheReservoirSide:
    def test_a_shale_over_sand_top_carries_its_reservoir_below(self):
        assert reservoir_side([0.8], [0.1])[0] == "below"

    def test_a_sand_over_shale_base_carries_it_above(self):
        """The case that makes 'always plot the layer below' wrong."""
        assert reservoir_side([0.1], [0.8])[0] == "above"

    def test_one_side_missing_leaves_the_other_side_standing(self):
        assert reservoir_side([np.nan], [0.3])[0] == "below"
        assert reservoir_side([0.3], [np.nan])[0] == "above"

    def test_neither_side_known_is_unknown(self):
        assert reservoir_side([np.nan], [np.nan])[0] == "unknown"

    def test_picking_follows_the_side(self):
        side = reservoir_side([0.8, 0.1, np.nan], [0.1, 0.8, np.nan])
        picked = pick_side(side, [0.10, 0.25, 0.3], [0.30, 0.05, 0.3])
        assert picked[0] == pytest.approx(0.30)      # took the lower
        assert picked[1] == pytest.approx(0.25)      # took the upper
        assert np.isnan(picked[2])


class TestClassSummary:
    def test_it_reports_a_row_per_class_in_the_order_given(self):
        classes = np.array(["III", "I", "III", "IV"], dtype=object)
        values = np.array([0.30, 0.05, 0.28, 0.12])
        rows = class_property_summary(classes, values, order=["I", "IIp", "III"])
        assert [r["avo_class"] for r in rows] == ["I", "IIp", "III"]
        assert rows[2]["n"] == 2
        assert rows[2]["median"] == pytest.approx(0.29)

    def test_an_empty_class_is_reported_rather_than_dropped(self):
        """'Class I has no porosity here' is a finding, not a missing row."""
        rows = class_property_summary(np.array(["III"], dtype=object),
                                      np.array([0.3]), order=["I", "III"])
        assert rows[0]["n"] == 0 and np.isnan(rows[0]["median"])

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError, match="same length"):
            class_property_summary(np.array(["III"]), np.array([1.0, 2.0]))


class TestClassDependence:
    @staticmethod
    def split(low, high, n=12):
        classes = np.array(["I"] * n + ["III"] * n, dtype=object)
        rng = np.random.default_rng(7)
        values = np.concatenate([rng.normal(low, 0.01, n),
                                 rng.normal(high, 0.01, n)])
        return classes, values

    def test_a_property_that_separates_the_classes_is_found(self):
        found = class_dependence(*self.split(0.08, 0.30))
        assert found["p"] < 0.01
        assert found["epsilon_squared"] > 0.5
        assert found["k"] == 2 and found["n"] == 24

    def test_a_property_that_does_not_separate_them_is_not_found(self):
        found = class_dependence(*self.split(0.20, 0.20))
        assert found["p"] > 0.05
        assert found["epsilon_squared"] < 0.2

    def test_the_effect_size_ranks_two_properties_apart(self):
        """The number worth reading: with the same n and the same p-value
        floor, the better separator has to score higher."""
        strong = class_dependence(*self.split(0.05, 0.35))
        weak = class_dependence(*self.split(0.19, 0.21))
        assert strong["epsilon_squared"] > weak["epsilon_squared"]

    def test_a_thin_class_is_dropped_and_named(self):
        classes = np.array(["I"] * 8 + ["III"] * 8 + ["IV"] * 2, dtype=object)
        values = np.arange(18, dtype=float)
        found = class_dependence(classes, values, min_per_group=3)
        assert found["dropped"] == ["IV"]
        assert found["groups"] == ["I", "III"]
        assert found["n"] == 16                       # the two are left out

    def test_one_class_cannot_be_tested_against_itself(self):
        found = class_dependence(np.array(["III"] * 9, dtype=object),
                                 np.arange(9, dtype=float))
        assert np.isnan(found["h"]) and np.isnan(found["p"])
        assert "two classes" in found["reason"]

    def test_a_constant_property_says_so_rather_than_raising(self):
        classes = np.array(["I"] * 6 + ["III"] * 6, dtype=object)
        found = class_dependence(classes, np.full(12, 0.25))
        assert found["reason"] == "the property is constant"
        assert np.isnan(found["p"])

    def test_missing_values_are_left_out_of_the_test(self):
        classes = np.array(["I"] * 6 + ["III"] * 6, dtype=object)
        values = np.concatenate([np.full(3, 0.1), np.full(3, np.nan),
                                 np.full(6, 0.3)])
        found = class_dependence(classes, values, min_per_group=3)
        assert found["n"] == 9


class TestThePipelineCarriesThem:
    """The columns the panel plots, on the wells they will actually be read
    from. What matters here is not the arithmetic — that is tested above — but
    that the properties describe the *same rock the class came from*."""

    @staticmethod
    def analysed(which="demo"):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from avo_qi.analysis import reflector_analysis
        from avo_qi.ui import Settings

        if which == "demo":
            from test_app_smoke import demo_well as load
        else:
            from test_multi_well import second_well as load
        return reflector_analysis(load()[0], Settings())["table"]

    def test_every_side_of_every_curve_arrives(self):
        table = self.analysed()
        for curve in ("vsh", "phi", "sw", "ntg"):
            for side in ("above", "below", "res"):
                assert f"{curve}_{side}" in table.columns, f"{curve}_{side}"
            assert f"d_{curve}" in table.columns

    def test_the_contrast_is_the_two_sides_subtracted(self):
        table = self.analysed()
        assert np.allclose(table["d_phi"],
                           table["phi_below"] - table["phi_above"],
                           equal_nan=True)

    def test_the_reservoir_side_is_the_cleaner_one(self):
        table = self.analysed()
        picked = table[table["reservoir_side"] != "unknown"]
        assert len(picked)
        chosen = np.where(picked["reservoir_side"] == "above",
                          picked["vsh_above"], picked["vsh_below"])
        other = np.where(picked["reservoir_side"] == "above",
                         picked["vsh_below"], picked["vsh_above"])
        assert (chosen <= other).all()
        assert np.allclose(chosen, picked["vsh_res"], equal_nan=True)

    def test_a_sand_base_takes_its_reservoir_from_above(self):
        """The demo well's gas sand has a base as well as a top, and the base
        is where 'always read the layer below' would report the shale."""
        table = self.analysed()
        bases = table[table["litho_pair"].str.startswith("sand over shale")]
        assert len(bases)
        assert (bases["reservoir_side"] == "above").all()
        assert np.allclose(bases["ntg_res"], bases["ntg_above"], equal_nan=True)

    def test_the_gas_sand_top_reads_as_clean_porous_reservoir(self):
        """A known answer on a well built by hand: the Class III top has the
        sand below it, at the porosity the generator put there."""
        table = self.analysed()
        top = table[table["litho_pair"] == "shale over sand"].iloc[0]
        assert top["reservoir_side"] == "below"
        assert top["ntg_below"] > 0.8
        assert top["phi_below"] > top["phi_above"]
        assert top["vsh_below"] < 0.2

    def test_the_real_well_reports_nothing_where_nothing_was_logged(self):
        table = self.analysed("real")
        shallow = table[table["depth"] < 3666.0]
        assert len(shallow) >= 3
        assert shallow["ntg_res"].isna().all()
        assert (shallow["reservoir_side"] == "unknown").all()

    def test_properties_can_be_turned_off(self):
        from avo_qi.analysis import reflector_analysis
        from avo_qi.ui import Settings
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from test_app_smoke import demo_well

        table = reflector_analysis(demo_well()[0], Settings(),
                                   properties=False)["table"]
        assert not [c for c in table.columns if c.endswith("_res")]

    def test_the_net_cutoff_moves_the_net_to_gross(self):
        """It is the sidebar's cutoff that decides, not a constant buried in
        the pipeline."""
        from avo_qi.analysis import reflector_analysis
        from avo_qi.ui import Settings
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from test_app_smoke import demo_well

        well = demo_well()[0]
        loose = Settings()
        loose.net_cutoffs = dict(loose.net_cutoffs, vsh=0.95)
        strict = Settings()
        strict.net_cutoffs = dict(strict.net_cutoffs, vsh=0.05)
        a = reflector_analysis(well, loose)["table"]["ntg_below"]
        b = reflector_analysis(well, strict)["table"]["ntg_below"]
        assert a.sum() > b.sum()
