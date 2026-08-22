"""Tests for the model-misfit measures."""

from __future__ import annotations

import numpy as np
import pytest

from avo_qi.core.gassmann import gassmann_dry
from avo_qi.core.misfit import bounds_check, curve_misfit, log_misfit
from avo_qi.core.rockphysics import FLUIDS, MINERALS, bulk_modulus, hashin_shtrikman


class TestBoundsCheck:
    def test_data_wholly_inside_scores_one_and_names_nobody(self):
        out = bounds_check([2.0, 3.0, 4.0], 1.0, 5.0)
        assert out.fraction_inside == 1.0
        assert out.n == 3 and out.n_inside == 3
        assert out.outside.size == 0
        assert "Nothing falls outside" in out.describe()

    def test_data_wholly_outside_scores_zero_and_names_everybody(self):
        out = bounds_check([10.0, 11.0, 12.0], 1.0, 5.0)
        assert out.fraction_inside == 0.0
        assert list(out.outside) == [0, 1, 2]
        assert out.above.all() and not out.below.any()

    def test_reports_which_side_each_sample_missed(self):
        out = bounds_check([0.5, 3.0, 9.0], 1.0, 5.0)
        assert list(out.below) == [True, False, False]
        assert list(out.above) == [False, False, True]
        assert out.fraction_inside == pytest.approx(1 / 3)

    def test_the_worst_breach_is_signed_by_the_side_it_missed(self):
        below = bounds_check([0.2], 1.0, 5.0)
        assert below.worst_breach == pytest.approx(-0.8)
        above = bounds_check([7.0], 1.0, 5.0)
        assert above.worst_breach == pytest.approx(2.0)

    def test_finds_the_largest_breach_not_the_first(self):
        out = bounds_check([0.9, 20.0, 0.5], 1.0, 5.0)
        assert out.worst_index == 1
        assert out.worst_breach == pytest.approx(15.0)

    def test_accepts_per_sample_bounds(self):
        values = np.array([1.0, 2.0, 3.0])
        out = bounds_check(values, [0.0, 2.5, 0.0], [5.0, 5.0, 5.0])
        assert list(out.outside) == [1]

    def test_bounds_given_the_wrong_way_round_still_work(self):
        assert bounds_check([3.0], 5.0, 1.0).fraction_inside == 1.0

    def test_ignores_non_finite_samples(self):
        out = bounds_check([2.0, np.nan, 9.0], 1.0, 5.0)
        assert out.n == 2 and out.n_inside == 1
        assert list(out.outside) == [2]

    def test_an_empty_selection_reports_nothing_rather_than_a_perfect_score(self):
        out = bounds_check([np.nan, np.nan], 1.0, 5.0)
        assert out.n == 0
        assert np.isnan(out.fraction_inside)
        assert "No samples" in out.describe()

    def test_describe_names_the_depth_of_the_worst_breach(self):
        out = bounds_check([3.0, 0.1], 1.0, 5.0)
        text = out.describe(unit="GPa", depths=[2100.0, 2104.0])
        assert "2104.0 m" in text and "GPa" in text and "below the lower" in text

    def test_the_suspension_bound_still_catches_the_original_finding(self):
        """The check that started all of this, run end to end.

        The demo well's brine sand was originally 2500 / 1450 / 2.30 at a
        porosity of 0.26.  Its bulk modulus falls below the Hashin-Shtrikman
        lower bound for quartz and brine, which is physically impossible and is
        what exposed the porosity as invented.  The rebuilt layer sits inside.
        """
        k_min, g_min = MINERALS["quartz"][0], MINERALS["quartz"][1]
        k_fl = FLUIDS["brine"][0]
        hs = hashin_shtrikman(k_min, g_min, k_fl, 0.0, 1.0 - 0.26)

        before = bulk_modulus(np.array([2500.0]), np.array([1450.0]), np.array([2.30]))
        after = bulk_modulus(np.array([2830.2]), np.array([1498.9]), np.array([2.2444]))

        assert bounds_check(before, hs["K_lower"], hs["K_upper"]).fraction_inside == 0.0
        assert bounds_check(after, hs["K_lower"], hs["K_upper"]).fraction_inside == 1.0
        # And the same rock is un-substitutable, which is the same finding
        # arrived at by the other route.
        assert not gassmann_dry(before, k_min, k_fl, 0.26).valid[0]


class TestLogMisfit:
    def test_a_perfect_prediction_has_no_bias_and_no_scatter(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        out = log_misfit(y, y)
        assert out.bias == 0.0 and out.scatter == 0.0 and out.rms == 0.0
        assert out.correlation == pytest.approx(1.0)

    def test_a_constant_offset_is_reported_as_bias(self):
        y = np.linspace(2000.0, 3000.0, 50)
        out = log_misfit(y + 100.0, y)
        assert out.bias == pytest.approx(100.0)
        assert out.scatter == pytest.approx(0.0, abs=1e-9)
        assert out.relative_bias == pytest.approx(100.0 / np.median(y))

    def test_running_low_is_signed_and_said_in_words(self):
        y = np.full(20, 3000.0)
        out = log_misfit(y - 60.0, y)
        assert out.bias == pytest.approx(-60.0)
        assert "low" in out.describe(name="Vp", unit="m/s")

    def test_the_median_shrugs_off_a_few_bad_samples(self):
        """A mean would be dragged around by one washed-out sample; the point
        of using a median is that it is not."""
        y = np.full(101, 3000.0)
        predicted = y.copy()
        predicted[0] = 30000.0
        out = log_misfit(predicted, y)
        assert out.bias == pytest.approx(0.0)
        assert out.rms > 1000.0            # the outlier is still visible here
        assert out.worst_index == 0

    def test_worst_index_points_into_the_original_array(self):
        predicted = np.array([1.0, np.nan, 1.0, 9.0])
        measured = np.array([1.0, 5.0, 1.0, 1.0])
        out = log_misfit(predicted, measured)
        assert out.n == 3
        assert out.worst_index == 3

    def test_ignores_samples_missing_from_either_side(self):
        out = log_misfit([1.0, np.nan, 3.0], [1.0, 2.0, np.nan])
        assert out.n == 1 and out.bias == 0.0

    def test_an_empty_comparison_says_so_rather_than_scoring_zero(self):
        out = log_misfit([np.nan], [1.0])
        assert out.n == 0
        assert np.isnan(out.bias) and np.isnan(out.rms)
        assert "nothing to compare" in out.describe(name="Vp")

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            log_misfit([1.0, 2.0], [1.0])


class TestCurveMisfit:
    def test_data_on_the_curve_has_no_misfit(self):
        mx = np.linspace(0.0, 0.4, 41)
        my = 4000.0 - 5000.0 * mx
        x = np.array([0.1, 0.2, 0.3])
        out = curve_misfit(x, 4000.0 - 5000.0 * x, mx, my)
        assert out.bias == pytest.approx(0.0, abs=1e-9)
        assert out.n == 3

    def test_measures_the_offset_from_the_curve(self):
        mx = np.linspace(0.0, 0.4, 41)
        my = np.full(41, 3000.0)
        out = curve_misfit(np.array([0.1, 0.2]), np.array([2900.0, 2900.0]), mx, my)
        assert out.bias == pytest.approx(100.0)

    def test_samples_beyond_the_curve_are_dropped_not_clamped(self):
        """Interpolating past the end would invent a fit where no model exists."""
        mx = np.linspace(0.1, 0.3, 21)
        my = np.full(21, 3000.0)
        out = curve_misfit(np.array([0.05, 0.2, 0.9]), np.full(3, 3000.0), mx, my)
        assert out.n == 1

    def test_handles_an_unsorted_model_curve(self):
        mx = np.array([0.3, 0.1, 0.2])
        my = np.array([2000.0, 4000.0, 3000.0])
        out = curve_misfit(np.array([0.15]), np.array([3500.0]), mx, my)
        assert out.bias == pytest.approx(0.0, abs=1e-9)

    def test_a_curve_of_one_point_is_not_a_curve(self):
        out = curve_misfit([0.2], [3000.0], [0.2], [3000.0])
        assert out.n == 0

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="x and y must have the same length"):
            curve_misfit([0.1, 0.2], [3000.0], [0.1, 0.3], [1.0, 2.0])
