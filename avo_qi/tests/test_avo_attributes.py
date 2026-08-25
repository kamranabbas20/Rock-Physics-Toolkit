"""AVO attributes, checked against the physics they are derived from.

The pseudo-shear reflectivity and the fluid factor are both algebra on top of
the two-term Aki-Richards form, which means they can be tested the strong way:
build an interface from Vp, Vs and rho, fit A and B off its *exact* Zoeppritz
response, and ask whether the attribute comes back to the S-impedance
reflectivity the logs say it should be. Anything less — asserting the formula
against itself — would pass with the coefficients transposed.
"""

from __future__ import annotations

import numpy as np
import pytest

from avo_qi.core.avo import background_trend, shuey_fit
from avo_qi.core.avo_attributes import (MUDROCK_SLOPE, anomaly_ranking,
                                        chi_rotation, chi_sweep, fluid_factor,
                                        pseudo_shear_reflectivity, trend_chi)
from avo_qi.core.reflectivity import reflectivity_series


def interface(vp1, vs1, rho1, vp2, vs2, rho2, angles=None):
    """A and B fitted off the exact Zoeppritz response of one interface."""
    angles = np.arange(0.0, 31.0, 2.0) if angles is None else angles
    rc = reflectivity_series(np.array([vp1, vp2]), np.array([vs1, vs2]),
                             np.array([rho1, rho2]), angles, method="zoeppritz")
    A, B = shuey_fit(rc[0:1, :], angles)
    return float(A[0]), float(B[0])


def impedance_reflectivity(v1, rho1, v2, rho2):
    """The answer the logs give: (I2 - I1) / (I2 + I1)."""
    i1, i2 = v1 * rho1, v2 * rho2
    return (i2 - i1) / (i2 + i1)


class TestPseudoShearReflectivity:
    def test_it_recovers_the_shear_impedance_reflectivity(self):
        """The claim the attribute makes, tested end to end: a soft gas sand
        under shale, A and B read off exact Zoeppritz, Rs back to within the
        linearisation's own error."""
        shale = (3000.0, 1500.0, 2.40)
        sand = (2600.0, 1600.0, 2.10)
        A, B = interface(*shale, *sand)

        expected = impedance_reflectivity(shale[1], shale[2], sand[1], sand[2])
        found = pseudo_shear_reflectivity(A, B, vp_vs=shale[0] / shale[1])
        assert found == pytest.approx(expected, abs=0.02)

    def test_the_intercept_is_already_the_p_impedance_reflectivity(self):
        """Not this function's job, but the pair only means anything together:
        A *is* Rp, so Rs from A and B has to be read beside it."""
        shale = (3000.0, 1500.0, 2.40)
        sand = (2600.0, 1600.0, 2.10)
        A, _ = interface(*shale, *sand)
        assert A == pytest.approx(
            impedance_reflectivity(shale[0], shale[2], sand[0], sand[2]),
            abs=0.005)

    def test_at_vp_vs_two_it_is_exactly_a_minus_b_over_two(self):
        """The density terms cancel identically at k = 1/4, which is why every
        textbook writes the relation there. A formula that only *approximately*
        agrees at 2 has a coefficient wrong."""
        A, B = 0.08, -0.16
        assert pseudo_shear_reflectivity(A, B, vp_vs=2.0) == pytest.approx(
            (A - B) / 2.0, rel=1e-12)

    def test_a_different_background_ratio_gives_a_different_answer(self):
        """Otherwise the parameter is decoration."""
        assert pseudo_shear_reflectivity(0.05, -0.10, vp_vs=1.7) != \
            pytest.approx(pseudo_shear_reflectivity(0.05, -0.10, vp_vs=2.4))

    def test_it_is_vectorised(self):
        found = pseudo_shear_reflectivity(np.array([0.1, 0.0, -0.1]),
                                          np.array([-0.2, 0.0, 0.2]))
        assert found.shape == (3,)
        assert found[1] == pytest.approx(0.0)

    @pytest.mark.parametrize("bad", [0.0, -2.0])
    def test_a_nonsense_ratio_is_refused(self, bad):
        with pytest.raises(ValueError, match="positive"):
            pseudo_shear_reflectivity(0.1, -0.1, vp_vs=bad)


class TestTheFluidFactor:
    @staticmethod
    def on_the_mudrock_line(vs1, vs2, rho=2.35):
        """Two rocks whose velocities both satisfy Vp = 1.16 Vs + 1360, at
        equal density — the case the fluid factor is defined to miss."""
        return ((MUDROCK_SLOPE * vs1 + 1360.0, vs1, rho),
                (MUDROCK_SLOPE * vs2 + 1360.0, vs2, rho))

    def test_a_brine_clastic_on_the_trend_has_none(self):
        upper, lower = self.on_the_mudrock_line(1400.0, 1650.0)
        A, B = interface(*upper, *lower)
        rs = pseudo_shear_reflectivity(A, B, vp_vs=upper[0] / upper[1])
        found = fluid_factor(A, rs, vp_vs=upper[0] / upper[1])
        assert abs(found) < 0.01

    def test_a_gas_sand_off_the_trend_has_one(self):
        """The whole point: same shale, one lower layer on the mudrock line
        and one softened by gas, and only the second shows up."""
        upper, brine = self.on_the_mudrock_line(1400.0, 1650.0)
        gas = (brine[0] * 0.85, brine[1] * 1.02, brine[2] * 0.93)
        ratio = upper[0] / upper[1]

        def factor(lower):
            A, B = interface(*upper, *lower)
            return fluid_factor(A, pseudo_shear_reflectivity(A, B, vp_vs=ratio),
                                vp_vs=ratio)

        assert abs(factor(gas)) > 5 * abs(factor(brine))

    def test_a_density_contrast_alone_leaves_a_residual(self):
        """Documented honestly rather than wished away: the reflectivity form
        is not zero on the trend when the densities differ, so a compaction
        boundary has a fluid factor and the attribute must be read against its
        own background, not against zero."""
        upper, lower = self.on_the_mudrock_line(1500.0, 1500.0)
        heavier = (lower[0], lower[1], lower[2] * 1.10)
        ratio = upper[0] / upper[1]
        A, B = interface(*upper, *heavier)
        found = fluid_factor(A, pseudo_shear_reflectivity(A, B, vp_vs=ratio),
                             vp_vs=ratio)

        # The predicted residual: ½ (drho/rho) (1 - slope Vs/Vp).
        contrast = 2 * (heavier[2] - upper[2]) / (heavier[2] + upper[2])
        predicted = 0.5 * contrast * (1 - MUDROCK_SLOPE / ratio)
        assert found == pytest.approx(predicted, abs=0.01)
        assert abs(found) > 0.01           # not zero, which is the point

    def test_the_trend_slope_is_a_parameter(self):
        """1.16 is a global clastic average; a field with its own Vp-Vs trend
        has to be able to use it."""
        assert fluid_factor(0.1, 0.05, slope=1.16) != \
            pytest.approx(fluid_factor(0.1, 0.05, slope=1.40))


class TestRotation:
    @pytest.mark.parametrize("chi, expected", [(0.0, "A"), (90.0, "B")])
    def test_the_axes_come_back_unchanged(self, chi, expected):
        A, B = 0.07, -0.13
        found = chi_rotation(A, B, chi)
        assert found == pytest.approx(A if expected == "A" else B)

    def test_forty_five_degrees_is_the_scaled_poisson_reflectivity(self):
        A, B = 0.07, -0.13
        assert chi_rotation(A, B, 45.0) == pytest.approx((A + B) / np.sqrt(2))

    def test_the_projection_keeps_its_scale(self):
        """Unit-norm, so one rotation can be compared with another and with A
        itself rather than each carrying its own arbitrary gain."""
        A, B = 0.06, 0.08                      # length 0.1
        best = max(chi_rotation(A, B, c) for c in np.arange(0, 90, 0.5))
        assert best == pytest.approx(0.1, abs=1e-3)

    def test_opposite_directions_differ_only_in_sign(self):
        A, B = 0.07, -0.13
        assert chi_rotation(A, B, 30.0) == pytest.approx(
            -chi_rotation(A, B, 210.0))


class TestTheTrendsOwnRotation:
    def test_across_the_trend_reproduces_the_deviation(self):
        """The connection worth pinning: rotating perpendicular to the
        background trend *is* the distance from it, up to the trend's offset.
        Two names for one quantity, and they must not drift apart."""
        rng = np.random.default_rng(3)
        A = rng.normal(0, 0.05, 40)
        B = 1.4 * A + rng.normal(0, 0.01, 40)
        trend = background_trend(A, B)

        across = chi_rotation(A, B, trend_chi(trend.slope)["across"])
        deviation = trend.deviation(A, B)
        # Both are the same projection; they differ by the constant the trend's
        # intercept contributes, so their *variation* is identical.
        assert np.allclose(across - across.mean(),
                           deviation - deviation.mean(), atol=1e-12)

    def test_along_the_trend_is_the_direction_the_cloud_runs_in(self):
        assert trend_chi(1.0)["along"] == pytest.approx(45.0)
        assert trend_chi(0.0)["along"] == pytest.approx(0.0)

    def test_the_two_are_a_right_angle_apart(self):
        found = trend_chi(-2.3)
        assert found["across"] - found["along"] == pytest.approx(90.0)

    def test_an_unfittable_trend_gives_no_angle(self):
        found = trend_chi(np.nan)
        assert np.isnan(found["along"]) and np.isnan(found["across"])


class TestTheChiSweep:
    @staticmethod
    def cloud(chi_true=60.0, n=40, noise=0.0, seed=5):
        """Reflectors whose target depends on one direction in the A-B plane
        and on nothing else — so the sweep has a right answer to find."""
        rng = np.random.default_rng(seed)
        A = rng.normal(0, 0.05, n)
        B = rng.normal(0, 0.05, n)
        target = chi_rotation(A, B, chi_true) + rng.normal(0, noise, n)
        return A, B, target

    def test_it_finds_the_direction_the_target_actually_lies_in(self):
        A, B, target = self.cloud(chi_true=60.0)
        found = chi_sweep(A, B, target)
        assert found["best_chi"] == pytest.approx(60.0, abs=3.0)
        assert found["best_correlation"] > 0.98

    def test_it_survives_noise_on_the_target(self):
        A, B, target = self.cloud(chi_true=-30.0, noise=0.01)
        found = chi_sweep(A, B, target)
        assert found["best_chi"] == pytest.approx(-30.0, abs=8.0)

    def test_the_sign_says_which_way_the_property_runs(self):
        A, B, target = self.cloud(chi_true=60.0)
        assert chi_sweep(A, B, target)["best_correlation"] > 0
        assert chi_sweep(A, B, -target)["best_correlation"] < 0

    def test_spearman_is_the_default_and_pearson_is_offered(self):
        A, B, target = self.cloud(chi_true=20.0)
        assert chi_sweep(A, B, target)["best_chi"] == pytest.approx(
            chi_sweep(A, B, target, method="pearson")["best_chi"], abs=5.0)

    def test_a_monotonic_but_bent_relationship_still_reads(self):
        """Why Spearman: a reflection attribute against a rock property is
        monotonic far more often than it is linear."""
        A, B, straight = self.cloud(chi_true=45.0)
        bent = np.sign(straight) * np.abs(straight) ** 3
        assert chi_sweep(A, B, bent)["best_correlation"] > 0.98

    def test_too_few_reflectors_is_refused_rather_than_fitted(self):
        found = chi_sweep([0.1, 0.2], [0.0, 0.1], [1.0, 2.0])
        assert np.isnan(found["best_chi"])
        assert "four" in found["reason"]

    def test_a_constant_target_correlates_with_nothing(self):
        A, B, _ = self.cloud()
        found = chi_sweep(A, B, np.full(A.size, 0.2))
        assert found["reason"] == "the target is constant"

    def test_missing_values_are_dropped_pairwise(self):
        A, B, target = self.cloud(chi_true=60.0)
        target = target.copy()
        target[:5] = np.nan
        found = chi_sweep(A, B, target)
        assert found["n"] == A.size - 5
        assert found["best_chi"] == pytest.approx(60.0, abs=4.0)

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError, match="same length"):
            chi_sweep([0.1, 0.2], [0.1, 0.2], [1.0])


class TestAnomalyRanking:
    def test_the_score_is_in_units_of_the_wells_own_scatter(self):
        """Which is what makes it comparable between wells: a noisy hole has a
        wide cloud, and 0.02 off the trend means something different in it."""
        quiet = np.array([0.001, -0.001, 0.002, -0.002, 0.020])
        noisy = quiet * 10.0
        assert anomaly_ranking(quiet)["z"][-1] == pytest.approx(
            anomaly_ranking(noisy)["z"][-1])

    def test_the_strongest_departure_ranks_first(self):
        found = anomaly_ranking([0.001, -0.030, 0.002, 0.011])
        assert found["rank"][1] == 1
        assert found["order"][0] == 1

    def test_the_sign_is_kept(self):
        """A Class III bright trough and a Class I dim one are both anomalies
        and are not the same finding."""
        found = anomaly_ranking([0.001, -0.030, 0.002, 0.030])
        assert found["z"][1] < 0 < found["z"][3]

    def test_a_standard_deviation_would_be_inflated_by_the_anomalies(self):
        """The reason for the MAD: the events being looked for would otherwise
        raise the yardstick they are measured with."""
        values = np.array([0.001, -0.001, 0.002, -0.002, 0.001, 0.30])
        robust = anomaly_ranking(values)
        plain = anomaly_ranking(values, robust=False)
        assert abs(robust["z"][-1]) > 5 * abs(plain["z"][-1])

    def test_every_reflector_on_the_trend_scores_nothing(self):
        """No scatter to measure against, so no reflector is unusual — and no
        division by zero on the way to saying so."""
        found = anomaly_ranking(np.zeros(6))
        assert np.isnan(found["z"]).all()
        assert found["scale"] == 0.0

    def test_the_trend_is_the_zero_not_the_median_deviation(self):
        """A cloud sitting entirely on one side of its trend is a cloud of
        reflectors that are all somewhat unusual, and re-centring on their
        median would report every one of them as ordinary. The test of that is
        shift-invariance: a centred score would not notice the offset at all."""
        base = np.array([0.001, -0.002, 0.010, -0.001])
        shifted = base + 0.02

        assert not np.allclose(anomaly_ranking(base)["z"],
                               anomaly_ranking(shifted)["z"])
        assert (anomaly_ranking(shifted)["z"] > 0).all()

    def test_missing_deviations_are_unranked_rather_than_last(self):
        found = anomaly_ranking([0.001, np.nan, -0.030])
        assert found["rank"][1] == -1
        assert np.isnan(found["z"][1])
        assert found["rank"][2] == 1

    def test_labels_come_back_beside_the_scores(self):
        found = anomaly_ranking([0.001, -0.030], labels=["top", "base"])
        assert list(found["labels"][found["order"]]) == ["base", "top"]

    def test_each_group_is_scored_in_its_own_scatter(self):
        """Pooling wells and scaling them together lets the noisiest hole set
        the yardstick, and the quiet well's best event vanishes into everyone
        else's scatter."""
        quiet = [0.001, -0.001, 0.002, -0.002, 0.020]
        noisy = [0.01, -0.01, 0.02, -0.02, 0.02]
        deviation = quiet + noisy
        wells = ["A"] * 5 + ["B"] * 5

        pooled = anomaly_ranking(deviation)
        grouped = anomaly_ranking(deviation, groups=wells)

        # Scaled together, the quiet well's standout is unremarkable.
        assert abs(pooled["z"][4]) < 2
        # Scaled apart, it is the standout it actually is — and ranks first.
        assert abs(grouped["z"][4]) > 6
        assert grouped["rank"][4] == 1

    def test_the_rank_still_spans_the_groups(self):
        """Scores made separately, but 'what should I look at first' is one
        question across the wells."""
        found = anomaly_ranking([0.001, -0.001, 0.002, 0.030,
                                 0.10, -0.10, 0.20, 0.21],
                                groups=["A"] * 4 + ["B"] * 4)
        assert sorted(found["rank"]) == list(range(1, 9))

    def test_the_scale_is_reported_per_group(self):
        found = anomaly_ranking([0.001, -0.001, 0.002, 0.030,
                                 0.01, -0.01, 0.02, 0.30],
                                groups=["A"] * 4 + ["B"] * 4)
        assert set(found["scale"]) == {"A", "B"}
        assert found["scale"]["B"] > found["scale"]["A"]

    def test_a_group_with_one_event_is_unscored_rather_than_infinite(self):
        found = anomaly_ranking([0.001, -0.001, 0.002, 0.030],
                                groups=["A", "A", "A", "B"])
        assert np.isnan(found["z"][3])
        assert found["rank"][3] == -1

    def test_mismatched_groups_are_refused(self):
        with pytest.raises(ValueError, match="one entry per reflector"):
            anomaly_ranking([0.1, 0.2, 0.3], groups=["A", "B"])


class TestThePipelineCarriesThem:
    """The attributes on the wells they will actually be read from.

    The one that is easy to get wrong and impossible to see: the background
    Vp/Vs. Away from 2 the density terms in the Aki-Richards form stop
    cancelling, so a hard-coded 2.0 quietly converts a density contrast into
    shear reflectivity — and it looks entirely plausible on the way out.
    """

    @staticmethod
    def analysed(which="demo"):
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from avo_qi.analysis import reflector_analysis
        from avo_qi.ui import Settings

        if which == "demo":
            from test_app_smoke import demo_well as load
        else:
            from test_multi_well import second_well as load
        return reflector_analysis(load()[0], Settings())["table"]

    def test_every_attribute_arrives(self):
        table = self.analysed()
        assert {"rp", "rs", "fluid_factor", "ab_product",
                "vp_vs_background"} <= set(table.columns)
        assert table["rs"].notna().all()

    def test_the_intercept_is_reported_as_the_p_reflectivity(self):
        table = self.analysed()
        assert np.allclose(table["rp"], table["A_shuey"], equal_nan=True)

    def test_the_background_ratio_is_the_wells_own_not_a_constant(self):
        """The demo well's sand sits near Vp/Vs 1.6 and its shale near 2.0, so
        a constant would be visible as a column that never moves."""
        table = self.analysed()
        ratio = table["vp_vs_background"].to_numpy(float)
        assert np.isfinite(ratio).all()
        assert ratio.min() < 1.75 and ratio.max() > 1.95

    def test_using_a_constant_ratio_would_change_the_answer(self):
        """Which is why it is read per reflector. If this came out equal, the
        parameter would be decoration and the docstring a lie."""
        from avo_qi.core.avo_attributes import pseudo_shear_reflectivity

        table = self.analysed()
        fixed = pseudo_shear_reflectivity(table["A_shuey"].to_numpy(float),
                                          table["B_shuey"].to_numpy(float),
                                          vp_vs=2.0)
        moved = np.abs(fixed - table["rs"].to_numpy(float))
        assert moved.max() > 0.005

    def test_the_gas_sand_top_has_the_wells_strongest_fluid_factor(self):
        """A known answer: the demo well is shale over a Class III gas sand,
        and the fluid factor exists to find exactly that."""
        table = self.analysed()
        top = table.index[table["litho_pair"] == "shale over sand"][0]
        assert abs(table.loc[top, "fluid_factor"]) == pytest.approx(
            table["fluid_factor"].abs().max(), rel=1e-9)
        assert table.loc[top, "avo_class"] == "III"

    def test_the_product_agrees_with_its_own_definition(self):
        table = self.analysed()
        assert np.allclose(table["ab_product"],
                           table["A_shuey"] * table["B_shuey"], equal_nan=True)

    def test_attributes_can_be_turned_off(self):
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from avo_qi.analysis import reflector_analysis
        from avo_qi.ui import Settings
        from test_app_smoke import demo_well

        table = reflector_analysis(demo_well()[0], Settings(),
                                   attributes=False)["table"]
        assert "fluid_factor" not in table.columns

    def test_the_real_well_gets_them_too(self):
        table = self.analysed("real")
        assert table["fluid_factor"].notna().all()
        assert table["vp_vs_background"].between(1.4, 3.0).all()
