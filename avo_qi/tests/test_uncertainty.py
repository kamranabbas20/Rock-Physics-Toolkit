"""Tests for the Monte Carlo uncertainty layer."""

from __future__ import annotations

import numpy as np
import pytest

from avo_qi.core.avo import CLASSES, shuey_fit
from avo_qi.core.petro import forward_model
from avo_qi.core.reflectivity import aki_richards_rpp
from avo_qi.core.uncertainty import (
    DEFAULT_LOG_NOISE,
    LogNoise,
    Prior,
    batched_aki_richards,
    batched_shuey_fit,
    class_probabilities,
    monte_carlo_forward,
    percentile_bands,
    perturb_logs,
)

ANGLES = np.arange(0.0, 41.0, 2.0)


def three_layer(n_each=40):
    """Shale over a gas sand over a brine sand, as logs the model can take."""
    vsh = np.concatenate([np.full(n_each, 0.85), np.full(n_each, 0.10),
                          np.full(n_each, 0.12)])
    phit = np.concatenate([np.full(n_each, 0.14), np.full(n_each, 0.23),
                           np.full(n_each, 0.26)])
    sw = np.concatenate([np.full(n_each, 1.0), np.full(n_each, 0.20),
                         np.full(n_each, 1.0)])
    return vsh, phit, sw


class TestPrior:
    def test_fixed_never_varies(self):
        rng = np.random.default_rng(0)
        assert np.all(Prior.fixed(0.36).draw(rng, 50) == 0.36)

    def test_normal_has_the_mean_and_spread_asked_for(self):
        rng = np.random.default_rng(1)
        draws = Prior.normal(20.0, 3.0).draw(rng, 40000)
        assert np.mean(draws) == pytest.approx(20.0, abs=0.1)
        assert np.std(draws) == pytest.approx(3.0, abs=0.1)

    def test_uniform_stays_between_its_ends(self):
        rng = np.random.default_rng(2)
        draws = Prior.uniform(0.6, 1.0).draw(rng, 5000)
        assert draws.min() >= 0.6 and draws.max() <= 1.0

    def test_triangular_concentrates_on_its_mode(self):
        rng = np.random.default_rng(3)
        draws = Prior.triangular(0.0, 0.9, 1.0).draw(rng, 20000)
        assert np.median(draws) > 0.6

    def test_lognormal_is_positive_and_multiplicative(self):
        rng = np.random.default_rng(4)
        draws = Prior.lognormal(20e6, 0.3).draw(rng, 20000)
        assert draws.min() > 0
        assert np.median(draws) == pytest.approx(20e6, rel=0.02)

    def test_truncation_is_applied(self):
        rng = np.random.default_rng(5)
        draws = Prior.normal(0.36, 0.20, low=0.20, high=0.50).draw(rng, 5000)
        assert draws.min() >= 0.20 and draws.max() <= 0.50

    def test_a_zero_width_normal_counts_as_fixed(self):
        assert Prior.normal(0.36, 0.0).is_fixed
        assert Prior.fixed(0.36).is_fixed
        assert not Prior.normal(0.36, 0.03).is_fixed

    def test_uniform_accepts_its_ends_the_wrong_way_round(self):
        rng = np.random.default_rng(6)
        draws = Prior.uniform(1.0, 0.6).draw(rng, 100)
        assert draws.min() >= 0.6 and draws.max() <= 1.0

    @pytest.mark.parametrize("build, match", [
        (lambda: Prior.normal(1.0, -1.0), "cannot be negative"),
        (lambda: Prior.lognormal(-1.0, 0.3), "must be positive"),
        (lambda: Prior.triangular(1.0, 0.0, 2.0), "low <= mode <= high"),
    ])
    def test_rejects_impossible_parameters(self, build, match):
        with pytest.raises(ValueError, match=match):
            build()

    def test_describes_itself_for_the_ui(self):
        assert "sd 0.03" in Prior.normal(0.36, 0.03).describe()
        assert "uniform" in Prior.uniform(0.6, 1.0).describe()
        assert "fixed" in Prior.fixed(0.36).describe()


class TestMonteCarloForward:
    def test_no_uncertainty_reproduces_the_deterministic_model_exactly(self):
        """The Monte Carlo must be the same model, not a second one."""
        vsh, phit, sw = three_layer()
        mc = monte_carlo_forward(vsh, phit, sw, n_realisations=5, log_noise={},
                                 pressure=20e6)
        want = forward_model(vsh, phit, sw, pressure=20e6)
        for curve in ("VP", "VS", "RHOB"):
            for r in range(5):
                assert np.allclose(mc.curve(curve)[r], want[curve],
                                   rtol=0, atol=0, equal_nan=True)

    def test_a_fixed_prior_is_the_same_as_passing_the_value(self):
        vsh, phit, sw = three_layer()
        mc = monte_carlo_forward(vsh, phit, sw, n_realisations=3, log_noise={},
                                 priors={"phi_c": Prior.fixed(0.32)},
                                 pressure=20e6)
        want = forward_model(vsh, phit, sw, phi_c=0.32, pressure=20e6)
        assert np.allclose(mc.VP[0], want["VP"], equal_nan=True)

    def test_the_shape_is_realisations_by_samples(self):
        vsh, phit, sw = three_layer(10)
        mc = monte_carlo_forward(vsh, phit, sw, n_realisations=17)
        assert mc.VP.shape == (17, 30)
        assert mc.n_realisations == 17

    def test_the_same_seed_gives_the_same_answer(self):
        vsh, phit, sw = three_layer(10)
        a = monte_carlo_forward(vsh, phit, sw, n_realisations=20, seed=42)
        b = monte_carlo_forward(vsh, phit, sw, n_realisations=20, seed=42)
        assert np.allclose(a.VP, b.VP, equal_nan=True)

    def test_a_different_seed_gives_a_different_answer(self):
        vsh, phit, sw = three_layer(10)
        a = monte_carlo_forward(vsh, phit, sw, n_realisations=20, seed=1)
        b = monte_carlo_forward(vsh, phit, sw, n_realisations=20, seed=2)
        assert not np.allclose(a.VP, b.VP, equal_nan=True)

    def test_more_uncertain_inputs_give_a_wider_band(self):
        vsh, phit, sw = three_layer()
        tight = monte_carlo_forward(vsh, phit, sw, n_realisations=300, seed=0,
                                    log_noise={"PHIT": LogNoise(0.01)},
                                    pressure=20e6)
        loose = monte_carlo_forward(vsh, phit, sw, n_realisations=300, seed=0,
                                    log_noise={"PHIT": LogNoise(0.05)},
                                    pressure=20e6)
        assert np.nanmedian(loose.spread("VP")) > 2 * np.nanmedian(tight.spread("VP"))

    def test_a_parameter_prior_widens_the_band_too(self):
        vsh, phit, sw = three_layer()
        without = monte_carlo_forward(vsh, phit, sw, n_realisations=300, seed=0,
                                      log_noise={}, pressure=20e6)
        with_prior = monte_carlo_forward(
            vsh, phit, sw, n_realisations=300, seed=0, log_noise={},
            priors={"pressure": Prior.normal(20e6, 6e6, low=1e6)})
        assert np.nanmedian(without.spread("VP")) == pytest.approx(0.0, abs=1e-9)
        assert np.nanmedian(with_prior.spread("VP")) > 50.0

    def test_a_parameter_is_drawn_once_per_realisation_not_per_sample(self):
        """A critical porosity belongs to the rock type, not to the sample."""
        vsh, phit, sw = three_layer(10)
        mc = monte_carlo_forward(vsh, phit, sw, n_realisations=25, log_noise={},
                                 priors={"phi_c": Prior.normal(0.36, 0.03)})
        assert mc.draws["phi_c"].shape == (25,)

    def test_log_noise_is_drawn_per_sample(self):
        vsh, phit, sw = three_layer(10)
        mc = monte_carlo_forward(vsh, phit, sw, n_realisations=25,
                                 log_noise={"VSH": LogNoise(0.05)})
        assert mc.draws["VSH"].shape == (25, 30)
        # Neighbouring samples are not obliged to err the same way.
        assert not np.allclose(mc.draws["VSH"][0, 0], mc.draws["VSH"][0, 1])

    def test_perturbed_logs_stay_inside_their_physical_range(self):
        vsh = np.full(50, 0.98)
        phit, sw = np.full(50, 0.30), np.full(50, 0.99)
        mc = monte_carlo_forward(vsh, phit, sw, n_realisations=200, seed=0,
                                 log_noise={"VSH": LogNoise(0.2),
                                            "SW": LogNoise(0.2)})
        assert mc.draws["VSH"].max() <= 1.0 and mc.draws["VSH"].min() >= 0.0
        assert mc.draws["SW"].max() <= 1.0

    def test_correlated_noise_narrows_a_shaly_tight_relationship(self):
        """Anti-correlating VSH and porosity is a claim about the rock, and
        it changes the answer — which is why it is never assumed by default."""
        vsh, phit, sw = three_layer()
        independent = monte_carlo_forward(
            vsh, phit, sw, n_realisations=400, seed=0, pressure=20e6,
            log_noise={"VSH": LogNoise(0.08), "PHIT": LogNoise(0.04)})
        correlated = monte_carlo_forward(
            vsh, phit, sw, n_realisations=400, seed=0, pressure=20e6,
            log_noise={"VSH": LogNoise(0.08), "PHIT": LogNoise(0.04)},
            correlation={("VSH", "PHIT"): -0.9})
        a = np.nanmedian(independent.spread("VP"))
        b = np.nanmedian(correlated.spread("VP"))
        assert a != pytest.approx(b, rel=0.01)

    def test_defaults_are_stated_rather_than_silent(self):
        assert set(DEFAULT_LOG_NOISE) == {"VSH", "PHIT", "SW"}
        # Saturation from resistivity is the least certain of the three.
        assert DEFAULT_LOG_NOISE["SW"].sigma > DEFAULT_LOG_NOISE["VSH"].sigma
        assert DEFAULT_LOG_NOISE["VSH"].sigma > DEFAULT_LOG_NOISE["PHIT"].sigma

    def test_rejects_a_prior_on_something_that_is_not_a_parameter(self):
        vsh, phit, sw = three_layer(5)
        with pytest.raises(ValueError, match="no prior can be put on"):
            monte_carlo_forward(vsh, phit, sw, priors={"porosity": Prior.fixed(1)})

    def test_rejects_mismatched_logs(self):
        with pytest.raises(ValueError, match="same length"):
            monte_carlo_forward([0.1, 0.2], [0.2], [1.0])

    def test_rejects_a_pointless_number_of_realisations(self):
        vsh, phit, sw = three_layer(5)
        with pytest.raises(ValueError, match="at least 1"):
            monte_carlo_forward(vsh, phit, sw, n_realisations=0)

    def test_rejects_inconsistent_correlations(self):
        vsh, phit, sw = three_layer(5)
        with pytest.raises(ValueError, match="not positive definite"):
            monte_carlo_forward(
                vsh, phit, sw, n_realisations=5,
                log_noise={"VSH": LogNoise(0.05), "PHIT": LogNoise(0.02),
                           "SW": LogNoise(0.1)},
                correlation={("VSH", "PHIT"): 0.99, ("PHIT", "SW"): 0.99,
                             ("VSH", "SW"): -0.99})

    def test_rejects_a_correlation_naming_a_log_with_no_noise(self):
        vsh, phit, sw = three_layer(5)
        with pytest.raises(ValueError, match="must both be among"):
            monte_carlo_forward(vsh, phit, sw, n_realisations=5,
                                log_noise={"VSH": LogNoise(0.05)},
                                correlation={("VSH", "PHIT"): -0.5})

    def test_an_unknown_curve_is_named_rather_than_guessed(self):
        vsh, phit, sw = three_layer(5)
        mc = monte_carlo_forward(vsh, phit, sw, n_realisations=2)
        with pytest.raises(ValueError, match="have VP, VS and RHOB"):
            mc.curve("AI")


class TestPercentileBands:
    def test_the_bands_come_back_in_order(self):
        rng = np.random.default_rng(0)
        values = rng.normal(3000.0, 100.0, (500, 20))
        bands = percentile_bands(values)
        assert np.all(bands[10.0] < bands[50.0])
        assert np.all(bands[50.0] < bands[90.0])

    def test_a_band_of_identical_realisations_has_no_width(self):
        values = np.full((50, 8), 2500.0)
        bands = percentile_bands(values)
        assert np.allclose(bands[90.0] - bands[10.0], 0.0)

    def test_a_sample_no_realisation_could_model_stays_nan(self):
        values = np.full((10, 3), 2500.0)
        values[:, 1] = np.nan
        bands = percentile_bands(values)
        assert np.isnan(bands[50.0][1])
        assert np.isfinite(bands[50.0][[0, 2]]).all()

    def test_partly_failed_samples_use_the_realisations_that_worked(self):
        values = np.full((10, 2), 2500.0)
        values[:4, 0] = np.nan
        bands = percentile_bands(values, (50,))
        assert bands[50.0][0] == pytest.approx(2500.0)

    def test_rejects_a_one_dimensional_input(self):
        with pytest.raises(ValueError, match="n_realisations, n_samples"):
            percentile_bands(np.zeros(10))


class TestBatchedPhysics:
    def test_batched_aki_richards_matches_the_scalar_one(self):
        rng = np.random.default_rng(0)
        n = 25
        args = [rng.uniform(2000, 4200, n), rng.uniform(900, 2400, n),
                rng.uniform(2.0, 2.7, n), rng.uniform(2000, 4200, n),
                rng.uniform(900, 2400, n), rng.uniform(2.0, 2.7, n)]
        got = batched_aki_richards(*args, ANGLES)
        want = np.array([aki_richards_rpp(*[a[i] for a in args], ANGLES)
                         for i in range(n)])
        assert np.allclose(got, want, rtol=0, atol=1e-15)

    def test_batched_aki_richards_broadcasts_over_extra_axes(self):
        shape = (7, 4)
        ones = np.ones(shape)
        out = batched_aki_richards(2500 * ones, 1200 * ones, 2.3 * ones,
                                   2800 * ones, 1500 * ones, 2.4 * ones, ANGLES)
        assert out.shape == shape + ANGLES.shape

    def test_batched_shuey_matches_the_tested_fit(self):
        rng = np.random.default_rng(1)
        rc = rng.normal(0.0, 0.1, (80, ANGLES.size))
        A1, B1 = shuey_fit(rc, ANGLES)
        A2, B2 = batched_shuey_fit(rc, ANGLES)
        assert np.allclose(A1, A2, rtol=0, atol=1e-14)
        assert np.allclose(B1, B2, rtol=0, atol=1e-14)

    def test_it_matches_with_angles_blanked_past_critical(self):
        """The case that matters: masking is why the slow path was slow."""
        rng = np.random.default_rng(2)
        rc = rng.normal(0.0, 0.1, (60, ANGLES.size))
        rc[::3, -6:] = np.nan
        A1, B1 = shuey_fit(rc, ANGLES)
        A2, B2 = batched_shuey_fit(rc, ANGLES)
        assert np.allclose(A1, A2, rtol=0, atol=1e-13)
        assert np.allclose(B1, B2, rtol=0, atol=1e-13)

    def test_a_row_with_too_few_angles_left_is_undetermined_not_invented(self):
        rc = np.full((3, ANGLES.size), np.nan)
        rc[1, :1] = 0.1                      # one angle cannot define a line
        rc[2, :4] = 0.1
        A, B = batched_shuey_fit(rc, ANGLES)
        assert np.isnan(A[0]) and np.isnan(A[1])
        assert np.isfinite(A[2])

    def test_it_recovers_a_line_it_was_given(self):
        A_true, B_true = 0.08, -0.19
        rc = A_true + B_true * np.sin(np.radians(ANGLES)) ** 2
        A, B = batched_shuey_fit(rc[np.newaxis, :], ANGLES)
        assert float(A[0]) == pytest.approx(A_true)
        assert float(B[0]) == pytest.approx(B_true)

    def test_rejects_an_angle_count_that_does_not_match(self):
        with pytest.raises(ValueError, match="one entry per"):
            batched_shuey_fit(np.zeros((2, 5)), ANGLES)


class TestClassProbabilities:
    @staticmethod
    def _run(sigma_scale=1.0, n=300, seed=0):
        vsh, phit, sw = three_layer()
        noise = {name: LogNoise(spec.sigma * sigma_scale)
                 for name, spec in DEFAULT_LOG_NOISE.items()}
        mc = monte_carlo_forward(vsh, phit, sw, n_realisations=n, seed=seed,
                                 log_noise=noise, pressure=20e6)
        return class_probabilities(mc.VP, mc.VS, mc.RHOB, samples=[39, 79],
                                   angles=ANGLES)

    def test_the_probabilities_are_a_distribution(self):
        table = self._run()
        total = table[CLASSES].sum(axis=1)
        assert np.allclose(total, 1.0)

    def test_it_reports_one_row_per_interface(self):
        table = self._run()
        assert list(table["sample"]) == [39, 79]

    def test_the_modal_class_carries_its_own_probability(self):
        table = self._run()
        for _, row in table.iterrows():
            assert row["confidence"] == pytest.approx(row[row["modal_class"]])
            assert row["confidence"] == pytest.approx(max(row[c] for c in CLASSES))

    def test_certainty_collapses_to_a_single_class(self):
        """With no uncertainty there is nothing to be uncertain about."""
        vsh, phit, sw = three_layer()
        mc = monte_carlo_forward(vsh, phit, sw, n_realisations=20, log_noise={},
                                 pressure=20e6)
        table = class_probabilities(mc.VP, mc.VS, mc.RHOB, samples=[39],
                                    angles=ANGLES)
        assert table.loc[0, "confidence"] == 1.0

    def test_more_uncertainty_makes_the_class_less_certain(self):
        """The headline: a label that looks like a fact can be a coin toss."""
        confident = self._run(sigma_scale=0.2).loc[0, "confidence"]
        vague = self._run(sigma_scale=3.0).loc[0, "confidence"]
        assert vague < confident

    def test_it_reports_the_spread_of_the_intercept_and_gradient(self):
        table = self._run()
        for name in ("A", "B"):
            assert table[f"{name}_P10"].iloc[0] <= table[f"{name}_P50"].iloc[0]
            assert table[f"{name}_P50"].iloc[0] <= table[f"{name}_P90"].iloc[0]

    def test_no_interfaces_gives_an_empty_table_not_an_error(self):
        vsh, phit, sw = three_layer(5)
        mc = monte_carlo_forward(vsh, phit, sw, n_realisations=5)
        table = class_probabilities(mc.VP, mc.VS, mc.RHOB, samples=[],
                                    angles=ANGLES)
        assert table.empty
        assert "modal_class" in table.columns

    def test_an_interface_needs_a_sample_below_it(self):
        vsh, phit, sw = three_layer(5)
        mc = monte_carlo_forward(vsh, phit, sw, n_realisations=5)
        with pytest.raises(ValueError, match="n_samples - 2"):
            class_probabilities(mc.VP, mc.VS, mc.RHOB,
                                samples=[mc.VP.shape[1] - 1], angles=ANGLES)

    def test_post_critical_masking_changes_the_answer(self):
        """It was a real bug once, so it stays a tested choice."""
        vsh, phit, sw = three_layer()
        mc = monte_carlo_forward(vsh, phit, sw, n_realisations=200, seed=0,
                                 pressure=20e6)
        masked = class_probabilities(mc.VP, mc.VS, mc.RHOB, samples=[39],
                                     angles=np.arange(0.0, 61.0, 2.0))
        raw = class_probabilities(mc.VP, mc.VS, mc.RHOB, samples=[39],
                                  angles=np.arange(0.0, 61.0, 2.0),
                                  mask_post_critical=False)
        assert not np.allclose(masked[CLASSES].to_numpy(),
                               raw[CLASSES].to_numpy())


class TestPerturbLogs:
    LOGS = (np.full(30, 2830.0), np.full(30, 1499.0), np.full(30, 2.244))

    def test_the_shape_is_realisations_by_samples(self):
        out = perturb_logs(*self.LOGS, n_realisations=40)
        assert all(out[c].shape == (40, 30) for c in ("VP", "VS", "RHOB"))

    def test_zero_uncertainty_returns_the_logs_untouched(self):
        out = perturb_logs(*self.LOGS, n_realisations=5, vp_pct=0.0,
                           vs_pct=0.0, rho_pct=0.0)
        for curve, base in zip(("VP", "VS", "RHOB"), self.LOGS):
            assert np.allclose(out[curve], base)

    def test_the_spread_is_the_percentage_asked_for(self):
        out = perturb_logs(*self.LOGS, n_realisations=20000, seed=0, vp_pct=2.0)
        assert np.std(out["VP"]) / 2830.0 * 100.0 == pytest.approx(2.0, abs=0.1)

    def test_shear_is_noisier_than_compressional_by_default(self):
        out = perturb_logs(*self.LOGS, n_realisations=4000, seed=0)
        relative = {c: np.std(out[c]) / np.mean(out[c]) for c in ("VP", "VS")}
        assert relative["VS"] > 1.5 * relative["VP"]

    def test_the_same_seed_gives_the_same_logs(self):
        a = perturb_logs(*self.LOGS, n_realisations=10, seed=7)
        b = perturb_logs(*self.LOGS, n_realisations=10, seed=7)
        assert np.allclose(a["VP"], b["VP"])

    def test_a_velocity_never_goes_negative(self):
        out = perturb_logs(*self.LOGS, n_realisations=5000, seed=0, vp_pct=80.0)
        assert out["VP"].min() > 0

    def test_correlated_tool_error_is_supported(self):
        out = perturb_logs(*self.LOGS, n_realisations=3000, seed=0,
                           correlation={("VP", "VS"): 0.9})
        r = np.corrcoef(out["VP"].ravel(), out["VS"].ravel())[0, 1]
        assert r > 0.8

    def test_it_feeds_class_probabilities(self):
        """The page-4 path end to end: measured logs in, class odds out."""
        vp = np.concatenate([np.full(20, 2400.0), np.full(20, 2100.0)])
        vs = np.concatenate([np.full(20, 1200.0), np.full(20, 1300.0)])
        rho = np.concatenate([np.full(20, 2.35), np.full(20, 2.10)])
        logs = perturb_logs(vp, vs, rho, n_realisations=300, seed=0)
        table = class_probabilities(logs["VP"], logs["VS"], logs["RHOB"],
                                    samples=[19], angles=ANGLES)
        assert table.loc[0, "modal_class"] == "III"
        assert np.isclose(table[CLASSES].sum(axis=1).iloc[0], 1.0)

    def test_a_noisier_tool_makes_the_class_less_certain(self):
        vp = np.concatenate([np.full(20, 2400.0), np.full(20, 2100.0)])
        vs = np.concatenate([np.full(20, 1200.0), np.full(20, 1300.0)])
        rho = np.concatenate([np.full(20, 2.35), np.full(20, 2.10)])

        def confidence(pct):
            logs = perturb_logs(vp, vs, rho, n_realisations=400, seed=0,
                                vp_pct=pct, vs_pct=2 * pct, rho_pct=pct)
            return class_probabilities(logs["VP"], logs["VS"], logs["RHOB"],
                                       samples=[19], angles=ANGLES
                                       ).loc[0, "confidence"]
        assert confidence(12.0) < confidence(0.5)

    def test_rejects_mismatched_logs(self):
        with pytest.raises(ValueError, match="same length"):
            perturb_logs([2500.0, 2600.0], [1200.0], [2.3])

    def test_rejects_a_negative_percentage(self):
        with pytest.raises(ValueError, match="cannot be negative"):
            perturb_logs(*self.LOGS, vp_pct=-1.0)
