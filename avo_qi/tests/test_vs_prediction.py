"""Predicting a shear sonic, and being honest about how well it went.

Vs prediction is the largest modelling assumption anyone using this toolkit
will make: everything downstream — the gradient, Vp/Vs, Poisson, LMR, every
substitution — is built on it. So the tests here are less about the algebra,
which is short, than about the two things that decide whether the result can be
trusted. A prediction must never overwrite a measurement. And the ranking
between models must be measured on a real well rather than assumed, because the
intuitive answer turns out to be wrong.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from avo_qi.core.vs_prediction import (GREENBERG_CASTAGNA,  # noqa: E402
                                       TRANSFORMS, apply_vp_vs_trend,
                                       fill_missing_vs, fit_vp_vs_trend,
                                       lithology_fractions_from_vsh,
                                       mudrock_vs, polynomial_vs,
                                       prediction_quality)


@pytest.fixture(scope="module")
def real():
    """15/9-19-A, which carries a real shear sonic on every sample — so every
    prediction here can be marked against the answer."""
    from test_multi_well import second_well

    frame = second_well()[0].df
    return {c: frame[c].to_numpy(float) for c in ("DEPTH", "VP", "VS", "VSH")}


class TestTheTransforms:
    def test_there_is_one_set_of_coefficients_in_the_repository(self):
        """Not two that a test watches for drift — one, imported. The trend
        lines the Rock Physics page draws and the curve predicted here have to
        come from the same numbers, or a well is predicted against one set and
        QC'd against the other."""
        from avo_qi.core.rockphysics import _GC_COEFFS

        assert set(_GC_COEFFS) == set(GREENBERG_CASTAGNA)
        for lith, coeffs in GREENBERG_CASTAGNA.items():
            assert tuple(_GC_COEFFS[lith]) == coeffs, lith
        # The predictor evaluates the same polynomial the page plots.
        from avo_qi.core.rockphysics import greenberg_castagna

        vp = np.array([3200.0, 3800.0])
        assert polynomial_vs(vp, {"shale": 1.0}) == pytest.approx(
            greenberg_castagna(vp, {"shale": 1.0}))

    def test_only_checkable_transforms_are_registered(self):
        """A transform carrying half-remembered coefficients is worse than no
        transform: every Vp/Vs, Poisson and gradient downstream inherits them
        with no mark on them."""
        assert set(TRANSFORMS) == {"greenberg_castagna"}

    def test_a_pure_lithology_is_its_own_polynomial(self):
        vp = np.array([3000.0, 4000.0])
        expected = np.polyval(GREENBERG_CASTAGNA["shale"], vp / 1000.0) * 1000.0
        assert np.allclose(polynomial_vs(vp, {"shale": 1.0}), expected)

    def test_a_mixture_sits_between_its_end_members(self):
        vp = np.full(3, 3600.0)
        sand = polynomial_vs(vp, {"sandstone": 1.0})
        shale = polynomial_vs(vp, {"shale": 1.0})
        mixed = polynomial_vs(vp, {"sandstone": 0.5, "shale": 0.5})
        assert (np.minimum(sand, shale) <= mixed).all()
        assert (mixed <= np.maximum(sand, shale)).all()

    def test_shale_volume_drives_the_mixture(self):
        vp = np.full(3, 3600.0)
        vsh = np.array([0.0, 0.5, 1.0])
        found = polynomial_vs(vp, lithology_fractions_from_vsh(vsh))
        # Monotonic in VSH, and the ends are the pure lithologies.
        assert np.all(np.diff(found) < 0) or np.all(np.diff(found) > 0)
        assert found[0] == pytest.approx(
            polynomial_vs(vp, {"sandstone": 1.0})[0])
        assert found[2] == pytest.approx(polynomial_vs(vp, {"shale": 1.0})[2])

    def test_a_missing_shale_volume_predicts_nothing(self):
        """Rather than defaulting to clean sand, which would be a confident
        answer about a sample nobody interpreted."""
        found = polynomial_vs(np.full(2, 3600.0),
                              lithology_fractions_from_vsh([0.3, np.nan]))
        assert np.isfinite(found[0]) and np.isnan(found[1])

    def test_a_missing_vp_predicts_nothing(self):
        assert np.isnan(polynomial_vs(np.array([np.nan]))[0])

    def test_an_unknown_lithology_is_refused(self):
        with pytest.raises(ValueError, match="no coefficients"):
            polynomial_vs(np.array([3000.0]), {"granite": 1.0})

    def test_the_mudrock_line_is_castagnas(self):
        assert mudrock_vs(np.array([3000.0]))[0] == pytest.approx(
            0.8621 * 3000.0 - 1172.4)


class TestFittingTheWellsOwnTrend:
    def test_it_recovers_a_line_it_was_given(self):
        vp = np.linspace(2500.0, 4500.0, 200)
        vs = 0.62 * vp - 300.0
        found = fit_vp_vs_trend(vp, vs)
        assert apply_vp_vs_trend(vp, found["coefficients"]) == \
            pytest.approx(vs, rel=1e-6)

    def test_too_few_samples_is_refused_rather_than_fitted(self):
        found = fit_vp_vs_trend(np.arange(10.0) + 3000, np.arange(10.0) + 1500)
        assert found["coefficients"] is None
        assert "are needed" in found["reason"]

    def test_a_constant_vp_cannot_define_a_trend(self):
        found = fit_vp_vs_trend(np.full(80, 3000.0), np.linspace(1400, 1600, 80))
        assert found["coefficients"] is None
        assert "does not vary" in found["reason"]

    def test_it_ignores_samples_with_no_shear_sonic(self):
        vp = np.linspace(2500.0, 4500.0, 200)
        vs = 0.62 * vp - 300.0
        holed = vs.copy()
        holed[::3] = np.nan
        assert fit_vp_vs_trend(vp, holed)["n"] == int(np.isfinite(holed).sum())

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError, match="same length"):
            fit_vp_vs_trend(np.zeros(5), np.zeros(4))


class TestOnARealWell:
    """The ranking, measured rather than assumed.

    The intuitive answer — that a trend fitted to this well must beat a
    published transform — is wrong here, and the reason is worth keeping in
    front of anyone reading this file: a single Vp-Vs line has nowhere to put
    shale volume, and shale volume is most of what moves Vs.
    """

    @staticmethod
    def error(measured, predicted, mask=None):
        if mask is not None:
            measured = np.where(mask, measured, np.nan)
        return prediction_quality(measured, predicted)["median_relative_error"]

    def test_the_lithology_aware_transform_wins(self, real):
        mix = self.error(real["VS"], polynomial_vs(
            real["VP"], lithology_fractions_from_vsh(real["VSH"])))
        sand = self.error(real["VS"], polynomial_vs(real["VP"]))
        mudrock = self.error(real["VS"], mudrock_vs(real["VP"]))

        assert mix < 0.055                       # ~4.4% on this well
        assert mix < sand and mix < mudrock

    def test_treating_shale_as_sand_reads_too_fast(self, real):
        """Which is the direction that matters: a Vs biased high pulls Vp/Vs
        down and moves every gradient with it."""
        quality = prediction_quality(real["VS"], polynomial_vs(real["VP"]))
        assert quality["bias"] > 0.03
        mixed = prediction_quality(real["VS"], polynomial_vs(
            real["VP"], lithology_fractions_from_vsh(real["VSH"])))
        assert abs(mixed["bias"]) < 0.02

    def test_a_representative_calibration_still_loses_to_it(self, real):
        """Even given half the well at random, the fitted line cannot catch a
        transform that knows the shale volume."""
        rng = np.random.default_rng(0)
        held = rng.random(real["VP"].size) < 0.5
        fit = fit_vp_vs_trend(np.where(~held, real["VP"], np.nan),
                              np.where(~held, real["VS"], np.nan))
        own = self.error(real["VS"],
                         apply_vp_vs_trend(real["VP"], fit["coefficients"]), held)
        mix = self.error(real["VS"], polynomial_vs(
            real["VP"], lithology_fractions_from_vsh(real["VSH"])), held)
        assert mix < own

    def test_an_unrepresentative_calibration_is_much_worse(self, real):
        """The failure that matters in practice, because it is the natural
        thing to do: the shear sonic starts partway down, so you calibrate on
        what you have and extrapolate. Here the upper half is shale (VSH 0.66)
        and the lower half is sand (VSH 0.25), and the fitted line predicts
        shale velocities for sand — 11% low."""
        deep = real["DEPTH"] >= np.nanmedian(real["DEPTH"])
        fit = fit_vp_vs_trend(np.where(~deep, real["VP"], np.nan),
                              np.where(~deep, real["VS"], np.nan))
        predicted = apply_vp_vs_trend(real["VP"], fit["coefficients"])
        quality = prediction_quality(np.where(deep, real["VS"], np.nan), predicted)

        assert quality["median_relative_error"] > 0.09
        assert quality["bias"] < -0.05           # systematically slow
        # And the lithology-aware transform is unbothered by the same split.
        mix = self.error(real["VS"], polynomial_vs(
            real["VP"], lithology_fractions_from_vsh(real["VSH"])), deep)
        assert mix < 0.055


class TestAPredictionNeverOverwritesAMeasurement:
    def test_measured_samples_are_kept_untouched(self):
        measured = np.array([1500.0, np.nan, 1700.0, np.nan])
        predicted = np.array([9.0, 1600.0, 9.0, 1800.0])
        found = fill_missing_vs(measured, predicted)
        assert found["vs"] == pytest.approx([1500.0, 1600.0, 1700.0, 1800.0])
        assert list(found["is_predicted"]) == [False, True, False, True]

    def test_a_gap_the_model_cannot_fill_stays_a_gap(self):
        found = fill_missing_vs(np.array([np.nan]), np.array([np.nan]))
        assert np.isnan(found["vs"][0])
        assert not found["is_predicted"][0]

    def test_a_fully_logged_well_is_returned_unchanged(self):
        measured = np.array([1500.0, 1600.0])
        found = fill_missing_vs(measured, np.array([1.0, 2.0]))
        assert found["vs"] == pytest.approx(measured)
        assert not found["is_predicted"].any()

    def test_mismatched_shapes_are_refused(self):
        with pytest.raises(ValueError, match="same shape"):
            fill_missing_vs(np.zeros(3), np.zeros(2))


class TestScoringAPrediction:
    def test_a_perfect_prediction_scores_zero(self):
        vs = np.linspace(1400, 1900, 50)
        quality = prediction_quality(vs, vs)
        assert quality["median_absolute_error"] == pytest.approx(0.0)
        assert quality["bias"] == pytest.approx(0.0)
        assert quality["correlation"] == pytest.approx(1.0)

    def test_bias_is_signed_so_a_correctable_error_looks_different(self):
        """A prediction 5% slow everywhere can be corrected; one that scatters
        cannot, and the two must not report the same number."""
        vs = np.linspace(1400, 1900, 50)
        rng = np.random.default_rng(1)
        slow = prediction_quality(vs, vs * 0.95)
        noisy = prediction_quality(vs, vs + rng.normal(0, 70, vs.size))
        assert slow["bias"] == pytest.approx(-0.05, abs=1e-9)
        assert abs(noisy["bias"]) < 0.02
        assert noisy["median_absolute_error"] > 0

    def test_nothing_to_compare_scores_nothing(self):
        quality = prediction_quality(np.array([np.nan, np.nan]),
                                     np.array([1500.0, 1600.0]))
        assert quality["n"] == 0
        assert np.isnan(quality["median_absolute_error"])

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError, match="same length"):
            prediction_quality(np.zeros(3), np.zeros(2))


class TestTheProvenanceTravels:
    """A predicted Vs has to say so wherever it is used, not only where it was
    made. Predicting it is accurate to a few per cent on 15/9-19-A and still
    moves 42% of the AVO classes, so "which of these reflectors rest on an
    invented curve" is the question the rest of the toolkit needs answered.
    """

    @staticmethod
    def predicted(partial_above=None):
        """The real well with its shear sonic removed — all of it, or only
        above a depth — then predicted through the page."""
        import os

        from streamlit.testing.v1 import AppTest

        from avo_qi.ui import Settings
        from test_app_smoke import PAGES
        from test_multi_well import second_well

        well, raw, units = second_well()
        if partial_above is None:
            well.df = well.df.drop(columns=["VS"])
        else:
            vs = well.df["VS"].to_numpy(float).copy()
            vs[well.df["DEPTH"].to_numpy(float) < partial_above] = np.nan
            well.df["VS"] = vs

        at = AppTest.from_file(os.path.join(PAGES, "1_Load_and_QC.py"),
                               default_timeout=300)
        at.session_state["well"] = well
        at.session_state["raw_df"] = raw
        at.session_state["raw_units"] = units
        at.session_state["settings"] = Settings()
        at.run()
        next(r for r in at.radio
             if "Where Vs comes from" in r.label).set_value(
                 "greenberg_castagna").run()
        next(b for b in at.button if "Apply shear sonic" in b.label).click().run()
        return at

    def test_the_well_carries_the_mask_as_a_curve(self):
        """On the well rather than in session state, so it resamples onto the
        time grid with everything else and works for a well that is not the
        active one."""
        from avo_qi.ui import VS_PREDICTED

        at = self.predicted()
        frame = at.session_state["well"].df
        assert VS_PREDICTED in frame.columns
        assert set(np.unique(frame[VS_PREDICTED].to_numpy(float))) <= {0.0, 1.0}

    def test_it_is_not_offered_as_something_to_plot(self):
        """Provenance, not a log. The curve pickers enumerate CANONICAL."""
        from avo_qi.io.loader import CANONICAL
        from avo_qi.ui import VS_PREDICTED

        assert VS_PREDICTED not in CANONICAL

    def test_every_reflector_says_how_much_of_it_was_invented(self):
        from avo_qi.analysis import reflector_analysis

        at = self.predicted()
        table = reflector_analysis(at.session_state["well"],
                                   at.session_state["settings"])["table"]
        assert "vs_predicted" in table.columns
        # No measured Vs anywhere, so every event rests entirely on the model.
        assert (table["vs_predicted"] > 0.99).all()

    def test_a_partly_logged_well_separates_the_two(self):
        """The case the column exists for: some reflectors sit on measured
        rock and some on invented rock, in one table."""
        from avo_qi.analysis import reflector_analysis

        at = self.predicted(partial_above=3850.0)
        table = reflector_analysis(at.session_state["well"],
                                   at.session_state["settings"])["table"]
        share = table["vs_predicted"].to_numpy(float)
        assert (share > 0.99).any()          # shallow: predicted
        assert (share < 0.01).any()          # deep: measured
        # And they are the right way round.
        shallow = table.loc[table["depth"] < 3850.0, "vs_predicted"]
        deep = table.loc[table["depth"] > 3900.0, "vs_predicted"]
        assert (shallow > 0.99).all() and (deep < 0.01).all()

    def test_a_fully_logged_well_carries_no_such_column(self):
        """Absence is the honest answer where nothing was predicted, rather
        than a column of zeros implying the question was asked."""
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from avo_qi.analysis import reflector_analysis
        from avo_qi.ui import Settings
        from test_app_smoke import demo_well

        table = reflector_analysis(demo_well()[0], Settings())["table"]
        assert "vs_predicted" not in table.columns

    def test_choosing_the_file_again_removes_the_mask(self):
        """Reverting to the measured curve must not leave the provenance of a
        prediction that is no longer there."""
        from avo_qi.ui import VS_PREDICTED

        at = self.predicted(partial_above=3850.0)
        assert VS_PREDICTED in at.session_state["well"].df.columns
        next(r for r in at.radio
             if "Where Vs comes from" in r.label).set_value("file").run()
        next(b for b in at.button if "Apply shear sonic" in b.label).click().run()
        assert VS_PREDICTED not in at.session_state["well"].df.columns
