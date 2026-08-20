"""Blocking logs to the seismic's resolution — a half cycle either side.

Two adjacent log samples carry the true layer contrast only when the boundary
is a step.  These tests pin how badly that fails on a gradational boundary and
how well a half-cycle average recovers it.
"""

from __future__ import annotations

import numpy as np
import pytest

from avo_qi.core.avo import classify, reflector_avo, shuey_fit
from avo_qi.core.blocking import (
    arithmetic_average,
    backus_average,
    block_properties,
    half_cycle_samples,
)
from avo_qi.core.reflectivity import zoeppritz_rpp
from avo_qi.core.tuning import apparent_period
from avo_qi.core.wavelet import ricker

SHALE = (2400.0, 1200.0, 2.35)
GAS_SAND = (2100.0, 1300.0, 2.10)
ANGLES = np.arange(0.0, 41.0, 2.0)
DT = 0.001
BOUNDARY = 80              # first sample of the sand
INTERFACE = BOUNDARY - 1   # interface index between samples 79 and 80


def gradational_model(ramp, n=200):
    """Shale over sand with the boundary smeared over `ramp` samples."""
    vp = np.full(n, SHALE[0])
    vs = np.full(n, SHALE[1])
    rho = np.full(n, SHALE[2])
    vp[BOUNDARY:130], vs[BOUNDARY:130], rho[BOUNDARY:130] = GAS_SAND
    if ramp > 1:
        for arr, (a, b) in zip((vp, vs, rho), zip(SHALE, GAS_SAND)):
            arr[BOUNDARY - ramp:BOUNDARY] = np.linspace(a, b, ramp + 2)[1:-1]
    return vp, vs, rho


def blocked_r0(vp, vs, rho, window, guard=0, method="backus", interface=INTERFACE):
    b = block_properties(vp, vs, rho, [interface], window=window, method=method,
                         guard=guard)
    return float(zoeppritz_rpp(
        b["vp_upper"][0], b["vs_upper"][0], b["rho_upper"][0],
        b["vp_lower"][0], b["vs_lower"][0], b["rho_lower"][0], 0.0,
    ))


TRUE_R0 = float(zoeppritz_rpp(*SHALE, *GAS_SAND, 0.0))


class TestHalfCycleWindow:
    def test_window_is_half_the_apparent_period(self):
        _, w = ricker(30.0, DT)
        period = apparent_period(w, DT)
        assert half_cycle_samples(period, DT) == pytest.approx(
            round(period / 2 / DT), abs=1
        )

    def test_matches_the_tuning_thickness(self):
        """A half cycle is the same thickness at which a bed tunes."""
        from avo_qi.core.tuning import tuning_thickness_from_wavelet

        _, w = ricker(30.0, DT)
        window = half_cycle_samples(apparent_period(w, DT), DT)
        assert window * DT == pytest.approx(tuning_thickness_from_wavelet(w, DT), abs=DT)

    def test_rejects_a_bad_sample_rate(self):
        with pytest.raises(ValueError):
            half_cycle_samples(0.026, 0.0)

    def test_never_returns_a_zero_window(self):
        assert half_cycle_samples(1e-9, DT) == 1


class TestAverages:
    def test_homogeneous_window_returns_its_own_values(self):
        vp = np.full(20, 2400.0)
        vs = np.full(20, 1200.0)
        rho = np.full(20, 2.35)
        for average in (backus_average, arithmetic_average):
            got = average(vp, vs, rho)
            assert got == pytest.approx((2400.0, 1200.0, 2.35))

    def test_backus_harmonically_averages_the_moduli(self):
        vp = np.array([2400.0, 3000.0])
        vs = np.array([1200.0, 1600.0])
        rho = np.array([2.35, 2.45])
        vp_eff, _, rho_eff = backus_average(vp, vs, rho)

        m = rho * vp ** 2
        harmonic = 1.0 / np.mean(1.0 / m)
        assert rho_eff == pytest.approx(np.mean(rho))
        assert vp_eff == pytest.approx(np.sqrt(harmonic / rho_eff))
        # The harmonic mean of the moduli is below the arithmetic mean, which
        # is why Backus is the slower — and the correct — average.
        assert harmonic < np.mean(m)

    def test_backus_is_not_faster_than_the_arithmetic_average(self):
        vp = np.array([2100.0, 2400.0, 3000.0, 2200.0])
        vs = np.array([1300.0, 1200.0, 1600.0, 1250.0])
        rho = np.array([2.10, 2.35, 2.45, 2.20])
        assert backus_average(vp, vs, rho)[0] <= arithmetic_average(vp, vs, rho)[0]

    def test_a_fluid_layer_has_no_shear_modulus(self):
        vp = np.array([1500.0, 1500.0])
        vs = np.array([0.0, 0.0])
        rho = np.array([1.02, 1.02])
        assert backus_average(vp, vs, rho)[1] == 0.0

    def test_non_finite_samples_are_ignored(self):
        vp = np.array([2400.0, np.nan, 2400.0])
        vs = np.array([1200.0, 1200.0, np.nan])
        rho = np.array([2.35, 2.35, 2.35])
        assert np.isfinite(arithmetic_average(vp, vs, rho)).all()
        assert np.isfinite(backus_average(vp, vs, rho)[2])

    def test_an_empty_window_is_nan_not_a_crash(self):
        empty = np.array([])
        assert all(np.isnan(v) for v in backus_average(empty, empty, empty))
        assert all(np.isnan(v) for v in arithmetic_average(empty, empty, empty))


class TestBlockingRecoversTheContrast:
    WINDOW = 13                       # half cycle of a 30 Hz Ricker at 1 ms

    def test_a_blocky_boundary_is_already_exact(self):
        vp, vs, rho = gradational_model(1)
        assert blocked_r0(vp, vs, rho, self.WINDOW) == pytest.approx(TRUE_R0, abs=1e-6)

    @pytest.mark.parametrize("ramp", [2, 3, 5, 8])
    def test_adjacent_samples_lose_a_gradational_contrast(self, ramp):
        """The failure this module exists to fix."""
        vp, vs, rho = gradational_model(ramp)
        table = reflector_avo(None, vp, vs, rho, ANGLES, threshold=0.0)
        near = table[(table["sample"] >= BOUNDARY - ramp - 2)
                     & (table["sample"] < BOUNDARY + 6)]
        strongest = near.loc[near["R0"].abs().idxmax()]
        assert abs(strongest["R0"]) < 0.5 * abs(TRUE_R0)

    @pytest.mark.parametrize("ramp", [2, 3, 5, 8])
    def test_blocking_recovers_most_of_it_with_no_guard(self, ramp):
        vp, vs, rho = gradational_model(ramp)
        recovered = abs(blocked_r0(vp, vs, rho, self.WINDOW) / TRUE_R0)
        assert recovered > 0.65

    @pytest.mark.parametrize("ramp", [2, 3, 5, 8])
    def test_a_guard_matched_to_the_ramp_recovers_it_exactly(self, ramp):
        vp, vs, rho = gradational_model(ramp)
        recovered = abs(blocked_r0(vp, vs, rho, self.WINDOW, guard=ramp) / TRUE_R0)
        assert recovered == pytest.approx(1.0, abs=1e-6)

    def test_a_wider_guard_never_recovers_less(self):
        vp, vs, rho = gradational_model(5)
        recovered = [abs(blocked_r0(vp, vs, rho, self.WINDOW, guard=g) / TRUE_R0)
                     for g in range(0, 6)]
        assert all(b >= a - 1e-9 for a, b in zip(recovered, recovered[1:]))

    def test_the_class_survives_blocking(self):
        vp, vs, rho = gradational_model(5)
        b = block_properties(vp, vs, rho, [INTERFACE], window=self.WINDOW, guard=3)
        R = zoeppritz_rpp(b["vp_upper"][0], b["vs_upper"][0], b["rho_upper"][0],
                          b["vp_lower"][0], b["vs_lower"][0], b["rho_lower"][0], ANGLES)
        assert classify(*shuey_fit(R, ANGLES)) == "III"


class TestBlockPropertiesMechanics:
    def test_shapes_and_counts(self):
        vp, vs, rho = gradational_model(1)
        b = block_properties(vp, vs, rho, [50, INTERFACE, 150], window=13)
        for key in ("vp_upper", "vs_lower", "rho_upper"):
            assert b[key].shape == (3,)
        assert np.all(b["n_upper"] == 13)
        assert np.all(b["n_lower"] == 13)

    def test_windows_clip_at_the_log_ends(self):
        vp, vs, rho = gradational_model(1)
        b = block_properties(vp, vs, rho, [2, vp.size - 3], window=13)
        assert b["n_upper"][0] == 3            # only three samples above
        assert b["n_lower"][1] <= 13
        assert np.all(np.isfinite(b["vp_upper"]))

    def test_upper_and_lower_do_not_overlap_the_boundary(self):
        vp, vs, rho = gradational_model(1)
        b = block_properties(vp, vs, rho, [INTERFACE], window=13)
        # Clean boundary: the two sides come back as the two pure layers.
        assert b["vp_upper"][0] == pytest.approx(SHALE[0])
        assert b["vp_lower"][0] == pytest.approx(GAS_SAND[0])

    def test_guard_shrinks_nothing_but_moves_the_windows(self):
        vp, vs, rho = gradational_model(1)
        b = block_properties(vp, vs, rho, [INTERFACE], window=13, guard=4)
        assert b["n_upper"][0] == 13
        assert b["n_lower"][0] == 13

    def test_unknown_method_is_rejected(self):
        vp, vs, rho = gradational_model(1)
        with pytest.raises(ValueError):
            block_properties(vp, vs, rho, [INTERFACE], window=13, method="magic")

    def test_mean_and_backus_agree_on_a_uniform_window(self):
        vp, vs, rho = gradational_model(1)
        a = block_properties(vp, vs, rho, [INTERFACE], window=13, method="backus")
        b = block_properties(vp, vs, rho, [INTERFACE], window=13, method="mean")
        assert a["vp_upper"][0] == pytest.approx(b["vp_upper"][0])

    def test_laminations_make_the_two_averages_disagree(self):
        """Fine interbedding is where the choice of average actually matters."""
        n = 200
        vp = np.full(n, SHALE[0]); vs = np.full(n, SHALE[1]); rho = np.full(n, SHALE[2])
        lam = np.arange(BOUNDARY, 130)[::2]
        vp[lam], vs[lam], rho[lam] = GAS_SAND
        backus = block_properties(vp, vs, rho, [INTERFACE], window=13, method="backus")
        mean = block_properties(vp, vs, rho, [INTERFACE], window=13, method="mean")
        assert backus["vp_lower"][0] < mean["vp_lower"][0]
        assert backus["rho_lower"][0] == pytest.approx(mean["rho_lower"][0])
