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
    blocked_reflectivity,
    half_cycle_samples,
    lobe_windows,
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


class TestBlockedAndAdjacentAreDifferentInterfaces:
    """Why a detail panel must plot the layers its fit was made on.

    A reflector fitted on half-cycle blocked layers cannot be shown over the
    adjacent-sample coefficient: on a gradational boundary the adjacent pair
    carries only part of the contrast, so the two describe different
    interfaces and the fit looks badly wrong when it is not.
    """

    @staticmethod
    def _demo_grid():
        import os

        from avo_qi.io.loader import (
            depth_to_twt,
            read_well,
            resample_to_time,
            standardise,
        )

        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        df, units = read_well(os.path.join(here, "sample_data", "demo_well.las"))
        well = standardise(df, units=units, name="DEMO-1")
        frame = well.complete().reset_index(drop=True)
        twt = depth_to_twt(frame["DEPTH"].to_numpy(float),
                           frame["VP"].to_numpy(float), t0=1.6)
        grid = resample_to_time(frame, twt, dt=0.001)
        return (grid["VP"].to_numpy(float), grid["VS"].to_numpy(float),
                grid["RHOB"].to_numpy(float))

    def test_the_fit_follows_the_blocked_layers_not_the_adjacent_pair(self):
        from avo_qi.core.avo import reflector_avo
        from avo_qi.core.reflectivity import reflectivity_series
        from avo_qi.core.tuning import apparent_period
        from avo_qi.core.wavelet import ricker

        vp, vs, rho = self._demo_grid()
        angles = np.arange(0.0, 41.0, 2.0)
        _, wavelet = ricker(30.0, 0.001)
        window = half_cycle_samples(apparent_period(wavelet, 0.001), 0.001)

        rc = reflectivity_series(vp, vs, rho, angles, method="zoeppritz")
        first = reflector_avo(rc, vp, vs, rho, angles, threshold=0.01)
        samples = first["sample"].to_numpy()

        blocked = blocked_reflectivity(vp, vs, rho, samples, angles,
                                       window=window, guard=2)
        rc_fit = np.zeros_like(rc)
        rc_fit[samples] = blocked["rc"]
        fitted = reflector_avo(rc_fit, vp, vs, rho, angles, samples=samples,
                               mask_post_critical=False)
        intercept = fitted["A_shuey"].to_numpy(float)

        blocked_zero = np.asarray(blocked["rc"])[:, 0]
        adjacent_zero = rc[samples, 0]

        # The intercept is the blocked normal-incidence coefficient, to within
        # the two-term Shuey approximation over 0-40 degrees.
        assert np.nanmax(np.abs(blocked_zero - intercept)) < 0.01
        # The adjacent-sample coefficient is a different number entirely — this
        # well shows a quarter of a unit of Rpp between them.
        assert np.nanmax(np.abs(adjacent_zero - intercept)) > 0.2

    def test_blocking_recovers_contrast_the_adjacent_pair_misses(self):
        """The gradational case, stated directly: a ramp spread over several
        samples has no single interface carrying the full step."""
        n = 400
        vp = np.full(n, 2400.0)
        vs = np.full(n, 1200.0)
        rho = np.full(n, 2.35)
        ramp = np.linspace(0.0, 1.0, 21)
        vp[190:211] = 2400.0 + 600.0 * ramp
        vs[190:211] = 1200.0 + 300.0 * ramp
        rho[190:211] = 2.35 + 0.15 * ramp
        vp[211:], vs[211:], rho[211:] = 3000.0, 1500.0, 2.50

        angles = np.array([0.0])
        blocked = blocked_reflectivity(vp, vs, rho, [200], angles, window=13,
                                       guard=2)
        from avo_qi.core.reflectivity import reflectivity_series

        rc = reflectivity_series(vp, vs, rho, angles, method="zoeppritz")
        adjacent = abs(float(rc[200, 0]))
        blocked_r = abs(float(np.asarray(blocked["rc"])[0, 0]))
        assert blocked_r > 5 * adjacent


class TestLobeWindows:
    """The window is measured on the trace, not assumed from the wavelet.

    A reflector shows up as a lobe running zero crossing to zero crossing.
    Its upper half belongs to the layer above and its lower half to the layer
    below, so halving it at the extremum gives the two layers the seismic
    actually resolves there.
    """

    @staticmethod
    def _isolated_lobe(freq=30.0, dt=0.001, coefficient=-0.2, n=400):
        from avo_qi.core.wavelet import ricker

        _, wavelet = ricker(freq, dt)
        rc = np.zeros(n)
        rc[n // 2] = coefficient
        return np.convolve(rc, wavelet, mode="same")

    def test_the_bounds_are_the_lobe_s_own_zero_crossings(self):
        """A Ricker's central lobe crosses zero at ±sqrt(2)/(2 pi f), so the
        half-lobe found here has to match that, not a nominal half period."""
        dt, freq = 0.001, 30.0
        trace = self._isolated_lobe(freq, dt)
        found = lobe_windows(trace, [200], polarity=[-1])
        upper = found["upper_stop"][0] - found["upper_start"][0]
        lower = found["lower_stop"][0] - found["lower_start"][0]
        analytic = np.sqrt(2) / (2 * np.pi * freq) / dt
        assert abs(upper - analytic) <= 1.0
        assert abs(lower - analytic) <= 1.0

    def test_a_reflector_with_no_extremum_of_its_own_gets_no_lobe(self):
        """A reflector buried in a neighbour's lobe has nothing of its own to
        halve.  ``trace_extrema`` falls its index back to the interface sample,
        which sits on the neighbour's flank; splitting there cuts that lobe at
        an arbitrary point and hands back two windows of wildly different
        length — halves of nothing.  Those must be reported unresolved so the
        caller falls back to the fixed window.
        """
        from avo_qi.core.synthetic import trace_extrema

        # One clean trough, 10 to 30, with its minimum at 20.
        trace = np.zeros(60)
        trace[10:31] = -np.sin(np.linspace(0, np.pi, 21))

        # Reflector A *is* that trough; reflector B sits on its upper flank at
        # 14 with no turning point of its own anywhere near.
        samples = np.array([20, 14])
        polarity = np.array([-1.0, -1.0])
        found = trace_extrema(trace, samples, half_window=3, polarity=polarity)
        assert found["is_extremum"].tolist() == [True, False]

        blind = lobe_windows(trace, found["index"], polarity=polarity)
        assert blind["resolved"].tolist() == [True, True]
        # What the flank split actually produces: nothing like two halves.
        assert blind["upper_stop"][1] - blind["upper_start"][1] == 4
        assert blind["lower_stop"][1] - blind["lower_start"][1] == 17

        told = lobe_windows(trace, found["index"], polarity=polarity,
                            is_extremum=found["is_extremum"])
        assert told["resolved"].tolist() == [True, False]
        # The genuine one is untouched by the extra argument.
        for key in ("upper_start", "upper_stop", "lower_start", "lower_stop"):
            assert told[key][0] == blind[key][0]

    def test_an_unresolved_reflector_falls_back_to_the_fixed_window(self):
        """The fallback is what makes rejecting it safe: the reflector keeps an
        answer, it just stops claiming a measured window."""
        from avo_qi.core.synthetic import trace_extrema

        trace = np.zeros(60)
        trace[10:31] = -np.sin(np.linspace(0, np.pi, 21))
        samples = np.array([20, 14])
        polarity = np.array([-1.0, -1.0])
        found = trace_extrema(trace, samples, half_window=3, polarity=polarity)
        bounds = lobe_windows(trace, found["index"], polarity=polarity,
                              is_extremum=found["is_extremum"])

        vp = np.linspace(2000.0, 3000.0, 60)
        vs = vp / 2.0
        rho = np.full(60, 2.3)
        blocked = block_properties(vp, vs, rho, samples, window=5,
                                   method="mean", bounds=bounds)
        assert blocked["from_lobe"].tolist() == [True, False]
        # The fallback window is the fixed one, so it is symmetric and finite.
        assert blocked["n_upper"][1] == 5 and blocked["n_lower"][1] == 5
        assert np.isfinite(blocked["vp_upper"][1])
        assert np.isfinite(blocked["vp_lower"][1])

    def test_the_flag_has_to_match_the_reflector_count(self):
        with pytest.raises(ValueError, match="one entry per reflector"):
            lobe_windows(np.zeros(20), [5, 10], is_extremum=[True])

    def test_a_higher_frequency_gives_a_narrower_window(self):
        widths = []
        for freq in (20.0, 30.0, 50.0):
            found = lobe_windows(self._isolated_lobe(freq), [200], polarity=[-1])
            widths.append(found["upper_stop"][0] - found["upper_start"][0])
        assert widths[0] > widths[1] > widths[2]

    def test_the_halves_meet_at_the_extremum(self):
        found = lobe_windows(self._isolated_lobe(), [200], polarity=[-1])
        assert found["upper_stop"][0] - 1 == found["lower_start"][0] == 200

    def test_it_is_narrower_than_the_fixed_half_cycle_it_replaces(self):
        """Halving the lobe gives about a quarter period a side, against the
        half period the fixed window used. Contrasts sharpen as a result."""
        from avo_qi.core.tuning import apparent_period
        from avo_qi.core.wavelet import ricker

        dt = 0.001
        _, wavelet = ricker(30.0, dt)
        fixed = half_cycle_samples(apparent_period(wavelet, dt), dt)
        found = lobe_windows(self._isolated_lobe(30.0, dt), [200], polarity=[-1])
        assert found["upper_stop"][0] - found["upper_start"][0] < fixed

    def test_a_peak_works_the_same_way_as_a_trough(self):
        trace = self._isolated_lobe(coefficient=+0.2)
        found = lobe_windows(trace, [200], polarity=[+1])
        assert found["resolved"][0]
        assert found["upper_start"][0] < 200 < found["lower_stop"][0] - 1

    def test_a_lobe_too_thin_to_halve_is_reported_not_guessed(self):
        """Both windows would collapse onto the extremum and the contrast
        would come out as exactly zero — a wrong answer, not a narrow one."""
        thin = np.concatenate([np.zeros(20), [0.3], np.zeros(29)])
        assert not lobe_windows(thin, [20], polarity=[1])["resolved"][0]

    def test_a_flat_trace_has_no_lobe(self):
        assert not lobe_windows(np.zeros(50), [25], polarity=[1])["resolved"][0]

    def test_a_lobe_of_the_wrong_sign_is_not_claimed(self):
        trace = np.concatenate([np.zeros(20), [0.1, 0.3, 0.1], np.zeros(27)])
        assert not lobe_windows(trace, [21], polarity=[-1])["resolved"][0]

    def test_it_gives_up_rather_than_swallowing_the_trace(self):
        ramp = np.linspace(0.1, 1.0, 50)
        assert not lobe_windows(ramp, [25], polarity=[1],
                                max_half_width=5)["resolved"][0]

    def test_unresolved_reflectors_fall_back_to_the_fixed_window(self):
        """A lobe that cannot be found costs that reflector its measured
        window, not its answer."""
        n = 200
        vp = np.concatenate([np.full(100, 2400.0), np.full(100, 3000.0)])
        vs = np.concatenate([np.full(100, 1200.0), np.full(100, 1500.0)])
        rho = np.concatenate([np.full(100, 2.35), np.full(100, 2.50)])
        bounds = lobe_windows(np.zeros(n), [99], polarity=[1])   # no lobe
        blocked = block_properties(vp, vs, rho, [99], window=10, bounds=bounds)
        assert not blocked["from_lobe"][0]
        assert blocked["vp_upper"][0] == pytest.approx(2400.0)
        assert blocked["vp_lower"][0] == pytest.approx(3000.0)

    def test_explicit_bounds_average_exactly_those_samples(self):
        vp = np.arange(100, dtype=float) * 10.0 + 1000.0
        vs = vp / 2.0
        rho = np.full(100, 2.4)
        bounds = {"upper_start": np.array([10]), "upper_stop": np.array([20]),
                  "lower_start": np.array([20]), "lower_stop": np.array([30]),
                  "resolved": np.array([True])}
        blocked = block_properties(vp, vs, rho, [19], window=5, method="mean",
                                   bounds=bounds)
        assert blocked["from_lobe"][0]
        assert blocked["vp_upper"][0] == pytest.approx(vp[10:20].mean())
        assert blocked["vp_lower"][0] == pytest.approx(vp[20:30].mean())
        assert blocked["n_upper"][0] == 10 and blocked["n_lower"][0] == 10
