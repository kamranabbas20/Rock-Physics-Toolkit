"""Acceptance tests for the AVO & QI core (SPEC.md section 6).

These are regression targets for behaviour validated in the prototype.  They
exercise ``core/`` only: no Streamlit, no plotting.
"""

from __future__ import annotations

import numpy as np
import pytest

from avo_qi.core.attributes import (
    acoustic_impedance,
    eei,
    lambda_rho,
    mu_rho,
    poisson,
    shear_impedance,
    vpvs,
)
from avo_qi.core.avo import (
    aki_richards_fit,
    background_trend,
    classify,
    reflector_avo,
    shuey_fit,
)
from avo_qi.core.reflectivity import (
    aki_richards_rpp,
    critical_angle,
    reflectivity_series,
    zoeppritz_rpp,
)
from avo_qi.core.synthetic import (
    _local_extrema,
    angle_stack,
    build_gather,
    convolve_series,
    full_stack,
    trace_extrema,
)
from avo_qi.core.wavelet import (
    bandpass_ormsby,
    dominant_frequency,
    is_zero_phase,
    load_wavelet,
    ricker,
)

# The prototype-validated soft gas-sand interface: shale over gas sand.
SHALE = (2400.0, 1200.0, 2.35)
GAS_SAND = (2100.0, 1300.0, 2.10)
SOFT_INTERFACE = SHALE + GAS_SAND


# ---------------------------------------------------------------- test 1 ----
class TestZoeppritzVsAkiRichards:
    """Acceptance test 1: the two reflectivity models must agree at small angle."""

    def test_agreement_within_tolerance_to_30_degrees(self):
        angles = np.arange(0.0, 30.5, 0.5)
        z = zoeppritz_rpp(*SOFT_INTERFACE, angles)
        a = aki_richards_rpp(*SOFT_INTERFACE, angles)
        assert np.max(np.abs(z - a)) < 0.006

    def test_both_negative_and_brightening_to_40_degrees(self):
        angles = np.arange(0.0, 41.0, 1.0)
        z = zoeppritz_rpp(*SOFT_INTERFACE, angles)
        a = aki_richards_rpp(*SOFT_INTERFACE, angles)
        # A Class III response: negative at zero offset and monotonically
        # more negative with angle.
        assert np.all(z < 0) and np.all(a < 0)
        assert np.all(np.diff(z) < 0)
        assert np.all(np.diff(a) < 0)

    def test_models_diverge_modestly_by_40_degrees(self):
        z = zoeppritz_rpp(*SOFT_INTERFACE, 40.0)
        a = aki_richards_rpp(*SOFT_INTERFACE, 40.0)
        divergence = abs(float(z) - float(a))
        # Larger than the near-angle agreement, but still a modest divergence.
        assert 0.006 < divergence < 0.05

    def test_vectorised_over_angles(self):
        angles = np.linspace(0, 40, 17)
        assert zoeppritz_rpp(*SOFT_INTERFACE, angles).shape == angles.shape
        assert aki_richards_rpp(*SOFT_INTERFACE, angles).shape == angles.shape


# ---------------------------------------------------------------- test 2 ----
class TestNormalIncidence:
    """Acceptance test 2: at zero offset, Rpp is the acoustic-impedance
    reflectivity."""

    @pytest.mark.parametrize(
        "interface",
        [
            SOFT_INTERFACE,
            (2400.0, 1200.0, 2.35, 3000.0, 1800.0, 2.45),   # hard streak
            (3300.0, 2000.0, 2.50, 2600.0, 1400.0, 2.20),   # Class IV geometry
        ],
    )
    def test_zoeppritz_at_zero_degrees(self, interface):
        vp1, vs1, rho1, vp2, vs2, rho2 = interface
        ai1, ai2 = vp1 * rho1, vp2 * rho2
        expected = (ai2 - ai1) / (ai2 + ai1)
        assert float(zoeppritz_rpp(*interface, 0.0)) == pytest.approx(expected, abs=1e-12)

    def test_aki_richards_at_zero_degrees(self):
        vp1, vs1, rho1, vp2, vs2, rho2 = SOFT_INTERFACE
        ai1, ai2 = vp1 * rho1, vp2 * rho2
        expected = (ai2 - ai1) / (ai2 + ai1)
        # The linearisation uses average-and-contrast, so it agrees with the
        # exact impedance reflectivity only to first order.
        assert float(aki_richards_rpp(*SOFT_INTERFACE, 0.0)) == pytest.approx(expected, abs=1e-3)


# ---------------------------------------------------------------- test 3 ----
class TestShueyFit:
    """Acceptance test 3: the fit recovers known A and B."""

    @pytest.mark.parametrize(
        "A,B",
        [(-0.12, -0.11), (0.13, -0.30), (0.0, 0.0), (-0.05, 0.42), (0.2, 0.05)],
    )
    def test_recovers_known_intercept_and_gradient(self, A, B):
        angles = np.arange(0.0, 41.0, 2.0)
        R = A + B * np.sin(np.radians(angles)) ** 2
        a_fit, b_fit = shuey_fit(R, angles)
        assert a_fit == pytest.approx(A, abs=1e-6)
        assert b_fit == pytest.approx(B, abs=1e-6)

    def test_fits_many_reflectors_at_once(self):
        angles = np.arange(0.0, 41.0, 5.0)
        truth = np.array([[-0.12, -0.11], [0.13, -0.30], [-0.05, 0.42]])
        sin2 = np.sin(np.radians(angles)) ** 2
        R = truth[:, [0]] + truth[:, [1]] * sin2[np.newaxis, :]
        A, B = shuey_fit(R, angles)
        assert np.allclose(A, truth[:, 0], atol=1e-9)
        assert np.allclose(B, truth[:, 1], atol=1e-9)

    def test_third_term_fit_recovers_c(self):
        angles = np.arange(0.0, 41.0, 2.0)
        A, B, C = -0.1, -0.2, 0.05
        th = np.radians(angles)
        R = A + B * np.sin(th) ** 2 + C * np.sin(th) ** 2 * np.tan(th) ** 2
        a, b, c = aki_richards_fit(R, angles, third_term=True)
        assert (a, b, c) == pytest.approx((A, B, C), abs=1e-6)

    def test_rejects_mismatched_angles(self):
        with pytest.raises(ValueError):
            shuey_fit(np.zeros(5), np.arange(4.0))


# ---------------------------------------------------------------- test 4 ----
#: Hand-built interfaces, one per AVO class, verified against Zoeppritz.
CLASS_INTERFACES = {
    "I": (2400.0, 1200.0, 2.35, 3000.0, 1800.0, 2.45),    # hard cemented sand
    "IIp": (2400.0, 1200.0, 2.35, 2500.0, 1450.0, 2.30),  # near-zero, positive
    "IIn": (2400.0, 1200.0, 2.35, 2450.0, 1450.0, 2.22),  # near-zero, negative
    "III": SOFT_INTERFACE,                                # classic bright gas sand
    "IV": (3300.0, 2000.0, 2.50, 2600.0, 1400.0, 2.20),   # soft sand, hard cap
}


class TestClassifier:
    """Acceptance test 4: the truth table classifies correctly at default a_tol."""

    @pytest.mark.parametrize("expected,interface", sorted(CLASS_INTERFACES.items()))
    def test_truth_table(self, expected, interface):
        angles = np.arange(0.0, 41.0, 2.0)
        R = zoeppritz_rpp(*interface, angles)
        A, B = shuey_fit(R, angles)
        assert classify(A, B) == expected

    def test_class_ii_signs_are_as_documented(self):
        angles = np.arange(0.0, 41.0, 2.0)
        for label in ("IIp", "IIn"):
            A, B = shuey_fit(zoeppritz_rpp(*CLASS_INTERFACES[label], angles), angles)
            assert abs(A) <= 0.02 and B < 0
            assert (A >= 0) if label == "IIp" else (A < 0)

    def test_tolerance_band_moves_the_ii_iii_boundary(self):
        # The prototype's marginal case: A = -0.037 sits on the II/III line.
        A, B = -0.037, -0.15
        assert classify(A, B, a_tol=0.02) == "III"
        assert classify(A, B, a_tol=0.05) == "IIn"

    def test_background_and_non_finite_fall_through(self):
        assert classify(0.15, 0.20) == "background/other"   # hard, brightening
        assert classify(np.nan, -0.1) == "background/other"


# ---------------------------------------------------------------- test 5 ----
def three_layer_model(n=300, top=100, base=180):  # noqa: D401
    """Shale / gas-sand / shale, sampled on a regular time grid."""
    vp = np.full(n, SHALE[0])
    vs = np.full(n, SHALE[1])
    rho = np.full(n, SHALE[2])
    vp[top:base], vs[top:base], rho[top:base] = GAS_SAND
    return vp, vs, rho, top, base


@pytest.fixture(scope="module")
def gather():
    """The validated three-layer gather: 30 Hz Ricker, dt = 1 ms, 0-40 deg."""
    vp, vs, rho, top, base = three_layer_model()
    angles = np.arange(0.0, 41.0, 5.0)
    _, w = ricker(30.0, 0.001)
    g = build_gather(vp, vs, rho, angles, w, dt=0.001, method="zoeppritz")
    return g, angles, top, base


class TestGatherSignature:
    """Acceptance test 5: the Class III gather signature."""

    def test_shape(self, gather):
        g, angles, _, _ = gather
        assert g.shape == (300, angles.size)

    def test_sand_top_is_a_trough_that_brightens_with_angle(self, gather):
        g, _, top, _ = gather
        trough = g[top - 15: top + 15, :].min(axis=0)
        assert np.all(trough < 0)                  # a trough, not a peak
        assert np.all(np.diff(trough) < 0)         # brightening (more negative)

    def test_sand_base_is_a_peak_that_brightens_with_angle(self, gather):
        g, _, _, base = gather
        peak = g[base - 15: base + 15, :].max(axis=0)
        assert np.all(peak > 0)
        assert np.all(np.diff(peak) > 0)

    def test_top_and_base_mirror_each_other_at_zero_offset(self, gather):
        g, _, top, base = gather
        assert g[top - 15: top + 15, 0].min() == pytest.approx(
            -g[base - 15: base + 15, 0].max(), rel=1e-6
        )

    def test_angle_and_full_stacks(self, gather):
        g, angles, top, _ = gather
        near = angle_stack(g, (0, 10), angles)
        far = angle_stack(g, (30, 40), angles)
        assert near.shape == far.shape == (g.shape[0],)
        # The far stack of a Class III sand top is brighter than the near.
        assert far[top - 15: top + 15].min() < near[top - 15: top + 15].min()
        assert np.allclose(full_stack(g), g.mean(axis=1))

    def test_angle_stack_rejects_an_empty_band(self, gather):
        g, angles, _, _ = gather
        with pytest.raises(ValueError):
            angle_stack(g, (60, 70), angles)


# ---------------------------------------------------------------- test 6 ----
class TestAmplitudeSpectrum:
    """What the wavelet can resolve, read off its spectrum."""

    def test_a_ricker_peaks_at_its_nominal_frequency(self):
        from avo_qi.core.wavelet import amplitude_spectrum, ricker

        for freq in (20.0, 30.0, 50.0):
            _, w = ricker(freq, 0.001)
            freqs, amplitude = amplitude_spectrum(w, 0.001)
            assert abs(freqs[int(np.argmax(amplitude))] - freq) < 1.0

    def test_the_amplitude_is_normalised_to_its_own_peak(self):
        """The shape is the point; the level depends on how the wavelet was
        scaled, which says nothing about bandwidth."""
        from avo_qi.core.wavelet import amplitude_spectrum, ricker

        _, w = ricker(30.0, 0.001)
        for scale in (1.0, 1e-4, 250.0):
            _, amplitude = amplitude_spectrum(w * scale, 0.001)
            assert amplitude.max() == pytest.approx(1.0)
            assert (amplitude >= 0).all()

    def test_padding_smooths_without_inventing_bandwidth(self):
        """Zero-padding interpolates the spectrum; it cannot move the band."""
        from avo_qi.core.wavelet import amplitude_spectrum, bandwidth, ricker

        _, w = ricker(30.0, 0.001)
        coarse = bandwidth(w, 0.001)
        fine_f, _ = amplitude_spectrum(w, 0.001, pad=32)
        raw_f, _ = amplitude_spectrum(w, 0.001, pad=1)
        assert fine_f.size > raw_f.size
        assert coarse[0] == pytest.approx(bandwidth(w, 0.001)[0])

    def test_the_band_scales_with_the_peak_frequency(self):
        from avo_qi.core.wavelet import bandwidth, ricker

        widths = []
        for freq in (20.0, 30.0, 50.0):
            low, high = bandwidth(ricker(freq, 0.001)[1], 0.001)
            assert 0 < low < freq < high
            widths.append(high - low)
        assert widths == sorted(widths)

    def test_an_ormsby_band_matches_the_corners_it_was_built_from(self):
        from avo_qi.core.wavelet import bandpass_ormsby, bandwidth

        _, w = bandpass_ormsby(5.0, 10.0, 60.0, 80.0, 0.001, 0.128)
        low, high = bandwidth(w, 0.001)
        # The -6 dB edges sit inside the outer corners and outside the flat
        # passband, which is what the taper does.
        assert 5.0 <= low <= 10.0
        assert 60.0 <= high <= 80.0

    def test_a_degenerate_wavelet_is_not_an_error(self):
        from avo_qi.core.wavelet import amplitude_spectrum, bandwidth

        for bad in (np.array([]), np.array([1.0])):
            freqs, amplitude = amplitude_spectrum(bad, 0.001)
            assert freqs.size == 0 and amplitude.size == 0
            assert np.isnan(bandwidth(bad, 0.001)).all()
        # An all-zero wavelet has a peak of zero and must not divide by it.
        freqs, amplitude = amplitude_spectrum(np.zeros(64), 0.001)
        assert np.isfinite(amplitude).all()


class TestWavelet:
    """Acceptance test 6: the Ricker wavelet is zero-phase with unit peak."""

    def test_symmetric_unit_peak_at_zero_time(self):
        t, w = ricker(30.0, 0.001)
        assert w.size % 2 == 1
        assert t[np.argmax(w)] == pytest.approx(0.0, abs=1e-12)
        assert np.max(np.abs(w)) == pytest.approx(1.0)
        assert np.allclose(w, w[::-1], atol=1e-12)
        assert is_zero_phase(w)

    @pytest.mark.parametrize("f", [15.0, 25.0, 30.0, 45.0])
    def test_peak_frequency_is_recovered_from_the_spectrum(self, f):
        _, w = ricker(f, 0.0005, length=0.512)
        assert dominant_frequency(w, 0.0005) == pytest.approx(f, rel=0.05)

    def test_ormsby_is_zero_phase_and_unit_peak(self):
        _, w = bandpass_ormsby(5.0, 10.0, 60.0, 80.0, 0.001)
        assert is_zero_phase(w)
        assert np.max(np.abs(w)) == pytest.approx(1.0)

    def test_load_wavelet_round_trip(self, tmp_path):
        t, w = ricker(30.0, 0.001)
        path = tmp_path / "wavelet.csv"
        np.savetxt(path, np.column_stack([t, w]), delimiter=",")
        t2, w2, meta = load_wavelet(str(path), dt=0.001)
        assert np.allclose(w2, w, atol=1e-9)
        assert meta["zero_phase"] is True
        assert meta["resampled"] is False

    def test_load_wavelet_resamples_to_trace_rate(self, tmp_path):
        t, w = ricker(30.0, 0.002)
        path = tmp_path / "coarse.csv"
        np.savetxt(path, np.column_stack([t, w]), delimiter=",")
        _, w2, meta = load_wavelet(str(path), dt=0.001)
        assert meta["resampled"] is True
        assert meta["n_samples"] > w.size
        assert np.max(np.abs(w2)) == pytest.approx(1.0)

    def test_load_wavelet_flags_a_time_shifted_wavelet(self):
        _, w = ricker(30.0, 0.001)
        _, _, meta = load_wavelet(np.roll(w, 10), dt=0.001)
        assert meta["zero_phase"] is False        # linear phase: peak off centre

    def test_load_wavelet_flags_a_phase_rotated_wavelet(self):
        from scipy.signal import hilbert

        _, w = ricker(30.0, 0.001)
        rotated = np.imag(hilbert(w))             # a true 90-degree rotation
        _, _, meta = load_wavelet(rotated, dt=0.001)
        assert meta["zero_phase"] is False

    def test_rejects_bad_parameters(self):
        with pytest.raises(ValueError):
            ricker(-5.0, 0.001)
        with pytest.raises(ValueError):
            bandpass_ormsby(10.0, 5.0, 60.0, 80.0, 0.001)


# ---------------------------------------------------------------- test 7 ----
class TestBothFitConsistency:
    """Acceptance test 7: Shuey and Aki-Richards agree at small angle, and the
    table reports the difference."""

    #: Stated tolerance for A/B agreement between the two fits out to 30 deg.
    A_TOL = 0.006
    B_TOL = 0.05

    @pytest.mark.parametrize("interface", sorted(CLASS_INTERFACES.values()))
    def test_shuey_and_aki_richards_agree_to_30_degrees(self, interface):
        angles = np.arange(0.0, 30.5, 2.0)
        r_exact = zoeppritz_rpp(*interface, angles)
        r_linear = aki_richards_rpp(*interface, angles)
        a_s, b_s = shuey_fit(r_exact, angles)
        a_a, b_a = aki_richards_fit(r_linear, angles)
        assert abs(a_s - a_a) < self.A_TOL
        assert abs(b_s - b_a) < self.B_TOL

    def test_reflector_table_reports_deltas_and_classes(self):
        vp, vs, rho, top, base = three_layer_model()
        angles = np.arange(0.0, 31.0, 2.0)
        depth = 2000.0 + np.arange(vp.size) * 0.1524

        table = reflector_avo(
            None, vp, vs, rho, angles, method="zoeppritz", depth=depth, threshold=0.01
        )

        assert len(table) == 2                       # sand top and sand base
        for col in ("A_shuey", "B_shuey", "A_ar", "B_ar", "dA", "dB", "avo_class"):
            assert col in table.columns

        # The Shuey and Aki-Richards fits of the same data agree closely.
        assert np.all(np.abs(table["dA"]) < 1e-9)
        assert np.all(np.abs(table["dB"]) < 1e-9)

        top_row = table.loc[table["sample"] == top - 1].iloc[0]
        assert top_row["avo_class"] == "III"
        assert top_row["A_shuey"] < 0 and top_row["B_shuey"] < 0

        base_row = table.loc[table["sample"] == base - 1].iloc[0]
        assert base_row["A_shuey"] > 0

    def test_reflector_table_accepts_a_precomputed_rc_matrix(self):
        vp, vs, rho, _, _ = three_layer_model()
        angles = np.arange(0.0, 31.0, 2.0)
        rc = reflectivity_series(vp, vs, rho, angles, method="aki_richards")
        table = reflector_avo(rc, vp, vs, rho, angles, threshold=0.01)
        assert len(table) == 2

    def test_empty_table_when_nothing_exceeds_the_threshold(self):
        vp, vs, rho, _, _ = three_layer_model()
        table = reflector_avo(None, vp, vs, rho, np.arange(0.0, 31.0, 5.0), threshold=0.99)
        assert len(table) == 0
        assert "avo_class" in table.columns


class TestBackgroundTrend:
    def test_fits_a_line_through_the_cloud(self):
        rng = np.random.default_rng(0)
        A = rng.normal(0, 0.08, 400)
        B = -1.4 * A + 0.01 + rng.normal(0, 0.004, 400)
        trend = background_trend(A, B)
        assert trend.slope == pytest.approx(-1.4, abs=0.05)
        assert trend.intercept == pytest.approx(0.01, abs=0.01)

    def test_is_robust_to_anomalies(self):
        rng = np.random.default_rng(1)
        A = rng.normal(0, 0.08, 400)
        B = -1.4 * A + rng.normal(0, 0.004, 400)
        A[:40], B[:40] = -0.15, -0.45          # a cluster of gas anomalies
        trend = background_trend(A, B)
        assert trend.slope == pytest.approx(-1.4, abs=0.1)
        assert np.all(np.abs(trend.deviation(A[:40], B[:40])) > 0.05)

    def test_degenerate_input(self):
        trend = background_trend([0.1], [0.2])
        assert np.isnan(trend.slope)


class TestReflectivitySeries:
    def test_shape_and_zero_last_row(self):
        vp, vs, rho, top, _ = three_layer_model()
        angles = np.arange(0.0, 31.0, 5.0)
        rc = reflectivity_series(vp, vs, rho, angles)
        assert rc.shape == (vp.size, angles.size)
        assert np.all(rc[-1] == 0.0)
        # Interface i sits between samples i and i+1.
        assert rc[top - 1, 0] < 0
        assert np.all(rc[:top - 1, 0] == 0.0)

    def test_unknown_method_is_rejected(self):
        with pytest.raises(ValueError):
            reflectivity_series(np.ones(3), np.ones(3), np.ones(3), [0.0], method="bogus")

    def test_mismatched_logs_are_rejected(self):
        with pytest.raises(ValueError):
            reflectivity_series(np.ones(3), np.ones(4), np.ones(3), [0.0])


class TestAttributes:
    def test_impedances_and_ratios(self):
        vp, vs, rho = np.array([2400.0]), np.array([1200.0]), np.array([2.35])
        assert acoustic_impedance(vp, rho)[0] == pytest.approx(5640.0)
        assert shear_impedance(vs, rho)[0] == pytest.approx(2820.0)
        assert vpvs(vp, vs)[0] == pytest.approx(2.0)
        assert poisson(vp, vs)[0] == pytest.approx(1.0 / 3.0)

    def test_lmr_identity(self):
        vp, vs, rho = np.array([2400.0]), np.array([1200.0]), np.array([2.35])
        ai = acoustic_impedance(vp, rho)
        si = shear_impedance(vs, rho)
        assert mu_rho(vs, rho)[0] == pytest.approx(si[0] ** 2)
        assert lambda_rho(vp, vs, rho)[0] == pytest.approx(ai[0] ** 2 - 2 * si[0] ** 2)

    def test_gas_sand_has_lower_vpvs_and_poisson_than_shale(self):
        shale_vpvs = vpvs(np.array([SHALE[0]]), np.array([SHALE[1]]))[0]
        sand_vpvs = vpvs(np.array([GAS_SAND[0]]), np.array([GAS_SAND[1]]))[0]
        assert sand_vpvs < shale_vpvs
        assert (
            poisson(np.array([GAS_SAND[0]]), np.array([GAS_SAND[1]]))[0]
            < poisson(np.array([SHALE[0]]), np.array([SHALE[1]]))[0]
        )

    def test_eei_reduces_to_acoustic_impedance_at_chi_zero(self):
        vp, vs, rho, _, _ = three_layer_model()
        e = eei(vp, vs, rho, 0.0)
        ai = acoustic_impedance(vp, rho)
        # At chi = 0 EEI is AI to within the normalisation constant.
        assert np.allclose(e / e.mean(), ai / ai.mean(), rtol=1e-9)

    def test_eei_is_finite_across_the_chi_sweep(self):
        vp, vs, rho, _, _ = three_layer_model()
        for chi in range(-90, 91, 15):
            assert np.all(np.isfinite(eei(vp, vs, rho, chi)))

    def test_zero_shear_yields_nan_not_inf(self):
        assert np.isnan(vpvs(np.array([1500.0]), np.array([0.0]))[0])


@pytest.fixture(scope="module")
def stacked_trace():
    """Full stack of the validated three-layer gather, plus its interfaces."""
    vp, vs, rho, top, base = three_layer_model()
    angles = np.arange(0.0, 41.0, 5.0)
    _, w = ricker(30.0, 0.001)
    return full_stack(build_gather(vp, vs, rho, angles, w, dt=0.001)), top, base


class TestTraceEvents:
    """Picking reflectors off the trace instead of off the logs.

    The logs know about every interface; the seismic shows only what its
    bandwidth resolves. Letting the trace pick means a reflector is always
    something that could actually be picked on a section.
    """

    @staticmethod
    def _trace():
        from avo_qi.core.wavelet import ricker

        _, wavelet = ricker(30.0, 0.001)
        rc = np.zeros(400)
        rc[100] = -0.20          # loud trough
        rc[200] = +0.10          # half as loud, a peak
        rc[300] = -0.01          # near-noise
        return np.convolve(rc, wavelet, mode="same")

    def test_each_event_is_a_turning_point_with_the_traces_own_polarity(self):
        from avo_qi.core.synthetic import _local_extrema, trace_events

        trace = self._trace()
        events = trace_events(trace, relative=0.0)
        turning = set(_local_extrema(trace).tolist())
        assert set(events["index"].tolist()) <= turning
        for i, pol in zip(events["index"], events["polarity"]):
            assert pol == (1 if trace[i] > 0 else -1)
        assert list(events["index"]) == sorted(events["index"])

    def test_the_cut_is_a_fraction_of_the_strongest_event(self):
        from avo_qi.core.synthetic import trace_events

        trace = self._trace()
        strongest = np.abs(trace_events(trace, relative=0.0)["amplitude"]).max()
        for relative in (0.0, 0.05, 0.25, 0.6):
            events = trace_events(trace, relative=relative)
            assert (np.abs(events["amplitude"]) >= relative * strongest - 1e-12).all()

    def test_raising_the_cut_only_ever_removes_events(self):
        from avo_qi.core.synthetic import trace_events

        trace = self._trace()
        previous = set(trace_events(trace, relative=0.0)["index"].tolist())
        for relative in (0.05, 0.2, 0.5, 0.9):
            kept = set(trace_events(trace, relative=relative)["index"].tolist())
            assert kept <= previous
            previous = kept

    def test_a_thin_bed_gives_one_event_not_two(self):
        """A top and a base closer than the wavelet can separate interfere into
        a single lobe. That lobe is the event; calling it two would be claiming
        resolution the trace does not have."""
        from avo_qi.core.wavelet import ricker
        from avo_qi.core.synthetic import trace_events

        _, wavelet = ricker(30.0, 0.001)
        rc = np.zeros(400)
        rc[200] = -0.2
        rc[204] = -0.2                       # 4 ms apart, far inside tuning
        trace = np.convolve(rc, wavelet, mode="same")
        events = trace_events(trace, relative=0.5)
        assert events["index"].size == 1
        assert events["polarity"][0] == -1

    def test_an_absolute_cut_overrides_the_fraction(self):
        from avo_qi.core.synthetic import trace_events

        trace = self._trace()
        events = trace_events(trace, relative=0.9, min_amplitude=0.0)
        assert events["index"].size > 1

    def test_a_flat_or_empty_trace_yields_nothing(self):
        from avo_qi.core.synthetic import trace_events

        for trace in (np.zeros(50), np.array([]), np.full(50, np.nan)):
            assert trace_events(trace)["index"].size == 0


class TestTraceExtrema:
    """Reflectors must be locatable as extrema on a trace, for class marking."""

    def test_finds_the_interface_extrema_with_correct_polarity(self, stacked_trace):
        trace, top, base = stacked_trace
        found = trace_extrema(trace, [top - 1, base - 1], half_window=8)
        # A zero-phase wavelet puts the extremum on the interface itself.
        assert list(found["offset"]) == [0, 0]
        assert found["polarity"][0] == -1        # gas sand top is a trough
        assert found["polarity"][1] == +1        # its base is a peak
        assert np.all(found["is_extremum"])

    def test_amplitudes_match_the_trace(self, stacked_trace):
        trace, top, _ = stacked_trace
        found = trace_extrema(trace, [top - 1], half_window=8)
        assert found["amplitude"][0] == pytest.approx(trace[found["index"][0]])

    def test_a_shifted_extremum_is_reported_with_its_offset(self):
        trace = np.zeros(100)
        trace[57] = -0.4                          # extremum three samples late
        found = trace_extrema(trace, [54], half_window=6)
        assert found["index"][0] == 57
        assert found["offset"][0] == 3
        assert found["polarity"][0] == -1

    def test_window_bounds_the_search(self):
        trace = np.zeros(100)
        trace[70] = 0.9
        found = trace_extrema(trace, [50], half_window=5)
        assert found["index"][0] != 70            # too far away to be claimed
        assert found["is_extremum"][0] == False   # noqa: E712 - flat window

    def test_a_nan_polarity_means_unknown_sign_not_undefined_behaviour(self):
        """A reflector whose R0 could not be computed has no expected sign.

        ``np.sign(nan).astype(int)`` is undefined and lands on INT_MIN, which
        matches no turning point at all — so the reflector was quietly reported
        as having no extremum on the strength of an integer overflow, with a
        RuntimeWarning as the only clue. Unknown must mean no constraint, which
        is what zero already means here.
        """
        import warnings

        trace = np.zeros(60)
        trace[10:31] = -np.sin(np.linspace(0, np.pi, 21))
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            found = trace_extrema(trace, [20, 20], half_window=3,
                                  polarity=[-1.0, np.nan])
        # Both find the trough: the second simply applies no sign filter.
        assert found["is_extremum"].tolist() == [True, True]
        assert found["index"].tolist() == [20, 20]

    def test_a_dead_trace_has_no_extrema_anywhere(self):
        """NaN != 0 and NaN != NaN are both True, so an unguarded slope test
        reads a gap in the trace as a direction reversal — and a blank trace
        came back with an extremum at every sample, each with NaN amplitude."""
        trace = np.full(20, np.nan)
        found = trace_extrema(trace, [10], half_window=3)
        assert not found["is_extremum"][0]
        assert found["polarity"][0] == 0

    def test_a_gap_in_the_trace_does_not_invent_extrema(self):
        trace = np.zeros(40)
        trace[10:21] = -np.sin(np.linspace(0, np.pi, 11))   # one real trough
        trace[28:33] = np.nan                                # a blank patch
        found = trace_extrema(trace, [15, 30], half_window=3)
        assert found["is_extremum"][0] and found["index"][0] == 15
        assert not found["is_extremum"][1]

    def test_flags_a_point_that_is_not_a_local_extremum(self):
        trace = np.linspace(0.0, 1.0, 50)         # monotonic: no interior peak
        found = trace_extrema(trace, [25], half_window=4)
        assert not found["is_extremum"][0]

    def test_handles_edges_and_empty_input(self):
        trace = np.array([0.0, -0.5, 0.0])
        found = trace_extrema(trace, [0, 2], half_window=5)
        assert found["index"].tolist() == [1, 1]
        empty = trace_extrema(np.array([]), [0])
        assert empty["index"].size == 0

    def test_out_of_range_samples_are_clipped_not_crashed(self, stacked_trace):
        trace, _, _ = stacked_trace
        found = trace_extrema(trace, [-5, 10_000], half_window=4)
        assert np.all(found["index"] >= 0)
        assert np.all(found["index"] < trace.size)

    def test_every_classified_reflector_gets_a_marker(self, stacked_trace):
        trace, _, _ = stacked_trace
        vp, vs, rho, _, _ = three_layer_model()
        angles = np.arange(0.0, 41.0, 5.0)
        table = reflector_avo(None, vp, vs, rho, angles, threshold=0.01)
        found = trace_extrema(trace, table["sample"].to_numpy(), half_window=8)
        assert found["index"].size == len(table)
        assert np.all(np.isfinite(found["amplitude"]))


class TestCriticalAngle:
    """Past critical the exact solution is complex; fits must stop short of it."""

    #: Shale over a cemented sand — fast enough to go critical inside 40 deg.
    HARD = (2395.0, 1198.0, 2.351, 4272.0, 2701.0, 2.369)

    def test_known_critical_angle(self):
        assert critical_angle(2395.0, 4272.0) == pytest.approx(34.1, abs=0.1)

    def test_no_critical_angle_for_a_soft_interface(self):
        assert np.isnan(critical_angle(2400.0, 2100.0))
        assert np.isnan(critical_angle(2400.0, 2400.0))

    def test_nan_mode_blanks_only_the_post_critical_angles(self):
        angles = np.arange(0.0, 41.0, 2.0)
        clipped = zoeppritz_rpp(*self.HARD, angles)
        blanked = zoeppritz_rpp(*self.HARD, angles, post_critical="nan")
        theta_c = critical_angle(self.HARD[0], self.HARD[3])
        pre, post = angles < theta_c, angles >= theta_c
        assert np.allclose(blanked[pre], clipped[pre])
        assert np.all(np.isnan(blanked[post]))
        assert np.all(np.isfinite(clipped))       # clip mode stays convolvable

    def test_clip_is_the_default_so_gathers_stay_finite(self):
        angles = np.arange(0.0, 41.0, 2.0)
        assert np.all(np.isfinite(zoeppritz_rpp(*self.HARD, angles)))
        rc = reflectivity_series(
            np.array([2395.0, 4272.0]), np.array([1198.0, 2701.0]),
            np.array([2.351, 2.369]), angles,
        )
        assert np.all(np.isfinite(rc))

    def test_invalid_mode_is_rejected(self):
        with pytest.raises(ValueError):
            zoeppritz_rpp(*self.HARD, 10.0, post_critical="wishful")

    def test_post_critical_angles_flip_the_gradient_if_left_in(self):
        """The bug this masking exists to prevent."""
        angles = np.arange(0.0, 41.0, 2.0)
        R = zoeppritz_rpp(*self.HARD, angles)
        a_all, b_all = shuey_fit(R, angles)
        theta_c = critical_angle(self.HARD[0], self.HARD[3])
        pre = angles < theta_c
        a_pre, b_pre = shuey_fit(R[pre], angles[pre])
        assert b_all > 0 and classify(a_all, b_all) == "background/other"
        assert b_pre < 0 and classify(a_pre, b_pre) == "I"

    def test_reflector_avo_masks_by_default_and_reports_the_angle(self):
        n = 200
        vp = np.full(n, self.HARD[0]); vs = np.full(n, self.HARD[1])
        rho = np.full(n, self.HARD[2])
        vp[100:150], vs[100:150], rho[100:150] = self.HARD[3:]
        angles = np.arange(0.0, 41.0, 2.0)

        masked = reflector_avo(None, vp, vs, rho, angles, threshold=0.05)
        loose = reflector_avo(None, vp, vs, rho, angles, threshold=0.05,
                              mask_post_critical=False)

        top = masked.iloc[0]
        assert top["avo_class"] == "I"
        assert top["B_shuey"] < 0
        assert top["critical_angle"] == pytest.approx(34.1, abs=0.1)
        assert top["n_angles"] < angles.size          # far angles dropped
        assert loose.iloc[0]["avo_class"] == "background/other"
        assert loose.iloc[0]["n_angles"] == angles.size

    def test_soft_interfaces_keep_every_angle(self):
        vp, vs, rho, _, _ = three_layer_model()
        angles = np.arange(0.0, 41.0, 2.0)
        table = reflector_avo(None, vp, vs, rho, angles, threshold=0.01)
        top = table.iloc[0]
        assert np.isnan(top["critical_angle"])        # no critical angle exists
        assert top["n_angles"] == angles.size
        assert top["avo_class"] == "III"

    def test_a_masked_fit_still_recovers_known_coefficients(self):
        angles = np.arange(0.0, 41.0, 2.0)
        A, B = 0.25, -0.4
        R = A + B * np.sin(np.radians(angles)) ** 2
        R[angles >= 34.0] = np.nan
        a_fit, b_fit = shuey_fit(R, angles)
        assert a_fit == pytest.approx(A, abs=1e-9)
        assert b_fit == pytest.approx(B, abs=1e-9)

    def test_too_few_surviving_angles_gives_nan_not_a_wrong_answer(self):
        angles = np.arange(0.0, 41.0, 2.0)
        R = np.full(angles.size, np.nan)
        R[0] = 0.2                                    # one angle cannot fit two terms
        a_fit, b_fit = shuey_fit(R, angles)
        assert np.isnan(a_fit) and np.isnan(b_fit)


class TestExtremumMatching:
    """Markers must land on a turning point of the reflector's own polarity.

    Picking the largest amplitude in a window instead hands a weak reflector
    its loud neighbour's lobe, putting the marker on the wrong sign.
    """

    @staticmethod
    def _two_events():
        """A weak peak at 40 beside a strong trough at 47."""
        t = np.linspace(-1.0, 1.0, 21)
        lobe = np.exp(-(t * 3) ** 2)
        trace = np.zeros(120)
        trace[30:51] += 0.05 * lobe          # weak peak centred on 40
        trace[37:58] -= 0.60 * lobe          # strong trough centred on 47
        return trace

    def test_polarity_stops_a_neighbour_stealing_the_marker(self):
        """A nearer turning point of the wrong sign must not win."""
        trace = np.zeros(100)
        trace[41] = -0.5                      # a neighbour's trough, very close
        trace[46] = +0.2                      # this reflector's own peak
        without = trace_extrema(trace, [40], half_window=10)
        with_sign = trace_extrema(trace, [40], half_window=10, polarity=[+1])

        # Nearest wins when nothing constrains the sign — here that is wrong.
        assert without["index"][0] == 41
        assert without["amplitude"][0] < 0
        # Told the reflector is a peak, the search skips the trough.
        assert with_sign["index"][0] == 46
        assert with_sign["amplitude"][0] > 0
        assert with_sign["is_extremum"][0]

    def test_every_marker_sits_on_a_real_turning_point(self):
        trace = self._two_events()
        turns = set(_local_extrema(trace).tolist())
        found = trace_extrema(trace, [40, 47], half_window=10, polarity=[+1, -1])
        for k in range(2):
            assert found["is_extremum"][k]
            assert int(found["index"][k]) in turns
            assert found["amplitude"][k] == pytest.approx(trace[found["index"][k]])

    def test_nearest_turning_point_wins_over_the_loudest(self):
        trace = np.zeros(100)
        trace[45] = 0.2                       # near and quiet
        trace[52] = 0.9                       # far and loud
        found = trace_extrema(trace, [44], half_window=10, polarity=[+1])
        assert found["index"][0] == 45

    def test_unresolved_reflectors_fall_back_to_the_interface(self):
        trace = np.zeros(100)
        trace[70] = -0.5
        found = trace_extrema(trace, [30], half_window=5, polarity=[-1])
        assert not found["is_extremum"][0]
        assert found["index"][0] == 30        # marker sits at the interface
        assert found["offset"][0] == 0

    def test_polarity_length_is_validated(self):
        with pytest.raises(ValueError):
            trace_extrema(np.zeros(50), [10, 20], polarity=[1])

    def test_zero_polarity_entries_impose_no_constraint(self):
        trace = self._two_events()
        found = trace_extrema(trace, [40], half_window=10, polarity=[0])
        assert found["is_extremum"][0]


class TestLocalExtrema:
    def test_finds_a_spike_not_its_shoulder(self):
        trace = np.zeros(100)
        trace[57] = -0.4
        assert _local_extrema(trace).tolist() == [57]

    def test_flat_and_monotonic_traces_have_no_turning_points(self):
        assert _local_extrema(np.zeros(50)).size == 0
        assert _local_extrema(np.linspace(0.0, 1.0, 50)).size == 0

    def test_finds_alternating_peaks_and_troughs(self):
        x = np.linspace(0, 4 * np.pi, 400)
        turns = _local_extrema(np.sin(x))
        assert turns.size == 4                       # two peaks, two troughs
        assert np.all(np.diff(np.sign(np.sin(x)[turns])) != 0)

    def test_short_traces_are_safe(self):
        assert _local_extrema(np.array([1.0, 2.0])).size == 0
        assert _local_extrema(np.array([])).size == 0


class TestShortTraceConvolution:
    """A window shorter than the wavelet must still come back trace-length.

    ``np.convolve(mode='same')`` returns ``max(len(trace), len(wavelet))``, so
    a thin analysis window used to raise a broadcast error instead of a gather.
    """

    @pytest.mark.parametrize("n", [400, 130, 129, 125, 60, 5, 2])
    def test_output_always_matches_the_trace_length(self, n):
        _, w = ricker(30.0, 0.001)              # 129 samples
        rc = np.zeros((n, 3))
        rc[n // 2, :] = 1.0
        assert convolve_series(rc, w).shape == (n, 3)

    @pytest.mark.parametrize("n", [400, 125, 60])
    def test_a_spike_stays_centred(self, n):
        _, w = ricker(30.0, 0.001)
        rc = np.zeros((n, 1))
        rc[n // 2, 0] = 1.0
        assert int(np.argmax(convolve_series(rc, w)[:, 0])) == n // 2

    def test_normal_lengths_are_unchanged(self):
        """The fix must not move any existing result."""
        rng = np.random.default_rng(0)
        _, w = ricker(30.0, 0.001)
        rc = rng.normal(0, 1, (400, 2))
        reference = np.column_stack([np.convolve(rc[:, j], w, mode="same")
                                     for j in range(2)])
        assert np.allclose(convolve_series(rc, w), reference)

    def test_a_short_window_builds_a_gather(self):
        vp, vs, rho, _, _ = three_layer_model(n=120, top=40, base=80)
        _, w = ricker(30.0, 0.001)
        gather = build_gather(vp, vs, rho, np.arange(0.0, 41.0, 5.0), w, dt=0.001)
        assert gather.shape == (120, 9)
        assert np.isfinite(gather).all()
