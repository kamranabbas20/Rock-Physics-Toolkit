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
    reflectivity_series,
    zoeppritz_rpp,
)
from avo_qi.core.synthetic import angle_stack, build_gather, full_stack
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
def three_layer_model(n=300, top=100, base=180):
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
