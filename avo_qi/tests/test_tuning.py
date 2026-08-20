"""Tuned versus untuned AVO.

Untuned is the interface response — reflection coefficients, no wavelet.
Tuned is the measured response — the same interface after convolution, where a
thin bed's top and base interfere.  These tests pin the difference.
"""

from __future__ import annotations

import numpy as np
import pytest

from avo_qi.core.avo import reflector_avo
from avo_qi.core.reflectivity import reflectivity_series
from avo_qi.core.synthetic import build_gather
from avo_qi.core.tuning import (
    apparent_frequency,
    apparent_period,
    tuned_amplitudes,
    tuned_avo,
    tuning_thickness_depth,
    tuning_thickness_from_wavelet,
    tuning_thickness_twt,
    wavelet_scale,
    wedge_model,
)
from avo_qi.core.wavelet import ricker

SHALE = (2400.0, 1200.0, 2.35)
GAS_SAND = (2100.0, 1300.0, 2.10)
#: A near-zero-intercept sand: Class IIn as rock, but it tunes into Class III.
CLASS_II_SAND = (2450.0, 1450.0, 2.22)
ANGLES = np.arange(0.0, 41.0, 2.0)
DT = 0.001


@pytest.fixture(scope="module")
def wavelet():
    _, w = ricker(30.0, DT)
    return w


def layer_model(reservoir, thickness, n=200, top=80):
    vp = np.full(n, SHALE[0])
    vs = np.full(n, SHALE[1])
    rho = np.full(n, SHALE[2])
    vp[top:top + thickness], vs[top:top + thickness], rho[top:top + thickness] = reservoir
    return vp, vs, rho


class TestWaveletScale:
    def test_unit_peak_wavelet_scales_by_one(self, wavelet):
        assert wavelet_scale(wavelet) == pytest.approx(1.0)

    def test_scaling_is_carried_through(self, wavelet):
        assert wavelet_scale(wavelet * 0.5) == pytest.approx(0.5)
        assert wavelet_scale(-wavelet) == pytest.approx(-1.0)   # reverse polarity

    def test_empty_wavelet_is_neutral(self):
        assert wavelet_scale(np.array([])) == 1.0


class TestApparentFrequency:
    """Tuning is governed by the waveform period, not the spectral peak."""

    @pytest.mark.parametrize("f_peak", [20.0, 30.0, 45.0])
    def test_ricker_apparent_period_matches_theory(self, f_peak):
        # A Ricker of peak frequency f has apparent period sqrt(6) / (pi f).
        _, w = ricker(f_peak, 0.0005, length=0.5)
        expected = np.sqrt(6) / (np.pi * f_peak)
        assert apparent_period(w, 0.0005) == pytest.approx(expected, rel=0.02)

    def test_apparent_frequency_exceeds_the_peak_frequency(self):
        _, w = ricker(30.0, 0.0005, length=0.5)
        # pi/sqrt(6) = 1.2825; using the spectral peak instead would overstate
        # the tuning thickness by that factor.
        assert apparent_frequency(w, 0.0005) == pytest.approx(30.0 * 1.2825, rel=0.02)

    def test_a_wavelet_without_side_lobes_is_rejected(self):
        with pytest.raises(ValueError):
            apparent_period(np.array([0.0, 1.0, 0.0]), DT)


class TestTuningThickness:
    def test_twt_form_is_half_the_apparent_period(self, wavelet):
        f_a = apparent_frequency(wavelet, DT)
        assert tuning_thickness_twt(f_a) == pytest.approx(
            tuning_thickness_from_wavelet(wavelet, DT), rel=1e-9
        )

    def test_quarter_wavelength_in_depth(self):
        assert tuning_thickness_depth(25.0, 3000.0) == pytest.approx(30.0)

    def test_rejects_non_positive_frequency(self):
        with pytest.raises(ValueError):
            tuning_thickness_twt(0.0)
        with pytest.raises(ValueError):
            tuning_thickness_depth(-5.0, 3000.0)

    @pytest.mark.parametrize("f_peak", [20.0, 30.0, 45.0])
    def test_prediction_matches_where_a_wedge_actually_tunes(self, f_peak):
        """The headline check: predicted tuning thickness is where the wedge
        amplitude actually peaks."""
        dt = 0.0005
        _, w = ricker(f_peak, dt)
        predicted = tuning_thickness_from_wavelet(w, dt)
        thicknesses = np.arange(0, int(0.09 / dt), 1)
        wedge = wedge_model(SHALE, GAS_SAND, SHALE, thicknesses, ANGLES, w,
                            dt=dt, pad=120)
        peak = int(np.nanargmax(np.abs(wedge["top_amplitude"][:, 0])))
        assert wedge["thickness_twt"][peak] == pytest.approx(predicted, abs=1.5 * dt)


class TestThickBedIsUntuned:
    """Far from tuning there is no interference, so tuned must equal untuned."""

    def test_thick_bed_reproduces_the_interface_response(self, wavelet):
        vp, vs, rho = layer_model(GAS_SAND, 80)
        table = tuned_avo(vp, vs, rho, ANGLES, wavelet, dt=DT, threshold=0.05)
        row = table.iloc[0]
        assert row["A_tuned"] == pytest.approx(row["A_shuey"], abs=1e-4)
        assert row["B_tuned"] == pytest.approx(row["B_shuey"], abs=1e-4)
        assert row["tuning_ratio"] == pytest.approx(1.0, abs=1e-3)
        assert not row["tuning_changes_class"]

    @pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
    def test_ratio_is_one_whatever_the_wavelet_amplitude(self, wavelet, scale):
        """The thick-bed reference keeps the comparison in consistent units."""
        vp, vs, rho = layer_model(GAS_SAND, 80)
        table = tuned_avo(vp, vs, rho, ANGLES, wavelet * scale, dt=DT, threshold=0.05)
        row = table.iloc[0]
        assert row["tuning_ratio"] == pytest.approx(1.0, abs=1e-3)
        # Pure reflection coefficients do not move with the wavelet...
        assert row["A_shuey"] == pytest.approx(-0.1210, abs=1e-3)
        # ...but the seismic-units reference does.
        assert row["A_thick"] == pytest.approx(row["A_shuey"] * scale, rel=1e-9)
        assert row["A_tuned"] == pytest.approx(row["A_thick"], abs=1e-4)


class TestThinBedTunes:
    def test_amplitude_brightens_towards_tuning_thickness(self, wavelet):
        thicknesses = np.arange(0, 61, 1)
        wedge = wedge_model(SHALE, GAS_SAND, SHALE, thicknesses, ANGLES, wavelet, dt=DT)
        zero_offset = np.abs(wedge["top_amplitude"][:, 0])
        reference = abs(wedge["reference_amplitude"][0])
        tuning = tuning_thickness_from_wavelet(wavelet, DT)
        peak = thicknesses[int(np.nanargmax(zero_offset))] * DT

        assert peak == pytest.approx(tuning, abs=2 * DT)
        # Tuning brightens the event well above its own interface strength.
        assert np.nanmax(zero_offset) > 1.3 * reference
        # A thick bed is back to the interface value.
        assert zero_offset[-1] == pytest.approx(reference, rel=0.02)

    def test_very_thin_beds_dim_again(self, wavelet):
        thicknesses = np.arange(1, 61, 1)
        wedge = wedge_model(SHALE, GAS_SAND, SHALE, thicknesses, ANGLES, wavelet, dt=DT)
        zero_offset = np.abs(wedge["top_amplitude"][:, 0])
        peak = int(np.nanargmax(zero_offset))
        # Below tuning the top and base start cancelling.
        assert zero_offset[0] < zero_offset[peak]

    def test_tuning_moves_the_gradient_not_just_the_brightness(self, wavelet):
        """Top and base vary differently with angle, so tuning is angular."""
        vp, vs, rho = layer_model(GAS_SAND, 13)
        table = tuned_avo(vp, vs, rho, ANGLES, wavelet, dt=DT, threshold=0.02)
        row = table.iloc[0]
        assert abs(row["dB_tuning"]) > 0.02
        assert abs(row["B_tuned"]) > abs(row["B_thick"])


class TestTuningCanChangeClass:
    """The reason this matters: thickness alone can restyle a reflector."""

    @pytest.mark.parametrize("thickness", [8, 10, 13, 17])
    def test_a_class_two_sand_reads_as_class_three_near_tuning(self, wavelet, thickness):
        vp, vs, rho = layer_model(CLASS_II_SAND, thickness)
        row = tuned_avo(vp, vs, rho, ANGLES, wavelet, dt=DT, threshold=0.005).iloc[0]
        assert row["avo_class"] == "IIn"          # the rock
        assert row["class_tuned"] == "III"        # the seismic
        assert bool(row["tuning_changes_class"])

    @pytest.mark.parametrize("thickness", [40, 80])
    def test_the_same_sand_reads_true_when_it_is_thick(self, wavelet, thickness):
        vp, vs, rho = layer_model(CLASS_II_SAND, thickness)
        row = tuned_avo(vp, vs, rho, ANGLES, wavelet, dt=DT, threshold=0.005).iloc[0]
        assert row["avo_class"] == "IIn"
        assert row["class_tuned"] == "IIn"
        assert not bool(row["tuning_changes_class"])


class TestTunedAmplitudes:
    def test_shape_and_angle_tracking(self, wavelet):
        vp, vs, rho = layer_model(GAS_SAND, 30)
        gather = build_gather(vp, vs, rho, ANGLES, wavelet, dt=DT)
        rc = reflectivity_series(vp, vs, rho, ANGLES)
        table = reflector_avo(rc, vp, vs, rho, ANGLES, threshold=0.05)
        picks = tuned_amplitudes(gather, table["sample"].to_numpy(),
                                 polarity=np.sign(table["R0"].to_numpy(float)))
        assert picks["amplitude"].shape == (len(table), ANGLES.size)
        assert np.all(picks["is_extremum"])
        # A Class III top stays a trough at every angle, and brightens.
        top = picks["amplitude"][0]
        assert np.all(top < 0)
        assert top[-1] < top[0]

    def test_matches_the_gather_it_was_picked_from(self, wavelet):
        vp, vs, rho = layer_model(GAS_SAND, 30)
        gather = build_gather(vp, vs, rho, ANGLES, wavelet, dt=DT)
        picks = tuned_amplitudes(gather, [79], polarity=[-1])
        for j in range(ANGLES.size):
            assert picks["amplitude"][0, j] == pytest.approx(
                gather[picks["index"][0, j], j]
            )


class TestTunedAvoTable:
    def test_carries_both_responses_and_their_difference(self, wavelet):
        vp, vs, rho = layer_model(GAS_SAND, 13)
        table = tuned_avo(vp, vs, rho, ANGLES, wavelet, dt=DT, threshold=0.02)
        for col in ("A_shuey", "B_shuey", "A_thick", "B_thick", "A_tuned", "B_tuned",
                    "class_tuned", "dA_tuning", "dB_tuning", "tuning_ratio",
                    "tuning_changes_class"):
            assert col in table.columns
        row = table.iloc[0]
        assert row["dA_tuning"] == pytest.approx(row["A_tuned"] - row["A_thick"])
        assert row["dB_tuning"] == pytest.approx(row["B_tuned"] - row["B_thick"])

    def test_empty_when_nothing_clears_the_threshold(self, wavelet):
        vp, vs, rho = layer_model(GAS_SAND, 30)
        table = tuned_avo(vp, vs, rho, ANGLES, wavelet, dt=DT, threshold=0.99)
        assert len(table) == 0
        assert "class_tuned" in table.columns


class TestWedgeModel:
    def test_shapes_and_axes(self, wavelet):
        thicknesses = np.arange(0, 41, 5)
        wedge = wedge_model(SHALE, GAS_SAND, SHALE, thicknesses, ANGLES, wavelet, dt=DT)
        assert wedge["top_amplitude"].shape == (thicknesses.size, ANGLES.size)
        assert wedge["base_amplitude"].shape == (thicknesses.size, ANGLES.size)
        assert np.allclose(wedge["thickness_twt"], thicknesses * DT)
        assert wedge["gathers"].shape[0] == thicknesses.size

    def test_zero_thickness_has_no_reservoir_to_pick(self, wavelet):
        wedge = wedge_model(SHALE, GAS_SAND, SHALE, [0, 20], ANGLES, wavelet, dt=DT)
        assert np.all(np.isnan(wedge["top_amplitude"][0]))
        assert np.all(np.isfinite(wedge["top_amplitude"][1]))

    def test_top_and_base_have_opposite_polarity(self, wavelet):
        wedge = wedge_model(SHALE, GAS_SAND, SHALE, [40], ANGLES, wavelet, dt=DT)
        assert wedge["top_amplitude"][0, 0] < 0     # soft sand: trough on top
        assert wedge["base_amplitude"][0, 0] > 0    # peak at its base

    def test_thick_end_returns_the_interface_fit(self, wavelet):
        wedge = wedge_model(SHALE, GAS_SAND, SHALE, [80], ANGLES, wavelet, dt=DT)
        assert wedge["A_top"][0] == pytest.approx(wedge["A_interface"], abs=1e-4)
        assert wedge["B_top"][0] == pytest.approx(wedge["B_interface"], abs=1e-4)
