"""Tests for Gassmann fluid substitution."""

from __future__ import annotations

import numpy as np
import pytest

from avo_qi.core.gassmann import gassmann_dry, gassmann_saturate, substitute
from avo_qi.core.rockphysics import FLUIDS, MINERALS, bulk_modulus, shear_modulus

K_QUARTZ = MINERALS["quartz"][0]
BRINE, OIL, GAS = FLUIDS["brine"], FLUIDS["oil"], FLUIDS["gas"]


def _biot_form(k_dry, k_mineral, k_fluid, porosity):
    """Gassmann written the other standard way, as an independent check.

    ``K_sat = K_dry + (1 - K_dry/K0)^2 /
              (phi/K_fl + (1-phi)/K0 - K_dry/K0^2)``

    Algebraically the same relation as the ratio form the module uses, but a
    different sequence of operations, so agreement tests the algebra rather
    than comparing the code with itself.
    """
    beta = 1.0 - k_dry / k_mineral
    denom = porosity / k_fluid + (1.0 - porosity) / k_mineral - k_dry / k_mineral ** 2
    return k_dry + beta ** 2 / denom


class TestAlgebra:
    def test_matches_the_biot_form(self):
        k_dry = np.array([2.0, 6.0, 12.0, 20.0])
        phi = np.array([0.30, 0.25, 0.18, 0.10])
        got = gassmann_saturate(k_dry, K_QUARTZ, BRINE[0], phi)
        want = _biot_form(k_dry, K_QUARTZ, BRINE[0], phi)
        assert got.valid.all()
        assert np.allclose(got.values, want, rtol=1e-12, atol=0)

    def test_round_trip_recovers_the_dry_frame(self):
        k_dry = np.array([1.5, 5.0, 11.0, 22.0])
        phi = np.array([0.32, 0.26, 0.19, 0.08])
        wet = gassmann_saturate(k_dry, K_QUARTZ, BRINE[0], phi)
        back = gassmann_dry(wet.values, K_QUARTZ, BRINE[0], phi)
        assert back.valid.all()
        assert np.allclose(back.values, k_dry, rtol=1e-12, atol=1e-12)

    def test_round_trip_the_other_way_round(self):
        k_sat = np.array([8.0, 14.0, 25.0])
        phi = np.array([0.28, 0.20, 0.12])
        dry = gassmann_dry(k_sat, K_QUARTZ, OIL[0], phi)
        wet = gassmann_saturate(dry.values, K_QUARTZ, OIL[0], phi)
        assert np.allclose(wet.values, k_sat, rtol=1e-12, atol=1e-12)

    def test_a_vacuum_leaves_the_frame_alone(self):
        """K_fl = 0 is the dry rock, so saturating with it must change nothing."""
        k_dry = np.array([2.0, 9.0, 18.0])
        wet = gassmann_saturate(k_dry, K_QUARTZ, 0.0, 0.25)
        assert np.allclose(wet.values, k_dry, rtol=0, atol=1e-14)

    def test_a_fluid_as_stiff_as_the_mineral_gives_the_mineral(self):
        near = K_QUARTZ * (1.0 - 1e-9)
        wet = gassmann_saturate(5.0, K_QUARTZ, near, 0.25)
        assert float(wet.values) == pytest.approx(K_QUARTZ, rel=1e-6)

    def test_saturation_stiffens(self):
        k_dry = np.linspace(1.0, 25.0, 25)
        wet = gassmann_saturate(k_dry, K_QUARTZ, BRINE[0], 0.25)
        assert np.all(wet.values > k_dry)

    def test_a_stiffer_fluid_gives_a_stiffer_rock(self):
        soft = gassmann_saturate(6.0, K_QUARTZ, GAS[0], 0.25)
        stiff = gassmann_saturate(6.0, K_QUARTZ, BRINE[0], 0.25)
        assert float(stiff.values) > float(soft.values)

    def test_broadcasts_scalars_against_logs(self):
        phi = np.linspace(0.05, 0.35, 7)
        wet = gassmann_saturate(6.0, K_QUARTZ, BRINE[0], phi)
        assert wet.values.shape == phi.shape
        # Less pore space means the fluid matters less.
        assert np.all(np.diff(wet.values) < 0)


class TestValidity:
    def test_a_negative_dry_frame_is_reported_not_returned(self):
        """The finding that started this: a rock that cannot be that rock.

        The demo well's brine sand was originally 2500 / 1450 / 2.30 with an
        invented porosity of 0.26.  Run backwards against quartz and brine it
        needs a dry frame of about -1.6 GPa, which is impossible, and that is
        what exposed the porosity as inconsistent with the density log.
        """
        k_sat = bulk_modulus(np.array([2500.0]), np.array([1450.0]), np.array([2.30]))
        out = gassmann_dry(k_sat, K_QUARTZ, BRINE[0], 0.26)
        assert not out.valid[0]
        assert out.reasons[0] == "dry-frame modulus is negative"
        assert np.isnan(out.values[0])

    def test_the_rebuilt_brine_sand_substitutes_cleanly(self):
        """The same layer after the rebuild — the other half of the regression."""
        k_sat = bulk_modulus(np.array([2830.2]), np.array([1498.9]), np.array([2.2444]))
        out = gassmann_dry(k_sat, K_QUARTZ, BRINE[0], 0.26)
        assert out.valid[0]
        assert 0.0 < float(out.values[0]) < K_QUARTZ

    @pytest.mark.parametrize("phi, reason", [
        (0.0, "porosity outside (0, 1)"),
        (1.0, "porosity outside (0, 1)"),
        (-0.1, "porosity outside (0, 1)"),
        (np.nan, "porosity is not finite"),
    ])
    def test_impossible_porosity_is_named_as_such(self, phi, reason):
        out = gassmann_saturate(6.0, K_QUARTZ, BRINE[0], phi)
        assert not out.valid.all()
        assert out.reasons.ravel()[0] == reason

    def test_the_first_failure_wins(self):
        """A bad porosity explains the sample better than what it goes on to break."""
        out = gassmann_dry(6.0, K_QUARTZ, BRINE[0], 0.0)
        assert out.reasons.ravel()[0] == "porosity outside (0, 1)"

    def test_a_fluid_stiffer_than_the_mineral_is_rejected(self):
        out = gassmann_saturate(6.0, 10.0, 20.0, 0.25)
        assert out.reasons.ravel()[0] == "fluid is stiffer than the mineral"

    def test_a_rock_as_stiff_as_its_mineral_is_rejected(self):
        out = gassmann_dry(K_QUARTZ, K_QUARTZ, BRINE[0], 0.25)
        assert out.reasons.ravel()[0] == "saturated rock is as stiff as the mineral"

    def test_non_finite_input_is_flagged_rather_than_crashing(self):
        out = gassmann_dry(np.array([10.0, np.nan]), K_QUARTZ, BRINE[0], 0.25)
        assert out.valid[0] and not out.valid[1]
        assert out.reasons[1] == "saturated modulus is not finite"

    def test_summary_counts_the_reasons(self):
        k_sat = np.array([10.0, 10.0, 10.0])
        out = gassmann_dry(k_sat, K_QUARTZ, BRINE[0], np.array([0.25, 0.0, np.nan]))
        assert out.n_invalid == 2
        assert out.summary() == {"porosity outside (0, 1)": 1,
                                 "porosity is not finite": 1}

    def test_valid_samples_survive_alongside_invalid_ones(self):
        phi = np.array([0.25, 0.0, 0.20])
        out = gassmann_saturate(6.0, K_QUARTZ, BRINE[0], phi)
        assert list(out.valid) == [True, False, True]
        assert np.isfinite(out.values[[0, 2]]).all()


class TestSubstitute:
    #: The demo well's layers, with the fluid cases the LAS already carries.
    #: Those numbers were produced by the well generator, not by this module,
    #: so reproducing them is an independent check rather than a self-comparison.
    LAYERS = [
        ("brine sand", 0.26, (2830.2, 1498.9, 2.2444),
         {"oil": (2496.8, 1526.6, 2.1638), "gas": (2315.8, 1577.6, 2.0260)}),
        ("cemented sand", 0.18, (4265.3, 2704.1, 2.3692),
         {"oil": (4176.4, 2736.5, 2.3134), "gas": (4180.6, 2794.8, 2.2180)}),
    ]

    @pytest.mark.parametrize("name, phi, insitu, targets", LAYERS,
                             ids=[row[0] for row in LAYERS])
    def test_reproduces_the_demo_wells_own_fluid_cases(self, name, phi, insitu, targets):
        for fluid, want in targets.items():
            out = substitute(insitu[0], insitu[1], insitu[2], phi, K_QUARTZ,
                             BRINE, FLUIDS[fluid])
            # The LAS records these to 0.1 m/s, which sets the tolerance.
            assert float(out["VP"]) == pytest.approx(want[0], abs=0.1)
            assert float(out["VS"]) == pytest.approx(want[1], abs=0.1)
            assert float(out["RHOB"]) == pytest.approx(want[2], abs=1e-4)

    def test_shear_modulus_is_carried_through_untouched(self):
        vp, vs, rho, phi = 2830.2, 1498.9, 2.2444, 0.26
        out = substitute(vp, vs, rho, phi, K_QUARTZ, BRINE, GAS)
        before = float(np.ravel(shear_modulus(vs, rho))[0])
        after = float(np.ravel(shear_modulus(out["VS"], out["RHOB"]))[0])
        assert after == pytest.approx(before, rel=1e-12)

    def test_density_moves_by_mass_balance_on_the_measured_density(self):
        rho, phi = 2.2444, 0.26
        out = substitute(2830.2, 1498.9, rho, phi, K_QUARTZ, BRINE, GAS)
        expected = rho + phi * (GAS[1] - BRINE[1])
        assert float(out["RHOB"]) == pytest.approx(expected, rel=1e-12)

    def test_brine_to_gas_slows_the_rock_down(self):
        out = substitute(2830.2, 1498.9, 2.2444, 0.26, K_QUARTZ, BRINE, GAS)
        assert float(out["VP"]) < 2830.2
        # Vs rises even though mu is fixed, because the rock got lighter.
        assert float(out["VS"]) > 1498.9

    def test_substituting_back_returns_the_original_log(self):
        vp, vs, rho, phi = 2830.2, 1498.9, 2.2444, 0.26
        gassy = substitute(vp, vs, rho, phi, K_QUARTZ, BRINE, GAS)
        back = substitute(gassy["VP"], gassy["VS"], gassy["RHOB"], phi,
                          K_QUARTZ, GAS, BRINE)
        assert float(back["VP"]) == pytest.approx(vp, rel=1e-10)
        assert float(back["VS"]) == pytest.approx(vs, rel=1e-10)
        assert float(back["RHOB"]) == pytest.approx(rho, rel=1e-12)

    def test_an_unsubstitutable_sample_comes_back_nan_with_its_reason(self):
        vp = np.array([2830.2, 2500.0])
        vs = np.array([1498.9, 1450.0])
        rho = np.array([2.2444, 2.30])
        out = substitute(vp, vs, rho, 0.26, K_QUARTZ, BRINE, GAS)
        assert list(out["valid"]) == [True, False]
        assert np.isnan(out["VP"][1])
        assert out["reasons"][1] == "dry-frame modulus is negative"
        # The good sample is unaffected by its neighbour.
        assert float(out["VP"][0]) == pytest.approx(2315.8, abs=0.1)

    def test_a_failure_keeps_the_reason_from_the_stage_that_failed(self):
        out = substitute(2500.0, 1450.0, 2.30, 0.26, K_QUARTZ, BRINE, GAS)
        assert out["reasons"].ravel()[0] == "dry-frame modulus is negative"

    def test_fluid_properties_may_vary_down_the_well(self):
        """Batzle-Wang values follow pressure and temperature, so they are logs."""
        n = 8
        vp, vs, rho = np.full(n, 2830.2), np.full(n, 1498.9), np.full(n, 2.2444)
        k_gas = np.linspace(0.02, 0.09, n)
        out = substitute(vp, vs, rho, 0.26, K_QUARTZ, BRINE, (k_gas, 0.25))
        assert out["valid"].all()
        # A stiffer gas at depth means a faster rock.
        assert np.all(np.diff(out["VP"]) > 0)

    def test_works_over_a_whole_log(self):
        rng = np.random.default_rng(7)
        n = 500
        phi = rng.uniform(0.10, 0.32, n)
        vp = rng.uniform(2600.0, 3400.0, n)
        vs = vp / rng.uniform(1.7, 2.0, n)
        rho = rng.uniform(2.15, 2.45, n)
        out = substitute(vp, vs, rho, phi, K_QUARTZ, BRINE, GAS)
        assert out["VP"].shape == (n,)
        # Every sample either substitutes or says why not; none is silent.
        silent = np.isnan(out["VP"]) & (out["reasons"] == "")
        assert not silent.any()
