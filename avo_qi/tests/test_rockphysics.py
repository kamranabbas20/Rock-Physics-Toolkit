"""Tests for the diagnostic rock-physics models in ``core/rockphysics.py``.

These models are overlays for diagnosing log data, not fluid substitution —
there is deliberately no Gassmann in the toolkit, so nothing here tests one.
"""

from __future__ import annotations

import numpy as np
import pytest

from avo_qi.core.rockphysics import (
    FLUIDS,
    MINERALS,
    bulk_modulus,
    castagna_mudrock,
    coordination_number,
    critical_porosity_dry,
    fit_gardner,
    gardner_density,
    gardner_velocity,
    greenberg_castagna,
    hashin_shtrikman,
    hertz_mindlin,
    raymer_hunt_gardner,
    reuss,
    shear_modulus,
    soft_sand_dry,
    stiff_sand_dry,
    velocities_from_moduli,
    voigt,
    voigt_reuss_hill,
    wyllie,
)

QUARTZ_K, QUARTZ_G, QUARTZ_RHO = MINERALS["quartz"]
CLAY_K, CLAY_G, CLAY_RHO = MINERALS["clay"]
BRINE_K, BRINE_RHO = FLUIDS["brine"]


class TestModuliFromVelocities:
    def test_round_trip(self):
        vp, vs, rho = 2400.0, 1200.0, 2.35
        K = bulk_modulus(vp, vs, rho)
        G = shear_modulus(vs, rho)
        vp2, vs2 = velocities_from_moduli(K, G, rho)
        assert float(vp2) == pytest.approx(vp, rel=1e-12)
        assert float(vs2) == pytest.approx(vs, rel=1e-12)

    def test_known_values_in_gigapascals(self):
        # A soft shale: single-digit GPa, not Pa and not MPa.
        assert float(bulk_modulus(2400.0, 1200.0, 2.35)) == pytest.approx(9.024, abs=1e-3)
        assert float(shear_modulus(1200.0, 2.35)) == pytest.approx(3.384, abs=1e-3)

    def test_vectorised(self):
        vp = np.array([2400.0, 3000.0])
        K = bulk_modulus(vp, vp / 2, np.array([2.35, 2.45]))
        assert K.shape == vp.shape


class TestBounds:
    FRACTIONS = [0.7, 0.3]
    MODULI = [QUARTZ_K, BRINE_K]

    def test_voigt_reuss_hill_ordering(self):
        v = voigt(self.MODULI, self.FRACTIONS)
        r = reuss(self.MODULI, self.FRACTIONS)
        h = voigt_reuss_hill(self.MODULI, self.FRACTIONS)
        assert r < h < v

    def test_bounds_collapse_for_a_single_phase(self):
        assert voigt([QUARTZ_K], [1.0]) == pytest.approx(QUARTZ_K)
        assert reuss([QUARTZ_K], [1.0]) == pytest.approx(QUARTZ_K)
        assert voigt_reuss_hill([QUARTZ_K], [1.0]) == pytest.approx(QUARTZ_K)

    def test_fractions_are_normalised(self):
        assert voigt(self.MODULI, [7.0, 3.0]) == pytest.approx(
            voigt(self.MODULI, [0.7, 0.3])
        )

    def test_reuss_is_zero_with_a_zero_modulus_phase(self):
        # A fluid's shear modulus drives the iso-stress bound to zero.
        assert reuss([QUARTZ_G, 0.0], [0.7, 0.3]) == 0.0

    def test_rejects_degenerate_fractions(self):
        with pytest.raises(ValueError):
            voigt([1.0, 2.0], [0.0, 0.0])
        with pytest.raises(ValueError):
            voigt([1.0, 2.0], [0.5, 0.25, 0.25])

    def test_hashin_shtrikman_sits_inside_voigt_reuss(self):
        f1 = np.linspace(0.05, 0.95, 19)
        hs = hashin_shtrikman(QUARTZ_K, QUARTZ_G, CLAY_K, CLAY_G, f1)
        for i, f in enumerate(f1):
            v = voigt([QUARTZ_K, CLAY_K], [f, 1 - f])
            r = reuss([QUARTZ_K, CLAY_K], [f, 1 - f])
            assert r <= hs["K_lower"][i] + 1e-9
            assert hs["K_lower"][i] <= hs["K_upper"][i] + 1e-9
            assert hs["K_upper"][i] <= v + 1e-9

    def test_hashin_shtrikman_collapses_at_the_end_members(self):
        hs = hashin_shtrikman(QUARTZ_K, QUARTZ_G, BRINE_K, 0.0, np.array([0.0, 1.0]))
        assert hs["K_lower"][0] == pytest.approx(BRINE_K)
        assert hs["K_upper"][0] == pytest.approx(BRINE_K)
        assert hs["K_lower"][1] == pytest.approx(QUARTZ_K)
        assert hs["K_upper"][1] == pytest.approx(QUARTZ_K)

    def test_shear_lower_bound_is_zero_for_a_fluid_suspension(self):
        hs = hashin_shtrikman(QUARTZ_K, QUARTZ_G, BRINE_K, 0.0, np.linspace(0.1, 0.9, 9))
        assert np.all(hs["G_lower"] == 0.0)
        assert np.all(hs["G_upper"] > 0.0)

    def test_fluid_suspension_lower_bound_equals_reuss(self):
        # With a zero-shear phase the Hashin-Shtrikman lower bound is the
        # Reuss bound; that coincidence is a correctness check, not an accident.
        f1 = np.array([0.7])
        hs = hashin_shtrikman(QUARTZ_K, QUARTZ_G, BRINE_K, 0.0, f1)
        assert hs["K_lower"][0] == pytest.approx(reuss([QUARTZ_K, BRINE_K], [0.7, 0.3]))

    def test_identical_phases_give_one_answer(self):
        hs = hashin_shtrikman(QUARTZ_K, QUARTZ_G, QUARTZ_K, QUARTZ_G, np.array([0.4]))
        assert hs["K_upper"][0] == pytest.approx(QUARTZ_K)
        assert hs["K_lower"][0] == pytest.approx(QUARTZ_K)


class TestFrameModels:
    def test_critical_porosity_end_points(self):
        K, G = critical_porosity_dry(QUARTZ_K, QUARTZ_G, [0.0, 0.2, 0.4], phi_c=0.4)
        assert K[0] == pytest.approx(QUARTZ_K)
        assert K[1] == pytest.approx(QUARTZ_K / 2)
        assert K[2] == pytest.approx(0.0)
        assert G[0] == pytest.approx(QUARTZ_G)

    def test_critical_porosity_clamps_above_phi_c(self):
        K, _ = critical_porosity_dry(QUARTZ_K, QUARTZ_G, [0.5], phi_c=0.4)
        assert K[0] == pytest.approx(0.0)

    def test_coordination_number_falls_with_porosity(self):
        n = coordination_number(np.array([0.2, 0.3, 0.4]))
        assert np.all(np.diff(n) < 0)
        assert float(coordination_number(0.36)) == pytest.approx(9.57, abs=0.01)

    def test_hertz_mindlin_is_soft_and_pressure_dependent(self):
        K_low, G_low = hertz_mindlin(QUARTZ_K, QUARTZ_G, pressure=5e6)
        K_high, G_high = hertz_mindlin(QUARTZ_K, QUARTZ_G, pressure=20e6)
        assert 0 < K_low < K_high < QUARTZ_K       # a pack is far softer than quartz
        assert 0 < G_low < G_high < QUARTZ_G
        # Hertz-Mindlin moduli scale with the cube root of effective pressure.
        assert K_high / K_low == pytest.approx(4.0 ** (1.0 / 3.0), rel=1e-6)

    @pytest.mark.parametrize("phi_c", [0.36, 0.40])
    def test_sand_models_bracket_and_meet_at_the_end_points(self, phi_c):
        phi = np.linspace(0.0, phi_c, 15)
        Ks, Gs = soft_sand_dry(QUARTZ_K, QUARTZ_G, phi, phi_c=phi_c)
        Kt, Gt = stiff_sand_dry(QUARTZ_K, QUARTZ_G, phi, phi_c=phi_c)
        K_hm, G_hm = hertz_mindlin(QUARTZ_K, QUARTZ_G, phi_c=phi_c)

        assert np.all(Kt >= Ks - 1e-9)             # stiff sand is never softer
        assert np.all(Gt >= Gs - 1e-9)
        assert Ks[0] == pytest.approx(QUARTZ_K)    # mineral point at zero porosity
        assert Kt[0] == pytest.approx(QUARTZ_K)
        assert Ks[-1] == pytest.approx(K_hm)       # the pack at critical porosity
        assert Kt[-1] == pytest.approx(K_hm)
        assert Gs[-1] == pytest.approx(G_hm)

    def test_sand_models_decrease_with_porosity(self):
        phi = np.linspace(0.01, 0.35, 20)
        Ks, _ = soft_sand_dry(QUARTZ_K, QUARTZ_G, phi)
        Kt, _ = stiff_sand_dry(QUARTZ_K, QUARTZ_G, phi)
        assert np.all(np.diff(Ks) < 0)
        assert np.all(np.diff(Kt) < 0)

    def test_frame_models_stay_non_negative(self):
        phi = np.linspace(0.0, 0.5, 30)
        for model in (soft_sand_dry, stiff_sand_dry):
            K, G = model(QUARTZ_K, QUARTZ_G, phi)
            assert np.all(K >= 0) and np.all(G >= 0)
            assert np.all(np.isfinite(K)) and np.all(np.isfinite(G))


class TestVelocityPorosityTrends:
    def test_wyllie_end_points(self):
        assert float(wyllie(0.0, 5500.0, 1500.0)) == pytest.approx(5500.0)
        assert float(wyllie(1.0, 5500.0, 1500.0)) == pytest.approx(1500.0)

    def test_raymer_end_points(self):
        assert float(raymer_hunt_gardner(0.0, 5500.0, 1500.0)) == pytest.approx(5500.0)
        assert float(raymer_hunt_gardner(1.0, 5500.0, 1500.0)) == pytest.approx(1500.0)

    def test_both_trends_fall_with_porosity(self):
        phi = np.linspace(0.0, 0.4, 20)
        assert np.all(np.diff(wyllie(phi, 5500.0, 1500.0)) < 0)
        assert np.all(np.diff(raymer_hunt_gardner(phi, 5500.0, 1500.0)) < 0)

    def test_raymer_is_faster_than_wyllie_at_moderate_porosity(self):
        phi = np.linspace(0.05, 0.35, 10)
        assert np.all(raymer_hunt_gardner(phi, 5500.0, 1500.0) > wyllie(phi, 5500.0, 1500.0))


class TestGardner:
    def test_round_trip(self):
        vp = np.array([2400.0, 3000.0, 4500.0])
        assert np.allclose(gardner_velocity(gardner_density(vp)), vp)

    def test_known_value(self):
        assert float(gardner_density(2400.0)) == pytest.approx(2.17, abs=0.01)

    def test_fit_recovers_planted_coefficients(self):
        vp = np.linspace(2000.0, 5000.0, 200)
        rho = 0.28 * vp ** 0.27
        a, b = fit_gardner(vp, rho)
        assert a == pytest.approx(0.28, rel=1e-6)
        assert b == pytest.approx(0.27, rel=1e-6)

    def test_fit_ignores_non_finite_and_degenerate_input(self):
        vp = np.array([2000.0, np.nan, 3000.0, -1.0])
        rho = np.array([2.1, 2.2, 2.3, 2.4])
        a, b = fit_gardner(vp, rho)
        assert np.isfinite(a) and np.isfinite(b)
        assert all(np.isnan(x) for x in fit_gardner([1000.0], [2.0]))


class TestVpVsTrends:
    def test_mudrock_line_known_value(self):
        assert float(castagna_mudrock(3000.0)) == pytest.approx(1413.9, abs=0.1)

    def test_mudrock_gives_a_shale_like_vpvs(self):
        vp = np.array([2400.0, 3000.0, 3600.0])
        assert np.all(vp / castagna_mudrock(vp) > 1.8)

    def test_greenberg_castagna_sandstone_is_faster_in_shear_than_shale(self):
        vp = 3000.0
        sand = float(greenberg_castagna(vp, {"sandstone": 1.0}))
        shale = float(greenberg_castagna(vp, {"shale": 1.0}))
        assert sand > shale

    def test_greenberg_castagna_mixture_lies_between_its_end_members(self):
        vp = 3500.0
        sand = float(greenberg_castagna(vp, {"sandstone": 1.0}))
        shale = float(greenberg_castagna(vp, {"shale": 1.0}))
        mix = float(greenberg_castagna(vp, {"sandstone": 0.5, "shale": 0.5}))
        assert min(sand, shale) < mix < max(sand, shale)

    def test_greenberg_castagna_normalises_fractions(self):
        a = float(greenberg_castagna(3000.0, {"sandstone": 2.0, "shale": 2.0}))
        b = float(greenberg_castagna(3000.0, {"sandstone": 0.5, "shale": 0.5}))
        assert a == pytest.approx(b)

    def test_greenberg_castagna_rejects_an_unknown_lithology(self):
        with pytest.raises(ValueError):
            greenberg_castagna(3000.0, {"granite": 1.0})

    def test_vectorised_over_a_log(self):
        vp = np.linspace(2000.0, 4500.0, 50)
        assert greenberg_castagna(vp).shape == vp.shape
        assert castagna_mudrock(vp).shape == vp.shape


class TestAgainstTheDemoWell:
    """The demo well's layers should land where rock physics says they do."""

    def test_gas_sand_sits_further_above_the_mudrock_line_than_the_shale(self):
        # A gas sand carries anomalously high Vs for its Vp, so it plots further
        # above the mudrock line than an encasing shale does.  The demo well's
        # layers were chosen for their AVO response, not calibrated to
        # Castagna, so neither one lands *on* the line - what matters for the
        # diagnostic is that the sand's shear excess is the larger of the two.
        shale_excess = 1200.0 - float(castagna_mudrock(2400.0))
        sand_excess = 1300.0 - float(castagna_mudrock(2100.0))
        assert shale_excess > 0
        assert sand_excess > shale_excess + 200.0

    def test_demo_layers_are_softer_than_their_mineral(self):
        layers = [(2400.0, 1200.0, 2.35), (2100.0, 1300.0, 2.10), (3000.0, 1800.0, 2.45)]
        for vp, vs, rho in layers:
            assert 0 < float(bulk_modulus(vp, vs, rho)) < QUARTZ_K
            assert 0 < float(shear_modulus(vs, rho)) < QUARTZ_G

    def test_demo_porosity_matches_the_density_mass_balance(self):
        """PHI in the demo LAS is derived from RHOB, so the two must agree."""
        import os

        from avo_qi.io.loader import read_well, standardise
        from avo_qi.sample_data.make_demo_well import density_porosity

        las = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "sample_data", "demo_well.las",
        )
        raw, units = read_well(las)
        df = standardise(raw, units=units).df

        cases = [                       # depth window, matrix rho, fluid rho
            ((2000, 2040), 2.58, 1.09),           # shale: clay, brine
            ((2040, 2070), 2.65, 0.25),           # gas sand: quartz, gas
            ((2090, 2105), 2.65, 1.09),           # brine sand: quartz, brine
        ]
        for (lo, hi), rho_ma, rho_fl in cases:
            window = df[(df["DEPTH"] >= lo) & (df["DEPTH"] < hi)]
            expected = density_porosity(window["RHOB"].mean(), rho_ma, rho_fl)
            assert window["PHI"].mean() == pytest.approx(expected, abs=0.01)

    def test_gas_sand_falls_below_a_brine_bound_but_not_a_gas_one(self):
        """The suspension bound is what flags the gas sand as anomalous."""
        K_gas_sand = float(bulk_modulus(2100.0, 1300.0, 2.10))
        phi = 0.229
        brine = hashin_shtrikman(QUARTZ_K, QUARTZ_G, BRINE_K, 0.0, np.array([1 - phi]))
        gas = hashin_shtrikman(QUARTZ_K, QUARTZ_G, FLUIDS["gas"][0], 0.0,
                               np.array([1 - phi]))
        assert K_gas_sand < brine["K_lower"][0]      # impossible if brine-filled
        assert gas["K_lower"][0] <= K_gas_sand <= gas["K_upper"][0]
