"""Tests for the per-sample forward model driven by VSH, PHIT and SW."""

from __future__ import annotations

import numpy as np
import pytest

from avo_qi.core.mixing import mineral_mix
from avo_qi.core.petro import (
    rock_physics_template,
    fluid_log,
    forward_model,
    mineral_log,
    porosity_provenance,
)
from avo_qi.core.rockphysics import FLUIDS, MINERALS

LAWS = ["voigt", "reuss", "hill", "hs_upper", "hs_lower", "hs_average"]
QUARTZ, CLAY = MINERALS["quartz"], MINERALS["clay"]


class TestMineralLog:
    @pytest.mark.parametrize("law", LAWS)
    def test_vectorised_bounds_agree_with_the_scalar_mixing_law(self, law):
        """The vectorised path must be the same physics, not a second version.

        ``mineral_log`` rewrites the bounds as array expressions so a page can
        redraw without looping over ten thousand samples.  That is only safe if
        it agrees with :func:`avo_qi.core.mixing.mineral_mix`, which is the
        tested scalar implementation.
        """
        vsh = np.array([0.0, 0.05, 0.12, 0.35, 0.6, 0.85, 1.0])
        got = mineral_log(vsh, law=law)
        want = np.array([
            mineral_mix([QUARTZ[0], CLAY[0]], [QUARTZ[1], CLAY[1]],
                        [QUARTZ[2], CLAY[2]], [1.0 - v, v], law)
            for v in vsh
        ])
        assert np.allclose(got.K, want[:, 0], rtol=1e-12, atol=1e-12)
        assert np.allclose(got.G, want[:, 1], rtol=1e-12, atol=1e-12)
        assert np.allclose(got.rho, want[:, 2], rtol=1e-12, atol=1e-12)

    def test_a_random_composition_sweep_also_agrees(self):
        rng = np.random.default_rng(3)
        vsh = rng.uniform(0.0, 1.0, 200)
        got = mineral_log(vsh, law="hs_average")
        want = np.array([
            mineral_mix([QUARTZ[0], CLAY[0]], [QUARTZ[1], CLAY[1]],
                        [QUARTZ[2], CLAY[2]], [1.0 - v, v], "hs_average")[0]
            for v in vsh
        ])
        assert np.allclose(got.K, want, rtol=1e-12, atol=1e-12)

    def test_the_end_members_are_the_pure_minerals(self):
        out = mineral_log([0.0, 1.0])
        assert out.K[0] == pytest.approx(QUARTZ[0])
        assert out.G[0] == pytest.approx(QUARTZ[1])
        assert out.rho[0] == pytest.approx(QUARTZ[2])
        assert out.K[1] == pytest.approx(CLAY[0])
        assert out.rho[1] == pytest.approx(CLAY[2])

    def test_gets_softer_and_denser_as_shale_comes_in(self):
        out = mineral_log(np.linspace(0.0, 1.0, 11))
        assert np.all(np.diff(out.K) < 0)
        assert np.all(np.diff(out.G) < 0)
        assert np.all(np.diff(out.rho) < 0)   # clay is lighter than quartz

    def test_the_matrix_can_be_a_blend_of_several_minerals(self):
        out = mineral_log([0.0], matrix={"quartz": 0.7, "calcite": 0.3})
        want = mineral_mix([QUARTZ[0], MINERALS["calcite"][0]],
                           [QUARTZ[1], MINERALS["calcite"][1]],
                           [QUARTZ[2], MINERALS["calcite"][2]],
                           [0.7, 0.3], "hill")
        assert out.K[0] == pytest.approx(want[0])

    def test_fractions_are_reported_and_sum_to_one(self):
        out = mineral_log([0.0, 0.3, 1.0], matrix={"quartz": 0.5, "feldspar": 0.5})
        total = sum(out.fractions.values())
        assert np.allclose(total, 1.0)
        assert out.fractions["clay"][1] == pytest.approx(0.3)
        assert out.fractions["quartz"][1] == pytest.approx(0.35)

    def test_a_missing_vsh_gives_a_missing_matrix_not_a_quartz_one(self):
        out = mineral_log([0.2, np.nan])
        assert np.isfinite(out.K[0])
        assert np.isnan(out.K[1]) and np.isnan(out.rho[1])

    def test_rejects_the_shale_mineral_appearing_twice(self):
        with pytest.raises(ValueError, match="cannot also be in the matrix"):
            mineral_log([0.2], matrix={"quartz": 0.5, "clay": 0.5}, shale="clay")

    def test_rejects_an_unknown_mineral(self):
        with pytest.raises(ValueError, match="unknown mineral"):
            mineral_log([0.2], matrix={"unobtainium": 1.0})


class TestFluidLog:
    def test_full_water_saturation_is_the_brine(self):
        out = fluid_log([1.0])
        assert out.K[0] == pytest.approx(FLUIDS["brine"][0])
        assert out.rho[0] == pytest.approx(FLUIDS["brine"][1])

    def test_no_water_is_the_hydrocarbon(self):
        out = fluid_log([0.0], hydrocarbon="gas")
        assert out.K[0] == pytest.approx(FLUIDS["gas"][0])
        assert out.rho[0] == pytest.approx(FLUIDS["gas"][1])

    def test_wood_is_the_soft_limit_and_patchy_the_stiff_one(self):
        sw = np.array([0.2, 0.5, 0.8])
        wood = fluid_log(sw, law="wood").K
        patchy = fluid_log(sw, law="patchy").K
        hill = fluid_log(sw, law="hill").K
        assert np.all(wood < hill)
        assert np.all(hill < patchy)

    def test_a_little_gas_collapses_the_modulus_under_wood(self):
        """The reason the mixing law matters at all."""
        brine = fluid_log([1.0], law="wood").K[0]
        almost = fluid_log([0.95], law="wood").K[0]
        assert almost < 0.35 * brine

    def test_brie_with_an_exponent_of_one_is_the_patchy_limit(self):
        sw = np.array([0.1, 0.4, 0.9])
        brie = fluid_log(sw, law="brie", brie_exponent=1.0).K
        assert np.allclose(brie, fluid_log(sw, law="patchy").K)

    def test_density_is_volume_weighted_whatever_the_law(self):
        sw = np.array([0.3, 0.7])
        for law in ("wood", "patchy", "brie", "hill"):
            out = fluid_log(sw, law=law, hydrocarbon="gas")
            want = sw * FLUIDS["brine"][1] + (1.0 - sw) * FLUIDS["gas"][1]
            assert np.allclose(out.rho, want)

    def test_accepts_properties_that_vary_down_the_well(self):
        """Batzle-Wang values along a pressure profile, rather than constants."""
        sw = np.full(5, 0.3)
        k_br = np.linspace(2.6, 3.1, 5)
        rho_br = np.linspace(1.02, 1.06, 5)
        out = fluid_log(sw, brine=(k_br, rho_br), hydrocarbon="gas")
        assert out.K.shape == (5,)
        assert np.all(np.diff(out.K) > 0)

    def test_rejects_an_unknown_law(self):
        with pytest.raises(ValueError, match="unknown fluid mixing law"):
            fluid_log([0.5], law="handwaving")


class TestForwardModel:
    #: Layers the demo well built from a Dvorkin-Nur frame at 20 MPa, with the
    #: velocities its generator recorded.  Those numbers were produced offline,
    #: outside this package, so reproducing them tests the whole chain —
    #: mineral mix, dry frame, Gassmann, velocities — against something that
    #: was not fitted to it.
    LAYERS = [
        ("brine sand", 0.26, "soft sand", (2830.2, 1498.9, 2.2444)),
        ("cemented sand", 0.18, "stiff sand", (4265.3, 2704.1, 2.3692)),
    ]

    @pytest.mark.parametrize("name, phi, frame, want", LAYERS,
                             ids=[row[0] for row in LAYERS])
    def test_reproduces_the_demo_wells_frame_built_layers(self, name, phi, frame, want):
        out = forward_model([0.0], [phi], [1.0], frame=frame, pressure=20e6)
        assert float(out["VP"][0]) == pytest.approx(want[0], abs=0.1)
        assert float(out["VS"][0]) == pytest.approx(want[1], abs=0.1)
        assert float(out["RHOB"][0]) == pytest.approx(want[2], abs=1e-4)

    def test_shale_softens_the_prediction(self):
        clean = forward_model([0.0], [0.26], [1.0], pressure=20e6)
        shaly = forward_model([0.30], [0.26], [1.0], pressure=20e6)
        assert float(shaly["VP"][0]) < float(clean["VP"][0])

    def test_gas_slows_the_rock_and_lightens_it(self):
        wet = forward_model([0.1], [0.26], [1.0], pressure=20e6)
        gassy = forward_model([0.1], [0.26], [0.2], pressure=20e6, hydrocarbon="gas")
        assert float(gassy["VP"][0]) < float(wet["VP"][0])
        assert float(gassy["RHOB"][0]) < float(wet["RHOB"][0])

    def test_a_stiff_frame_is_less_fluid_sensitive_than_a_soft_one(self):
        def drop(frame, phi):
            wet = forward_model([0.1], [phi], [1.0], frame=frame, pressure=20e6)
            dry = forward_model([0.1], [phi], [0.1], frame=frame, pressure=20e6)
            return 1.0 - float(dry["VP"][0]) / float(wet["VP"][0])
        assert drop("stiff sand", 0.18) < drop("soft sand", 0.18)

    def test_the_shear_modulus_is_the_dry_one(self):
        """Gassmann does not touch mu, so the model must not either."""
        out = forward_model([0.1], [0.24], [0.3], pressure=20e6)
        mu = out["RHOB"] * out["VS"] ** 2 * 1e-6
        assert float(mu[0]) == pytest.approx(float(out["G_dry"][0]), rel=1e-9)

    def test_an_impossible_porosity_is_reported_not_predicted(self):
        out = forward_model([0.1, 0.1], [0.26, 1.4], [1.0, 1.0], pressure=20e6)
        assert out["valid"][0] and not out["valid"][1]
        assert np.isnan(out["VP"][1])
        assert out["reasons"][1] == "porosity outside (0, 1)"

    def test_runs_over_a_whole_log_at_once(self):
        rng = np.random.default_rng(11)
        n = 5000
        out = forward_model(rng.uniform(0, 0.9, n), rng.uniform(0.05, 0.33, n),
                            rng.uniform(0, 1, n), pressure=20e6)
        assert out["VP"].shape == (n,)
        assert np.isfinite(out["VP"]).sum() > 0.95 * n

    def test_rejects_an_unknown_frame_model(self):
        with pytest.raises(ValueError, match="unknown frame model"):
            forward_model([0.1], [0.26], [1.0], frame="wishful thinking")


class TestPorosityProvenance:
    def test_detects_density_porosity_and_recovers_its_constants(self):
        rng = np.random.default_rng(0)
        rhob = rng.uniform(2.05, 2.45, 300)
        phid = (2.65 - rhob) / (2.65 - 1.09)
        out = porosity_provenance(phid, rhob)
        assert out.density_derived is True
        assert out.rho_matrix == pytest.approx(2.65, abs=1e-6)
        assert out.rho_fluid == pytest.approx(1.09, abs=1e-6)
        assert not out.rhob_check_is_meaningful

    def test_detects_it_whatever_constants_the_petrophysicist_used(self):
        """Fitting the densities rather than assuming them is the point."""
        rng = np.random.default_rng(1)
        rhob = rng.uniform(2.10, 2.50, 200)
        phid = (2.71 - rhob) / (2.71 - 0.85)     # calcite matrix, light fluid
        out = porosity_provenance(phid, rhob)
        assert out.density_derived is True
        assert out.rho_matrix == pytest.approx(2.71, abs=1e-6)

    def test_survives_the_rounding_in_a_las_file(self):
        rng = np.random.default_rng(2)
        rhob = np.round(rng.uniform(2.05, 2.45, 300), 4)
        phid = np.round((2.65 - rhob) / (2.65 - 1.09), 4)
        assert porosity_provenance(phid, rhob).density_derived is True

    def test_leaves_an_independent_porosity_alone(self):
        rng = np.random.default_rng(4)
        rhob = rng.uniform(2.05, 2.45, 300)
        phi = np.clip((2.65 - rhob) / 1.56 + rng.normal(0, 0.03, 300), 0.01, 0.4)
        out = porosity_provenance(phi, rhob)
        assert out.density_derived is False
        assert out.rhob_check_is_meaningful
        assert "meaningful" in out.note

    def test_says_it_cannot_tell_when_porosity_barely_varies(self):
        """A single-zone selection often has a constant porosity, and then
        there is no line to fit — which is not the same as "independent"."""
        rng = np.random.default_rng(5)
        out = porosity_provenance(np.full(50, 0.26), rng.uniform(2.2, 2.3, 50))
        assert out.density_derived is None
        assert out.rhob_check_is_meaningful
        assert "too little" in out.note

    def test_says_it_cannot_tell_from_two_samples(self):
        out = porosity_provenance([0.1, 0.3], [2.4, 2.2])
        assert out.density_derived is None
        assert "Too few samples" in out.note

    def test_a_tight_but_unphysical_line_is_not_density_porosity(self):
        phi = np.linspace(0.05, 0.35, 100)
        rhob = 5.0 - 2.0 * phi          # a perfect fit to an impossible matrix
        out = porosity_provenance(phi, rhob)
        assert out.density_derived is False
        assert "physically sensible" in out.note

    def test_the_source_mnemonic_corroborates_but_does_not_decide(self):
        rng = np.random.default_rng(6)
        rhob = rng.uniform(2.05, 2.45, 200)
        phid = (2.65 - rhob) / 1.56
        assert "PHID" in porosity_provenance(phid, rhob, mnemonic="PHID").note
        # An independent log keeps its verdict even with a suggestive mnemonic.
        phi = np.clip(phid + rng.normal(0, 0.03, 200), 0.01, 0.4)
        assert porosity_provenance(phi, rhob, mnemonic="PHID").density_derived is False

    def test_ignores_non_finite_samples(self):
        rng = np.random.default_rng(8)
        rhob = rng.uniform(2.05, 2.45, 100)
        phid = (2.65 - rhob) / 1.56
        phid[::10] = np.nan
        assert porosity_provenance(phid, rhob).density_derived is True

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            porosity_provenance([0.1, 0.2], [2.3])


class TestRockPhysicsTemplate:
    """A template turns the crossplot axes into porosity and saturation.

    These pin the four things an interpreter reads off one, all of which are
    directions rather than magnitudes — the magnitudes depend on the model and
    are the interpreter's to argue with.
    """

    PHI = np.linspace(0.02, 0.36, 12)
    SW = np.linspace(1.0, 0.0, 5)

    @classmethod
    @pytest.fixture(scope="class")
    def template(cls):
        return rock_physics_template(cls.PHI, cls.SW, vsh=0.0,
                                     frame="soft sand")

    def test_it_is_shaped_by_the_grid_it_was_given(self, template):
        assert template["AI"].shape == (self.PHI.size, self.SW.size)
        assert template["vpvs"].shape == (self.PHI.size, self.SW.size)
        assert np.array_equal(template["porosity"], self.PHI)
        assert np.array_equal(template["saturation"], self.SW)
        assert template["valid"].all()
        assert np.isfinite(template["AI"]).all()

    def test_porosity_softens_the_rock(self, template):
        """More pore space, lower impedance — at every saturation."""
        for j in range(self.SW.size):
            assert np.all(np.diff(template["AI"][:, j]) < 0)

    def test_gas_drops_the_impedance_and_the_vpvs(self, template):
        """The whole reason to look at this crossplot. Sw runs 1 -> 0, so
        moving along a row is filling the pore with hydrocarbon."""
        for i in range(self.PHI.size):
            assert template["AI"][i, -1] < template["AI"][i, 0]
            assert template["vpvs"][i, -1] < template["vpvs"][i, 0]

    def test_gas_leaves_the_shear_modulus_alone(self, template):
        """Gassmann does not touch it, so Vs moves only because the rock got
        lighter — which means Vs goes *up* with gas even as Vp goes down."""
        for i in range(self.PHI.size):
            assert template["VS"][i, -1] > template["VS"][i, 0]
            assert template["VP"][i, -1] < template["VP"][i, 0]
            assert template["RHOB"][i, -1] < template["RHOB"][i, 0]

    def test_a_shalier_template_is_a_different_curve(self):
        clean = rock_physics_template(self.PHI, [1.0], vsh=0.0)
        shaly = rock_physics_template(self.PHI, [1.0], vsh=0.6)
        # Shale is softer and its Vp/Vs is higher; a template drawn for clean
        # sand says nothing about a shaly point, which is why vsh is explicit.
        assert np.all(shaly["vpvs"][:, 0] > clean["vpvs"][:, 0])

    def test_a_stiffer_frame_lifts_the_whole_template(self):
        soft = rock_physics_template(self.PHI, [1.0], frame="soft sand")
        stiff = rock_physics_template(self.PHI, [1.0], frame="stiff sand")
        assert np.all(stiff["AI"][:, 0] >= soft["AI"][:, 0] - 1e-9)
        assert np.any(stiff["AI"][:, 0] > soft["AI"][:, 0])

    def test_zero_porosity_is_reported_invalid_rather_than_guessed(self):
        """Gassmann divides by porosity. At phi = 0 there is no pore to
        saturate, and the mask says so instead of the grid carrying a made-up
        mineral point."""
        template = rock_physics_template([0.0, 0.1, 0.2], [1.0])
        assert not template["valid"][0, 0]
        assert template["valid"][1:, 0].all()
        assert not np.isfinite(template["AI"][0, 0])

    def test_it_is_the_same_physics_as_the_per_sample_forward_model(self):
        """The template must not be a second implementation: a well that misses
        it should mean the model is wrong, not that two code paths disagree."""
        from avo_qi.core.attributes import acoustic_impedance

        phi, sw = 0.25, 0.3
        template = rock_physics_template([phi], [sw], vsh=0.1,
                                         frame="stiff sand", phi_c=0.38)
        direct = forward_model([0.1], [phi], [sw], frame="stiff sand",
                               phi_c=0.38)
        assert template["VP"][0, 0] == pytest.approx(float(direct["VP"][0]))
        assert template["VS"][0, 0] == pytest.approx(float(direct["VS"][0]))
        assert template["AI"][0, 0] == pytest.approx(
            float(acoustic_impedance(direct["VP"], direct["RHOB"])[0]))
