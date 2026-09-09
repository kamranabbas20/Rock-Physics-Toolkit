"""Rüger's VTI reflectivity, and what it does to a gradient.

The properties worth holding are the ones that make the module trustworthy
rather than merely present: that switching it on with no anisotropy changes
nothing, that Thomsen's delta lands in the gradient and nowhere else, and that
the effect is a *contrast* — two anisotropic shales against each other look
isotropic, which is the mistake a fixed "shale is anisotropic" flag would make.
"""

from __future__ import annotations

import numpy as np
import pytest

from avo_qi.core.anisotropy import (
    ISOTROPIC,
    LITERATURE_SHALES,
    Thomsen,
    fitted_gradient_shift,
    gradient_shift,
    ruger_terms,
    ruger_vti_rpp,
    thomsen_from_vsh,
    thomsen_logs_from_vsh,
)
from avo_qi.core.avo import shuey_fit
from avo_qi.core.reflectivity import aki_richards_rpp

# A shale over a gas sand: the interface the whole toolkit is pointed at.
SHALE = (3000.0, 1500.0, 2.40)
SAND = (2900.0, 1800.0, 2.15)
ANGLES = np.arange(0.0, 41.0, 2.5)


class TestTheIsotropicLimit:
    """Turning anisotropy on with no anisotropy must be a no-op. This is the
    first thing anyone will check, and the thing that makes the rest safe to
    switch on by default."""

    def test_it_matches_aki_richards(self):
        mine = ruger_vti_rpp(*SHALE, *SAND, ANGLES)
        theirs = aki_richards_rpp(*SHALE, *SAND, ANGLES)
        # The two differ only in dZ/Z versus da/a + drho/rho, which is second
        # order in the contrasts: a few parts in ten thousand here.
        assert np.max(np.abs(mine - theirs)) < 5e-4

    def test_the_difference_is_third_order_in_the_contrast(self):
        """Not just small — small *because* the contrast is, and smaller than
        it first looks.

        Measured at normal incidence, where the only difference between the
        two is ``dZ/Z`` against ``da/a + drho/rho``. Both are second-order
        accurate stand-ins for ``d(ln Z)``, so they agree through second order
        and differ at *third*: halving the contrast has to divide the gap by
        eight, not four.
        """
        def gap(scale):
            lower = tuple(u + scale * (l - u) for u, l in zip(SHALE, SAND))
            return abs(float(ruger_vti_rpp(*SHALE, *lower, [0.0])[0])
                       - float(aki_richards_rpp(*SHALE, *lower, [0.0])[0]))

        # Converging on 8 from above as the contrast shrinks.
        assert gap(1.0) / gap(0.5) == pytest.approx(8.0, rel=0.07)
        assert gap(0.25) / gap(0.125) == pytest.approx(8.0, rel=0.03)

    def test_none_and_an_explicit_zero_medium_agree(self):
        assert ruger_terms(*SHALE, *SAND, upper=ISOTROPIC, lower=ISOTROPIC) \
            == ruger_terms(*SHALE, *SAND)


class TestWhereTheAnisotropyLands:
    def test_delta_moves_the_gradient_and_only_the_gradient(self):
        """The claim the module is built on: in the three-term form, delta
        appears in B alone."""
        plain = ruger_terms(*SHALE, *SAND)
        shaley = ruger_terms(*SHALE, *SAND, upper=Thomsen(delta=0.10))

        assert shaley[0] == pytest.approx(plain[0])           # A untouched
        assert shaley[2] == pytest.approx(plain[2])           # C untouched
        assert shaley[1] - plain[1] == pytest.approx(-0.05)   # B by d(delta)/2

    def test_epsilon_moves_the_far_term_and_only_that(self):
        plain = ruger_terms(*SHALE, *SAND)
        shaley = ruger_terms(*SHALE, *SAND, upper=Thomsen(epsilon=0.20))

        assert shaley[0] == pytest.approx(plain[0])
        assert shaley[1] == pytest.approx(plain[1])
        assert shaley[2] - plain[2] == pytest.approx(-0.10)

    def test_gamma_does_nothing_to_a_pp_reflection(self):
        """Carried on the dataclass for completeness, but P-P cannot see it.
        Worth pinning so nobody wires it in by accident."""
        assert ruger_terms(*SHALE, *SAND, upper=Thomsen(gamma=0.3)) \
            == ruger_terms(*SHALE, *SAND)

    def test_normal_incidence_is_untouched(self):
        strong = LITERATURE_SHALES["strong"]
        assert ruger_vti_rpp(*SHALE, *SAND, [0.0], upper=strong)[0] == \
            pytest.approx(ruger_vti_rpp(*SHALE, *SAND, [0.0])[0])


class TestItIsAContrastNotAProperty:
    """The mistake this guards against is treating anisotropy as something a
    rock *has* rather than something an interface *sees*."""

    def test_two_identical_anisotropic_media_look_isotropic(self):
        both = LITERATURE_SHALES["strong"]
        assert ruger_terms(*SHALE, *SAND, upper=both, lower=both) == \
            pytest.approx(ruger_terms(*SHALE, *SAND))

    def test_the_sign_follows_lower_minus_upper(self):
        """An isotropic sand under a positive-delta shale pushes the gradient
        down — towards Class III — not up."""
        assert gradient_shift(upper=Thomsen(delta=0.1)) < 0
        assert gradient_shift(lower=Thomsen(delta=0.1)) > 0

    def test_the_shift_is_exactly_half_the_delta_contrast(self):
        shift = gradient_shift(upper=Thomsen(delta=0.12), lower=Thomsen(delta=0.02))
        assert shift == pytest.approx(-0.05)

    def test_it_agrees_with_a_fitted_gradient(self):
        """The reported shift and the gradient a Shuey fit actually recovers
        have to be the same number, or the panel would report one thing and
        the classification would use another."""
        shale = LITERATURE_SHALES["moderate"]
        plain = shuey_fit(ruger_vti_rpp(*SHALE, *SAND, ANGLES), ANGLES)
        vti = shuey_fit(ruger_vti_rpp(*SHALE, *SAND, ANGLES, upper=shale), ANGLES)

        assert vti[1] - plain[1] == pytest.approx(
            fitted_gradient_shift(upper=shale, angles=ANGLES), abs=1e-12)


class TestTheEpsilonLeak:
    """The finding that made ``fitted_gradient_shift`` necessary.

    ``d(delta)/2`` is the whole of the anisotropy in the three-term gradient,
    and it is the number every textbook quotes. It is not the number the class
    is made of. A two-term fit cannot separate ``sin^2 tan^2`` from ``sin^2``
    over a finite angle range, so the epsilon contrast leaks into the fitted
    gradient too — and on a 40 degree gather it roughly doubles the effect.
    Quoting the textbook number to a user would understate it twofold.
    """

    SHALE_ANISO = LITERATURE_SHALES["moderate"]

    def test_the_fit_sees_more_than_the_textbook_shift(self):
        textbook = gradient_shift(upper=self.SHALE_ANISO)
        fitted = fitted_gradient_shift(upper=self.SHALE_ANISO, angles=ANGLES)
        assert abs(fitted) > abs(textbook)
        assert fitted / textbook == pytest.approx(2.11, rel=0.02)

    def test_the_leak_grows_with_the_angle_range(self):
        """Which is why the angle range is an argument and not a constant: the
        same rock reads differently on a near gather and a full one."""
        factors = [fitted_gradient_shift(upper=self.SHALE_ANISO,
                                         angles=np.arange(0.0, hi + 1, 2.5))
                   / gradient_shift(upper=self.SHALE_ANISO)
                   for hi in (20, 30, 40, 50)]
        assert factors == sorted(factors)
        assert factors[0] == pytest.approx(1.23, rel=0.02)
        assert factors[-1] == pytest.approx(3.10, rel=0.02)

    def test_with_no_epsilon_contrast_the_two_agree(self):
        """The leak is epsilon's doing, and nothing else's."""
        only_delta = Thomsen(delta=0.08)
        assert fitted_gradient_shift(upper=only_delta, angles=ANGLES) == \
            pytest.approx(gradient_shift(upper=only_delta))

    def test_a_gradient_needs_at_least_two_angles(self):
        assert np.isnan(fitted_gradient_shift(upper=self.SHALE_ANISO,
                                              angles=[15.0]))
        assert np.isnan(fitted_gradient_shift(upper=self.SHALE_ANISO,
                                              angles=[15.0, 15.0]))
        assert np.isnan(fitted_gradient_shift(upper=self.SHALE_ANISO))

    def test_no_anisotropy_is_no_shift_however_wide_the_gather(self):
        assert fitted_gradient_shift(angles=ANGLES) == 0.0


class TestItCanChangeAClass:
    """The reason for the module. If a literature-range shale could not move a
    label, this would be a curiosity rather than a gap."""

    def test_a_moderate_shale_moves_the_gradient_by_more_than_a_class_band(self):
        from avo_qi.core.avo import classify

        shale = LITERATURE_SHALES["moderate"]
        a, b, _ = ruger_terms(*SHALE, *SAND)
        a_vti, b_vti, _ = ruger_terms(*SHALE, *SAND, upper=shale)

        assert abs(b_vti - b) == pytest.approx(0.04)
        # ...which is larger than the default near-zero intercept band.
        assert abs(b_vti - b) > 0.02
        # On this interface both still read III, so the test says what it can:
        # the shift is real and large, not that it always flips a label.
        assert classify(a, b) == classify(a_vti, b_vti) == "III"

    def test_an_interface_near_the_boundary_does_flip(self):
        """Constructed rather than found: a Class IV gradient parked just
        above zero, which an anisotropic seal pushes below it.

        The label goes from *dims with offset* to *brightens with offset* on
        the shale's fabric alone — same rock, same fluid.
        """
        from avo_qi.core.avo import classify

        a, b = -0.05, 0.03
        assert classify(a, b) == "IV"
        shifted = b + fitted_gradient_shift(
            upper=LITERATURE_SHALES["moderate"], angles=ANGLES)
        assert classify(a, shifted) == "III"


class TestCarryingItDownAWell:
    def test_shale_volume_interpolates_between_the_end_members(self):
        shale = Thomsen(epsilon=0.2, delta=0.1, gamma=0.15)
        assert thomsen_from_vsh(1.0, shale) == shale
        assert thomsen_from_vsh(0.0, shale) == ISOTROPIC
        assert thomsen_from_vsh(0.5, shale).delta == pytest.approx(0.05)

    def test_an_unknown_shale_volume_is_unknown_not_isotropic(self):
        """A missing curve must not read as "no fabric" — that would silently
        report an isotropic answer as a measured one."""
        assert thomsen_from_vsh(np.nan, LITERATURE_SHALES["moderate"]) is None

    def test_a_shale_volume_outside_the_unit_interval_is_clipped(self):
        shale = Thomsen(epsilon=0.2, delta=0.1)
        assert thomsen_from_vsh(1.4, shale) == shale
        assert thomsen_from_vsh(-0.3, shale) == ISOTROPIC

    def test_the_log_form_matches_the_scalar_one(self):
        shale = Thomsen(epsilon=0.2, delta=0.1, gamma=0.05)
        vsh = np.array([0.0, 0.25, 0.5, 1.0])
        logs = thomsen_logs_from_vsh(vsh, shale)
        assert np.allclose(logs["delta"],
                           [thomsen_from_vsh(v, shale).delta for v in vsh])

    def test_the_log_form_keeps_a_gap_a_gap(self):
        logs = thomsen_logs_from_vsh([0.5, np.nan], LITERATURE_SHALES["moderate"])
        assert np.isfinite(logs["delta"][0]) and np.isnan(logs["delta"][1])


class TestItRefusesToInvent:
    def test_the_defaults_are_isotropic_everywhere(self):
        assert Thomsen().is_isotropic
        assert ISOTROPIC.is_isotropic
        assert gradient_shift() == 0.0

    def test_the_literature_values_are_a_range_not_a_default(self):
        deltas = [t.delta for t in LITERATURE_SHALES.values()]
        assert len(set(deltas)) == len(deltas), "a single value would read as a default"
        assert min(deltas) > 0 and max(deltas) < 0.3

    def test_a_non_finite_layer_gives_nothing_rather_than_a_number(self):
        assert all(np.isnan(v) for v in
                   ruger_terms(np.nan, 1500, 2.4, *SAND))

    def test_a_nonsense_anisotropy_is_refused(self):
        with pytest.raises(ValueError):
            Thomsen(delta=float("nan"))
        with pytest.raises(TypeError):
            ruger_terms(*SHALE, *SAND, upper="moderate")


class TestDegenerateMedia:
    def test_a_fluid_layer_does_not_divide_by_zero(self):
        terms = ruger_terms(1500.0, 0.0, 1.03, *SAND)
        assert all(np.isfinite(v) for v in terms)

    def test_a_mapping_is_accepted_in_place_of_a_thomsen(self):
        assert ruger_terms(*SHALE, *SAND, upper={"delta": 0.1}) == \
            ruger_terms(*SHALE, *SAND, upper=Thomsen(delta=0.1))


class TestWalkingItDownALog:
    """``reflectivity_series`` is where the rest of the toolkit meets this, so
    the wiring gets its own tests rather than being assumed from the scalar
    ones."""

    VP = np.array([3000.0, 2900.0, 3100.0, 3050.0])
    VS = np.array([1500.0, 1800.0, 1600.0, 1580.0])
    RHO = np.array([2.40, 2.15, 2.35, 2.36])
    VSH = np.array([0.90, 0.10, 0.85, 0.88])

    def series(self, **kwargs):
        from avo_qi.core.reflectivity import reflectivity_series

        return reflectivity_series(self.VP, self.VS, self.RHO, ANGLES, **kwargs)

    def test_the_method_is_reachable_by_name(self):
        for alias in ("ruger", "rueger", "vti", "ruger_vti", "anisotropic"):
            assert np.allclose(self.series(method=alias),
                               self.series(method="ruger"))

    def test_without_anisotropy_it_is_the_linear_method(self):
        """Selecting the anisotropic method and supplying no anisotropy must
        not change the answer — otherwise the method choice alone would move
        classes, and nobody could tell which change was which."""
        assert np.max(np.abs(self.series(method="ruger")
                             - self.series(method="aki_richards"))) < 5e-4

    def test_shale_volume_drives_a_real_difference(self):
        from avo_qi.core.anisotropy import thomsen_logs_from_vsh

        logs = thomsen_logs_from_vsh(self.VSH, LITERATURE_SHALES["moderate"])
        gap = np.max(np.abs(self.series(method="ruger", anisotropy=logs)
                            - self.series(method="ruger")))
        # Two orders of magnitude larger than the isotropic-limit error above,
        # so the effect cannot be confused with the approximation.
        assert gap > 0.02

    def test_the_last_row_stays_zero(self):
        from avo_qi.core.anisotropy import thomsen_logs_from_vsh

        logs = thomsen_logs_from_vsh(self.VSH, LITERATURE_SHALES["strong"])
        assert np.all(self.series(method="ruger", anisotropy=logs)[-1] == 0)

    def test_an_unknown_sample_is_read_as_isotropic_not_dropped(self):
        """Blanking the interface would silently remove a reflector from the
        section, which is a worse failure than under-reporting its shift."""
        logs = {"epsilon": np.array([0.15, np.nan, 0.15, 0.15]),
                "delta": np.array([0.08, np.nan, 0.08, 0.08])}
        rc = self.series(method="ruger", anisotropy=logs)
        assert np.all(np.isfinite(rc))
        assert np.any(rc[1] != 0.0)

    def test_a_scalar_anisotropy_is_broadcast(self):
        both = self.series(method="ruger",
                           anisotropy={"epsilon": 0.15, "delta": 0.08})
        # Uniform anisotropy is no contrast at all, so it changes nothing.
        assert np.allclose(both, self.series(method="ruger"))

    def test_the_other_methods_ignore_it(self):
        from avo_qi.core.anisotropy import thomsen_logs_from_vsh

        logs = thomsen_logs_from_vsh(self.VSH, LITERATURE_SHALES["strong"])
        for method in ("zoeppritz", "aki_richards"):
            assert np.allclose(self.series(method=method, anisotropy=logs),
                               self.series(method=method))

    def test_a_malformed_anisotropy_is_refused(self):
        with pytest.raises(ValueError):
            self.series(method="ruger", anisotropy={"epsilon": [0.1]})
        with pytest.raises(ValueError):
            self.series(method="ruger",
                        anisotropy={"epsilon": np.zeros(3), "delta": np.zeros(3)})
