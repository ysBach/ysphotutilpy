"""
Tests for ysphotutilpy.polarimetry module.

All expected values are analytically derived from polarimetry formulas.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ..polarimetry import (
    calc_qu_4set,
    correct_eff,
    correct_off,
    correct_pa,
    calc_pol,
    calc_pol_r,
    calc_stokes,
)


# =============================================================================
# Tests for calc_qu_4set
# =============================================================================
class TestCalcQu4Set:
    """Tests for calc_qu_4set function (raw q, u calculation)."""

    def test_unpolarized_source(self, polarimetry_unpolarized):
        """
        Test unpolarized source gives q = u = 0.

        When all o-ray and e-ray intensities are equal:
        r_q = sqrt((e_000/o_000)/(e_450/o_450)) = sqrt(1/1) = 1
        q = (r_q - 1)/(r_q + 1) = 0

        Same for u.
        """
        q, u, dq, du = calc_qu_4set(**polarimetry_unpolarized)

        assert_allclose(q, 0.0, atol=1e-10)
        assert_allclose(u, 0.0, atol=1e-10)

    def test_q_only_polarization(self, polarimetry_q_only):
        """
        Test source with q != 0, u = 0.

        e_000/o_000 = 2, e_450/o_450 = 0.5
        r_q = sqrt(2 / 0.5) = sqrt(4) = 2
        q = (2 - 1)/(2 + 1) = 1/3

        e_225/o_225 = e_675/o_675 = 1
        r_u = sqrt(1/1) = 1
        u = 0
        """
        q, u, dq, du = calc_qu_4set(**polarimetry_q_only)

        expected_q = 1.0 / 3.0
        assert_allclose(q, expected_q, rtol=1e-10)
        assert_allclose(u, 0.0, atol=1e-10)

    def test_qu_known_values(self, polarimetry_qu_known):
        """
        Test source with known q and u values.

        Both q and u should be 1/3 based on fixture setup.
        """
        q, u, dq, du = calc_qu_4set(**polarimetry_qu_known)

        expected = 1.0 / 3.0
        assert_allclose(q, expected, rtol=1e-10)
        assert_allclose(u, expected, rtol=1e-10)

    def test_calc_qu_4set_output_percentage(self, polarimetry_q_only):
        """Test output in percentage."""
        q, u, dq, du = calc_qu_4set(**polarimetry_q_only, out_pct=True)

        # q = 1/3 -> 33.333...%
        expected_q_pct = 100.0 / 3.0
        assert_allclose(q, expected_q_pct, rtol=1e-10)

    def test_calc_qu_4set_eminuso_false(self, polarimetry_q_only):
        """
        Test eminuso=False gives opposite sign.

        When eminuso=False, sign flips: q = -(r_q - 1)/(r_q + 1)
        """
        q_eminuso, _, _, _ = calc_qu_4set(**polarimetry_q_only, eminuso=True)
        q_ominuse, _, _, _ = calc_qu_4set(**polarimetry_q_only, eminuso=False)

        assert_allclose(q_ominuse, -q_eminuso, rtol=1e-10)


# =============================================================================
# Tests for correct_eff
# =============================================================================
class TestCorrectEff:
    """Tests for correct_eff function (polarimetric efficiency correction)."""

    def test_correct_eff_unity_efficiency(self):
        """
        Test with p_eff = 1 (no correction needed).

        q_eff = q / 1 = q
        """
        q, u = 0.1, 0.2
        q_eff, u_eff, dq_eff, du_eff = correct_eff(q, u, p_eff=1.0)

        assert_allclose(q_eff, q, rtol=1e-10)
        assert_allclose(u_eff, u, rtol=1e-10)

    def test_correct_eff_90_percent(self):
        """
        Test with p_eff = 0.9.

        q = 0.09 -> q_eff = 0.09 / 0.9 = 0.1
        u = 0.18 -> u_eff = 0.18 / 0.9 = 0.2
        """
        q, u = 0.09, 0.18
        q_eff, u_eff, dq_eff, du_eff = correct_eff(q, u, p_eff=0.9)

        assert_allclose(q_eff, 0.1, rtol=1e-10)
        assert_allclose(u_eff, 0.2, rtol=1e-10)

    def test_correct_eff_percentage_io(self):
        """Test with percentage input/output."""
        # 9% measured, 90% efficiency -> 10% true
        q, u = 9.0, 18.0  # percent
        q_eff, u_eff, dq_eff, du_eff = correct_eff(
            q, u, p_eff=90.0, in_pct=True, out_pct=True
        )

        assert_allclose(q_eff, 10.0, rtol=1e-10)
        assert_allclose(u_eff, 20.0, rtol=1e-10)


# =============================================================================
# Tests for correct_off
# =============================================================================
class TestCorrectOff:
    """Tests for correct_off function (instrumental offset correction)."""

    def test_correct_off_no_offset(self):
        """Test with zero offset (no correction)."""
        q, u = 0.1, 0.2
        q_rot, u_rot, dq_rot, du_rot = correct_off(
            q, u, q_off=0, u_off=0, rot_q=0, rot_u=0
        )

        assert_allclose(q_rot, q, rtol=1e-10)
        assert_allclose(u_rot, u, rtol=1e-10)

    def test_correct_off_with_offset(self):
        """
        Test with non-zero offset.

        q_rot = q - (cos(2*rot_q)*q_off - sin(2*rot_q)*u_off)

        For rot_q = rot_u = 0:
        q_rot = q - q_off
        u_rot = u - u_off
        """
        q, u = 0.15, 0.25
        q_off, u_off = 0.05, 0.05

        q_rot, u_rot, dq_rot, du_rot = correct_off(
            q, u, q_off=q_off, u_off=u_off, rot_q=0, rot_u=0
        )

        assert_allclose(q_rot, 0.10, rtol=1e-10)
        assert_allclose(u_rot, 0.20, rtol=1e-10)


# =============================================================================
# Tests for correct_pa
# =============================================================================
class TestCorrectPa:
    """Tests for correct_pa function (position angle correction)."""

    def test_correct_pa_no_rotation(self):
        """Test with zero rotation (no correction)."""
        q, u = 0.1, 0.2
        q_inst, u_inst, dq_inst, du_inst = correct_pa(
            q, u, pa_off=0, pa_obs=0
        )

        assert_allclose(q_inst, q, rtol=1e-10)
        assert_allclose(u_inst, u, rtol=1e-10)

    def test_correct_pa_90_degree_rotation(self):
        """
        Test with 90 degree rotation.

        For theta = 90 deg = pi/2 rad:
        cos(2*theta) = cos(pi) = -1
        sin(2*theta) = sin(pi) = 0

        q_inst = -q
        u_inst = -u
        """
        q, u = 0.1, 0.2
        q_inst, u_inst, dq_inst, du_inst = correct_pa(
            q, u, pa_off=np.pi/2, pa_obs=0, in_deg=False
        )

        assert_allclose(q_inst, -q, rtol=1e-10)
        assert_allclose(u_inst, -u, rtol=1e-10)

    def test_correct_pa_45_degree_rotation(self):
        """
        Test with 45 degree rotation.

        For theta = 45 deg = pi/4 rad:
        cos(2*theta) = cos(pi/2) = 0
        sin(2*theta) = sin(pi/2) = 1

        q_inst = 0*q + 1*u = u
        u_inst = -1*q + 0*u = -q
        """
        q, u = 0.1, 0.2
        q_inst, u_inst, dq_inst, du_inst = correct_pa(
            q, u, pa_off=np.pi/4, pa_obs=0, in_deg=False
        )

        assert_allclose(q_inst, u, rtol=1e-10)
        assert_allclose(u_inst, -q, rtol=1e-10)


# =============================================================================
# Tests for calc_pol
# =============================================================================
class TestCalcPol:
    """Tests for calc_pol function (polarization degree and angle)."""

    def test_calc_pol_3_4_5_triangle(self):
        """
        Test polarization with 3-4-5 Pythagorean triple.

        q = 0.6, u = 0.8
        pol = sqrt(0.36 + 0.64) = sqrt(1) = 1.0
        theta = 0.5 * arctan2(0.8, 0.6) = 0.5 * arctan(4/3)
              = 0.5 * 0.9273 rad = 0.4636 rad = 26.565 deg
        """
        q, u = 0.6, 0.8
        pol, thp, dpol, dthp = calc_pol(q, u)

        assert_allclose(pol, 1.0, rtol=1e-10)
        expected_theta = 0.5 * np.arctan2(0.8, 0.6)
        assert_allclose(thp, expected_theta, rtol=1e-10)

    def test_calc_pol_pure_q(self):
        """
        Test polarization with u = 0.

        q = 0.5, u = 0
        pol = 0.5
        theta = 0.5 * arctan2(0, 0.5) = 0
        """
        q, u = 0.5, 0.0
        pol, thp, dpol, dthp = calc_pol(q, u)

        assert_allclose(pol, 0.5, rtol=1e-10)
        assert_allclose(thp, 0.0, atol=1e-10)

    def test_calc_pol_pure_u(self):
        """
        Test polarization with q = 0.

        q = 0, u = 0.5
        pol = 0.5
        theta = 0.5 * arctan2(0.5, 0) = 0.5 * (pi/2) = pi/4 = 45 deg
        """
        q, u = 0.0, 0.5
        pol, thp, dpol, dthp = calc_pol(q, u)

        assert_allclose(pol, 0.5, rtol=1e-10)
        assert_allclose(thp, np.pi/4, rtol=1e-10)

    def test_calc_pol_output_percentage(self):
        """Test output in percentage."""
        q, u = 0.06, 0.08  # natural units
        pol, thp, dpol, dthp = calc_pol(q, u, out_pct=True)

        # pol = 0.1 -> 10%
        assert_allclose(pol, 10.0, rtol=1e-10)

    def test_calc_pol_output_degrees(self):
        """Test angle output in degrees."""
        q, u = 0.6, 0.8
        pol, thp, dpol, dthp = calc_pol(q, u, out_deg=True)

        # theta = 0.5 * arctan2(0.8, 0.6) rad -> 26.565 deg
        expected_theta_deg = np.rad2deg(0.5 * np.arctan2(0.8, 0.6))
        assert_allclose(thp, expected_theta_deg, rtol=1e-5)


# =============================================================================
# Tests for calc_pol_r
# =============================================================================
class TestCalcPolR:
    """Tests for calc_pol_r function (proper polarization following Lyot)."""

    def test_calc_pol_r_aligned(self):
        """
        Test proper polarization when aligned with scattering plane.

        pol = 0.1, theta = 0
        suntargetpa = 0

        theta_r = theta + suntargetpa = 0
        polr = pol * cos(2 * theta_r) = pol * cos(0) = pol = 0.1
        """
        pol, thp = 0.1, 0.0
        polr, thr, dpolr, dthr = calc_pol_r(pol, thp, suntargetpa=0)

        assert_allclose(polr, 0.1, rtol=1e-10)
        assert_allclose(thr, 0.0, atol=1e-10)

    def test_calc_pol_r_perpendicular(self):
        """
        Test proper polarization when perpendicular to scattering plane.

        pol = 0.1, theta = pi/4
        suntargetpa = 0

        theta_r = pi/4
        polr = pol * cos(2 * pi/4) = pol * cos(pi/2) = 0
        """
        pol, thp = 0.1, np.pi/4
        polr, thr, dpolr, dthr = calc_pol_r(pol, thp, suntargetpa=0)

        assert_allclose(polr, 0.0, atol=1e-10)

    def test_calc_pol_r_negative(self):
        """
        Test proper polarization can be negative.

        pol = 0.1, theta = pi/2
        suntargetpa = 0

        theta_r = pi/2
        polr = pol * cos(2 * pi/2) = pol * cos(pi) = -pol = -0.1
        """
        pol, thp = 0.1, np.pi/2
        polr, thr, dpolr, dthr = calc_pol_r(pol, thp, suntargetpa=0)

        assert_allclose(polr, -0.1, rtol=1e-10)


# =============================================================================
# Tests for calc_stokes (full pipeline)
# =============================================================================
class TestCalcStokes:
    """Tests for calc_stokes function (full Stokes parameter calculation)."""

    def test_calc_stokes_unpolarized(self, polarimetry_unpolarized):
        """Test full pipeline with unpolarized source."""
        result = calc_stokes(**polarimetry_unpolarized)

        pol, thp, dpol, dthp = result

        # Polarization should be ~0
        assert_allclose(pol, 0.0, atol=1e-10)

    def test_calc_stokes_known_polarization(self, polarimetry_qu_known):
        """
        Test full pipeline with known polarization.

        q = u = 1/3
        pol = sqrt(q^2 + u^2) = sqrt(2/9) = sqrt(2)/3 = 0.4714...
        """
        result = calc_stokes(**polarimetry_qu_known)

        pol, thp, dpol, dthp = result

        expected_pol = np.sqrt(2) / 3
        assert_allclose(pol, expected_pol, rtol=1e-5)

    def test_calc_stokes_with_corrections(self, polarimetry_qu_known):
        """Test full pipeline with efficiency and offset corrections."""
        result = calc_stokes(
            **polarimetry_qu_known,
            p_eff=0.95,  # 95% efficiency
            q_off=0.01,  # Small instrumental offset
            u_off=0.01,
        )

        pol, thp, dpol, dthp = result

        # Should still get reasonable polarization
        assert 0.0 < pol < 1.0


# =============================================================================
# Analytical polarimetry tests
# =============================================================================
class TestPolarimetryAnalytical:
    """Analytical tests for polarimetry formulas."""

    def test_qu_ratio_formula(self):
        """
        Verify q calculation formula.

        r_q = sqrt((e_000/o_000) / (e_450/o_450))
        q = (r_q - 1) / (r_q + 1)

        For e_000 = 2, o_000 = 1, e_450 = 1, o_450 = 2:
        r_q = sqrt((2/1) / (1/2)) = sqrt(4) = 2
        q = (2-1)/(2+1) = 1/3
        """
        e_000, o_000 = 2.0, 1.0
        e_450, o_450 = 1.0, 2.0

        r_q = np.sqrt((e_000 / o_000) / (e_450 / o_450))
        q = (r_q - 1) / (r_q + 1)

        assert_allclose(r_q, 2.0, rtol=1e-10)
        assert_allclose(q, 1.0/3.0, rtol=1e-10)

    def test_polarization_degree_formula(self):
        """
        Verify polarization degree formula.

        pol = sqrt(q^2 + u^2)

        For q = 3/5, u = 4/5:
        pol = sqrt(9/25 + 16/25) = sqrt(25/25) = 1
        """
        q, u = 3.0/5.0, 4.0/5.0
        pol = np.sqrt(q**2 + u**2)

        assert_allclose(pol, 1.0, rtol=1e-10)

    def test_polarization_angle_formula(self):
        """
        Verify polarization angle formula.

        theta = 0.5 * arctan2(u, q)

        For q = 1, u = 1:
        theta = 0.5 * arctan2(1, 1) = 0.5 * (pi/4) = pi/8 = 22.5 deg
        """
        q, u = 1.0, 1.0
        theta = 0.5 * np.arctan2(u, q)

        expected = np.pi / 8
        assert_allclose(theta, expected, rtol=1e-10)

    def test_rotation_matrix_formula(self):
        """
        Verify rotation formula for PA correction.

        q' = cos(2*theta)*q + sin(2*theta)*u
        u' = -sin(2*theta)*q + cos(2*theta)*u

        This is a rotation by 2*theta in the q-u plane.
        """
        q, u = 1.0, 0.0
        theta = np.pi / 4  # 45 degrees

        cos2t = np.cos(2 * theta)
        sin2t = np.sin(2 * theta)

        q_rot = cos2t * q + sin2t * u
        u_rot = -sin2t * q + cos2t * u

        # For theta=45 deg, cos(90)=0, sin(90)=1
        # q_rot = 0*1 + 1*0 = 0
        # u_rot = -1*1 + 0*0 = -1
        assert_allclose(q_rot, 0.0, atol=1e-10)
        assert_allclose(u_rot, -1.0, rtol=1e-10)
