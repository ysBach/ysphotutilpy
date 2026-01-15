"""
Tests for ysphotutilpy.radprof module.

All expected values are analytically derived.
"""

import numpy as np
import pytest
from astropy.nddata import CCDData
from numpy.testing import assert_allclose
from photutils.aperture import CircularAnnulus

from ..background import annul2values
from ..radprof import (
    gauss_r,
    moffat_r,
    bivt_r,
    fwhm_r,
    radial_profile,
    radprof_pix,
)


# =============================================================================
# Tests for gauss_r (radial Gaussian function)
# =============================================================================
class TestGaussR:
    """Tests for gauss_r function."""

    def test_gauss_r_at_center(self):
        """
        Test Gaussian at r = 0.

        gauss_r(0, amp, sig, const) = amp * exp(0) + const = amp + const
        """
        amp, sig, const = 100.0, 3.0, 10.0
        result = gauss_r(r=0, amp=amp, sig=sig, const=const)

        expected = amp + const  # 110.0
        assert_allclose(result, expected, rtol=1e-10)

    def test_gauss_r_at_sigma(self):
        """
        Test Gaussian at r = sigma.

        gauss_r(sig, amp, sig, const) = amp * exp(-0.5) + const
        """
        amp, sig, const = 100.0, 3.0, 10.0
        result = gauss_r(r=sig, amp=amp, sig=sig, const=const)

        expected = amp * np.exp(-0.5) + const
        assert_allclose(result, expected, rtol=1e-10)

    def test_gauss_r_at_2sigma(self):
        """
        Test Gaussian at r = 2*sigma.

        gauss_r(2*sig, amp, sig, const) = amp * exp(-2) + const
        """
        amp, sig, const = 100.0, 3.0, 10.0
        result = gauss_r(r=2*sig, amp=amp, sig=sig, const=const)

        expected = amp * np.exp(-2.0) + const
        assert_allclose(result, expected, rtol=1e-10)

    def test_gauss_r_at_infinity(self):
        """
        Test Gaussian at large r approaches constant.

        As r -> infinity: gauss_r -> const
        """
        amp, sig, const = 100.0, 3.0, 10.0
        result = gauss_r(r=100.0, amp=amp, sig=sig, const=const)

        assert_allclose(result, const, atol=1e-10)

    def test_gauss_r_array_input(self):
        """Test Gaussian with array input."""
        amp, sig, const = 100.0, 3.0, 0.0
        r = np.array([0, 1, 2, 3])
        result = gauss_r(r=r, amp=amp, sig=sig, const=const)

        expected = amp * np.exp(-0.5 * (r / sig)**2)
        assert_allclose(result, expected, rtol=1e-10)


# =============================================================================
# Tests for moffat_r (radial Moffat function)
# =============================================================================
class TestMoffatR:
    """Tests for moffat_r function."""

    def test_moffat_r_at_center(self):
        """
        Test Moffat at r = 0.

        moffat_r(0, amp, core, power, const) = amp * (1 + 0)^(-power) + const
                                             = amp + const
        """
        amp, core, power, const = 100.0, 2.0, 2.5, 10.0
        result = moffat_r(r=0, amp=amp, core=core, power=power, const=const)

        expected = amp + const
        assert_allclose(result, expected, rtol=1e-10)

    def test_moffat_r_at_core(self):
        """
        Test Moffat at r = core.

        moffat_r(core, amp, core, power, const) = amp * (1 + 1)^(-power) + const
                                                = amp * 2^(-power) + const
        """
        amp, core, power, const = 100.0, 2.0, 2.5, 10.0
        result = moffat_r(r=core, amp=amp, core=core, power=power, const=const)

        expected = amp * (2.0)**(-power) + const
        assert_allclose(result, expected, rtol=1e-10)

    def test_moffat_r_specific_values(self):
        """
        Test Moffat with specific analytical values.

        For amp=100, core=1, power=2.5, const=0:
        At r=core=1: moffat_r = 100 * 2^(-2.5) = 100/5.657 = 17.678...
        """
        result = moffat_r(r=1, amp=100, core=1, power=2.5, const=0)

        expected = 100 * (2.0)**(-2.5)  # 17.6776695...
        assert_allclose(result, expected, rtol=1e-10)

    def test_moffat_r_at_large_r(self):
        """Test Moffat at large r approaches constant."""
        amp, core, power, const = 100.0, 2.0, 2.5, 10.0
        result = moffat_r(r=1000.0, amp=amp, core=core, power=power, const=const)

        assert_allclose(result, const, atol=0.1)


# =============================================================================
# Tests for bivt_r (bivariate t / generalized Moffat)
# =============================================================================
class TestBivtR:
    """Tests for bivt_r function."""

    def test_bivt_r_at_center(self):
        """
        Test bivariate t at r = 0.

        bivt_r(0, amp, num2, sig, const) = amp * 1^(-(num2+4)/2) + const
                                         = amp + const
        """
        amp, num2, sig, const = 100.0, 3.0, 2.0, 10.0
        result = bivt_r(r=0, amp=amp, num2=num2, sig=sig, const=const)

        expected = amp + const
        assert_allclose(result, expected, rtol=1e-10)

    def test_bivt_r_formula(self):
        """
        Test bivariate t formula.

        bivt_r(r, amp, num2, sig, const) = amp * (1 + r^2/(num2*sig^2))^(-(num2+4)/2) + const
        """
        r, amp, num2, sig, const = 2.0, 100.0, 3.0, 2.0, 0.0

        result = bivt_r(r=r, amp=amp, num2=num2, sig=sig, const=const)

        # Manual calculation
        exponent = -(num2 + 4) / 2  # -(7/2) = -3.5
        base = 1 + r**2 / (num2 * sig**2)  # 1 + 4/(3*4) = 1 + 1/3 = 4/3
        expected = amp * base**exponent

        assert_allclose(result, expected, rtol=1e-10)


# =============================================================================
# Tests for fwhm_r (FWHM calculation)
# =============================================================================
class TestFwhmR:
    """Tests for fwhm_r function."""

    def test_fwhm_r_gauss(self):
        """
        Test FWHM for Gaussian.

        FWHM_gauss = 2 * sqrt(2 * ln(2)) * sigma = 2.3548... * sigma
        """
        popt = [100, 3.0, 0]  # [amp, sig, const]

        result = fwhm_r(popt, fun="gauss")

        expected = 2 * np.sqrt(2 * np.log(2)) * 3.0
        assert_allclose(result, expected, rtol=1e-10)

    def test_fwhm_r_gauss_sigma1(self):
        """
        Test FWHM for Gaussian with sigma = 1.

        FWHM = 2.3548...
        """
        popt = [100, 1.0, 0]

        result = fwhm_r(popt, fun="gauss")

        expected = 2 * np.sqrt(2 * np.log(2))  # 2.3548...
        assert_allclose(result, expected, rtol=1e-10)

    def test_fwhm_r_moffat(self):
        """
        Test FWHM for Moffat.

        FWHM_moffat = 2 * core * sqrt(2^(1/power) - 1)
        """
        popt = [100, 2.0, 2.5, 0]  # [amp, core, power, const]

        result = fwhm_r(popt, fun="moffat")

        expected = 2 * 2.0 * np.sqrt(2**(1/2.5) - 1)
        assert_allclose(result, expected, rtol=1e-10)


# =============================================================================
# Tests for radial_profile
# =============================================================================
class TestRadialProfile:
    """Tests for radial_profile function."""

    def test_radial_profile_uniform(self, uniform_100x100):
        """
        Test radial profile on uniform image.

        All radii should have same value = 10.0
        """
        radii = [5, 10, 15, 20]
        profs, center_val = radial_profile(
            uniform_100x100, center=(50, 50), radii=radii, thickness=3
        )

        # All mean pixel values should be 10.0
        assert_allclose(profs['mpix'].values, 10.0, rtol=1e-5)

        # Standard deviation should be 0
        assert_allclose(profs['spix'].values, 0.0, atol=1e-10)

    def test_radial_profile_center_value(self, uniform_100x100):
        """Test center value is correctly returned."""
        profs, center_val = radial_profile(
            uniform_100x100, center=(50, 50), radii=[10], thickness=3
        )

        assert_allclose(center_val, 10.0, rtol=1e-10)

    def test_radial_profile_with_add_center(self, uniform_100x100):
        """Test add_center option adds center pixel to profile."""
        profs, center_val = radial_profile(
            uniform_100x100, center=(50, 50), radii=[5, 10], thickness=3,
            add_center=True
        )

        # Should have 3 rows: r=0, r=5, r=10
        assert len(profs) == 3
        assert profs['r'].iloc[0] == 0

    def test_radial_profile_normalized(self, uniform_100x100):
        """Test normalization by center value."""
        profs, center_val = radial_profile(
            uniform_100x100, center=(50, 50), radii=[10], thickness=3,
            norm_by_center=True
        )

        # mpix should be 1.0 (normalized)
        assert_allclose(profs['mpix'].values, 1.0, rtol=1e-5)


# =============================================================================
# Tests for radprof_pix
# =============================================================================
class TestRadprofPix:
    """Tests for radprof_pix function."""

    def test_radprof_pix_uniform(self, uniform_100x100):
        """Test radprof_pix on uniform image."""
        r, vals = radprof_pix(uniform_100x100, pos=(50, 50), rmax=10)

        # All values should be 10.0
        assert_allclose(vals, 10.0, rtol=1e-10)

    def test_radprof_pix_sorted(self, uniform_100x100):
        """Test radprof_pix with sort_dist=True."""
        r, vals = radprof_pix(
            uniform_100x100, pos=(50, 50), rmax=10, sort_dist=True
        )

        # Radii should be sorted
        assert np.all(np.diff(r) >= 0)

    def test_radprof_pix_ccddata_input(self, ccd_uniform):
        """Test radprof_pix accepts CCDData input."""
        r, vals = radprof_pix(ccd_uniform, pos=(50, 50), rmax=10)

        assert_allclose(vals, 100.0, rtol=1e-10)


# =============================================================================
# Analytical radial profile tests
# =============================================================================
class TestRadialProfileAnalytical:
    """Analytical tests for radial profile calculations."""

    def test_gaussian_half_max_radius(self):
        """
        Verify Gaussian half-maximum occurs at r = FWHM/2.

        For Gaussian: G(r) = A * exp(-0.5 * (r/sigma)^2)
        Half-max at: G(r) = A/2
        => exp(-0.5 * (r/sigma)^2) = 0.5
        => r = sigma * sqrt(2 * ln(2)) = FWHM/2
        """
        amp, sig = 100.0, 3.0
        fwhm = 2 * np.sqrt(2 * np.log(2)) * sig

        r_halfmax = fwhm / 2
        value_at_halfmax = gauss_r(r_halfmax, amp=amp, sig=sig, const=0)

        expected_halfmax = amp / 2
        assert_allclose(value_at_halfmax, expected_halfmax, rtol=1e-10)

    def test_moffat_half_max_radius(self):
        """
        Verify Moffat half-maximum occurs at r = core * sqrt(2^(1/power) - 1).

        For Moffat: M(r) = A * (1 + (r/core)^2)^(-power)
        Half-max at: M(r) = A/2
        => (1 + (r/core)^2)^(-power) = 0.5
        => r = core * sqrt(2^(1/power) - 1)
        """
        amp, core, power = 100.0, 2.0, 2.5

        r_halfmax = core * np.sqrt(2**(1/power) - 1)
        value_at_halfmax = moffat_r(r_halfmax, amp=amp, core=core, power=power, const=0)

        expected_halfmax = amp / 2
        assert_allclose(value_at_halfmax, expected_halfmax, rtol=1e-10)

    def test_radial_annulus_pixel_count(self, uniform_100x100):
        """
        Verify number of pixels in annulus.

        Annulus area = pi * (r_out^2 - r_in^2)
        For r_in=5, r_out=7: area = pi * (49 - 25) = 24*pi ≈ 75.4
        """
        an = CircularAnnulus((50, 50), r_in=5, r_out=7)

        # Using annul2values to count pixels
        vals = annul2values(uniform_100x100, an)

        # Should have approximately pi * (49 - 25) = 75.4 pixels
        # (method='center' gives integer pixel count)
        expected_approx = np.pi * (49 - 25)
        assert abs(len(vals[0]) - expected_approx) < 5


# =============================================================================
# Edge cases
# =============================================================================
class TestRadprofEdgeCases:
    """Edge case tests for radial profile functions."""

    def test_gauss_r_zero_amplitude(self):
        """Test Gaussian with zero amplitude."""
        result = gauss_r(r=5, amp=0, sig=3, const=10)
        assert_allclose(result, 10.0, rtol=1e-10)

    def test_moffat_r_large_power(self):
        """Test Moffat with large power index approaches delta function."""
        # Large power -> profile becomes sharply peaked
        result_center = moffat_r(r=0, amp=100, core=1, power=10, const=0)
        result_away = moffat_r(r=1, amp=100, core=1, power=10, const=0)

        # At center should be amp, away should be much smaller
        assert_allclose(result_center, 100.0, rtol=1e-10)
        assert result_away < result_center / 100

    def test_radprof_pix_at_edge(self, uniform_100x100):
        """Test radprof_pix at image edge."""
        # Near corner
        r, vals = radprof_pix(uniform_100x100, pos=(5, 5), rmax=3)

        # Should still return values
        assert len(vals) > 0
        assert_allclose(vals, 10.0, rtol=1e-10)
