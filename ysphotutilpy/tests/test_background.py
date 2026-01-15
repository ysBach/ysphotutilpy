"""
Tests for ysphotutilpy.background module.

All expected values are analytically derived.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from photutils.aperture import CircularAnnulus, EllipticalAnnulus

from . import STAR_1_2
from ..background import annul2values, sky_fit, quick_sky_circ, mmm_dao


# =============================================================================
# Tests for annul2values (existing tests expanded)
# =============================================================================
class TestAnnul2Values:
    """Tests for annul2values function."""

    @pytest.mark.parametrize(
        "positions, num1, num2, num500",
        [
            ((24.5, 24.5), 38, 38, 0),
            ((10, 10), 0, 76, 0),
            ((10, 40), 76, 0, 0),
            ((35, 40), 76, 0, 0),
            ((35, 11), 0, 75, 1),
        ]
    )
    def test_annul2values_CircularAnnulus(self, positions, num1, num2, num500):
        """Test annul2values with CircularAnnulus at various positions."""
        an = CircularAnnulus(positions=positions, r_in=5, r_out=7)
        vals = annul2values(STAR_1_2, an, mask=None)
        assert len(vals[0]) == 76
        assert np.count_nonzero(vals[0] == 1) == num1
        assert np.count_nonzero(vals[0] == 2) == num2
        assert np.count_nonzero(vals[0] == 500) == num500

    def test_annul2values_uniform_array(self, uniform_100x100):
        """
        Test annul2values on uniform array.

        All extracted values should equal the uniform value (10.0).
        """
        an = CircularAnnulus(positions=(50, 50), r_in=10, r_out=15)
        vals = annul2values(uniform_100x100, an, mask=None)

        assert_allclose(vals[0], 10.0, rtol=1e-10)

    def test_annul2values_with_mask(self, uniform_100x100):
        """
        Test annul2values respects mask.

        Masked pixels should not be included in output.
        """
        mask = np.zeros_like(uniform_100x100, dtype=bool)
        # Mask some pixels in the annulus region
        mask[50, 60] = True
        mask[50, 61] = True
        mask[50, 62] = True

        an = CircularAnnulus(positions=(50, 50), r_in=10, r_out=15)
        vals_nomask = annul2values(uniform_100x100, an, mask=None)
        vals_masked = annul2values(uniform_100x100, an, mask=mask)

        # Masked version should have fewer values
        assert len(vals_masked[0]) < len(vals_nomask[0])

    def test_annul2values_elliptical(self, uniform_100x100):
        """Test annul2values with EllipticalAnnulus."""
        an = EllipticalAnnulus(positions=(50, 50), a_in=8, a_out=12, b_out=6, theta=0)
        vals = annul2values(uniform_100x100, an, mask=None)

        # All values should be 10.0
        assert_allclose(vals[0], 10.0, rtol=1e-10)


# =============================================================================
# Tests for sky_fit
# =============================================================================
class TestSkyFit:
    """Tests for sky_fit function."""

    def test_sky_fit_uniform_mean(self, uniform_100x100):
        """
        Test sky_fit with method='mean' on uniform array.

        Expected: msky = 10.0, ssky = 0.0
        """
        an = CircularAnnulus(positions=(50, 50), r_in=10, r_out=20)
        result = sky_fit(uniform_100x100, an, method='mean')

        assert_allclose(result['msky'][0], 10.0, rtol=1e-10)
        assert_allclose(result['ssky'][0], 0.0, atol=1e-10)

    def test_sky_fit_uniform_median(self, uniform_100x100):
        """
        Test sky_fit with method='median' on uniform array.

        Expected: msky = 10.0, ssky = 0.0
        """
        an = CircularAnnulus(positions=(50, 50), r_in=10, r_out=20)
        result = sky_fit(uniform_100x100, an, method='median')

        assert_allclose(result['msky'][0], 10.0, rtol=1e-10)
        assert_allclose(result['ssky'][0], 0.0, atol=1e-10)

    def test_sky_fit_uniform_sex(self, uniform_100x100):
        """
        Test sky_fit with method='sex' (SExtractor) on uniform array.

        For uniform array: mean = median = 10.0
        Since (mean - median)/std is undefined (std=0), should return median.
        """
        an = CircularAnnulus(positions=(50, 50), r_in=10, r_out=20)
        result = sky_fit(uniform_100x100, an, method='sex')

        assert_allclose(result['msky'][0], 10.0, rtol=1e-10)

    def test_sky_fit_with_noise(self, uniform_with_noise):
        """
        Test sky_fit on noisy data recovers approximate mean.

        Data: N(100, 10), method='mean'
        Expected: msky ≈ 100 (within several std/sqrt(n))
        """
        an = CircularAnnulus(positions=(50, 50), r_in=10, r_out=30)
        result = sky_fit(uniform_with_noise, an, method='mean')

        # Should be close to 100, allow 3-sigma tolerance
        # With ~500 pixels, std of mean ≈ 10/sqrt(500) ≈ 0.45
        assert_allclose(result['msky'][0], 100.0, atol=3.0)

    def test_sky_fit_nsky_nrej(self, uniform_100x100):
        """Test nsky and nrej are correctly reported."""
        an = CircularAnnulus(positions=(50, 50), r_in=10, r_out=15)
        result = sky_fit(uniform_100x100, an, method='mean')

        # nsky should be positive
        assert result['nsky'][0] > 0
        # nrej should be 0 for uniform array (no sigma clipping rejects)
        assert result['nrej'][0] == 0

    def test_sky_fit_no_annulus(self, uniform_100x100):
        """
        Test sky_fit with annulus=None uses whole image.
        """
        result = sky_fit(uniform_100x100, annulus=None, method='mean')

        assert_allclose(result['msky'][0], 10.0, rtol=1e-10)
        assert result['nsky'][0] == 100 * 100  # whole image

    def test_sky_fit_iraf_method(self, uniform_with_noise):
        """
        Test sky_fit with method='iraf'.

        IRAF: if mean < median, use mean; else use 3*median - 2*mean
        """
        an = CircularAnnulus(positions=(50, 50), r_in=10, r_out=30)
        result = sky_fit(uniform_with_noise, an, method='iraf')

        # Should be close to 100
        assert_allclose(result['msky'][0], 100.0, atol=5.0)

    def test_sky_fit_mmm_method(self, uniform_with_noise):
        """
        Test sky_fit with method='mmm'.

        MMM: 3*median - 2*mean
        """
        an = CircularAnnulus(positions=(50, 50), r_in=10, r_out=30)
        result = sky_fit(uniform_with_noise, an, method='mmm')

        # Should be close to 100
        assert_allclose(result['msky'][0], 100.0, atol=5.0)

    def test_sky_fit_callable_method(self, uniform_100x100):
        """
        Test sky_fit with callable method.

        Custom method: return max of sky array.
        """
        def custom_method(skyarr, ssky):
            return np.max(skyarr)

        an = CircularAnnulus(positions=(50, 50), r_in=10, r_out=20)
        result = sky_fit(uniform_100x100, an, method=custom_method)

        assert_allclose(result['msky'][0], 10.0, rtol=1e-10)

    def test_sky_fit_return_dict(self, uniform_100x100):
        """Test sky_fit returns dict when to_table=False."""
        an = CircularAnnulus(positions=(50, 50), r_in=10, r_out=20)
        result = sky_fit(uniform_100x100, an, method='mean', to_table=False)

        assert isinstance(result, list)
        assert isinstance(result[0], dict)
        assert 'msky' in result[0]

    def test_sky_fit_return_skyarr(self, uniform_100x100):
        """Test sky_fit returns sky array when return_skyarr=True."""
        an = CircularAnnulus(positions=(50, 50), r_in=10, r_out=20)
        result, skys = sky_fit(
            uniform_100x100, an, method='mean', return_skyarr=True
        )

        assert isinstance(skys, list)
        assert len(skys) == 1
        assert_allclose(skys[0], 10.0, rtol=1e-10)


# =============================================================================
# Tests for quick_sky_circ
# =============================================================================
class TestQuickSkyCirc:
    """Tests for quick_sky_circ convenience function."""

    def test_quick_sky_circ_uniform(self, uniform_100x100):
        """Test quick_sky_circ on uniform array."""
        result = quick_sky_circ(uniform_100x100, pos=(50, 50), r_in=10, r_out=20)

        assert_allclose(result['msky'][0], 10.0, rtol=1e-10)


# =============================================================================
# Tests for mmm_dao
# =============================================================================
class TestMmmDao:
    """Tests for mmm_dao function (DAOPHOT MMM algorithm)."""

    def test_mmm_dao_uniform(self):
        """
        Test mmm_dao on uniform array.

        For uniform data, mmm_dao should return that value.
        """
        sky = np.full(1000, 100.0)
        result = mmm_dao(sky)
        assert_allclose(result, 100.0, rtol=1e-5)

    def test_mmm_dao_gaussian_noise(self):
        """
        Test mmm_dao on Gaussian noise.

        N(100, 10) should give mmm estimate close to 100.
        """
        np.random.seed(42)
        sky = np.random.normal(loc=100.0, scale=10.0, size=5000)
        result = mmm_dao(sky)

        # Should be close to 100
        assert_allclose(result, 100.0, atol=2.0)

    def test_mmm_dao_with_outliers(self):
        """
        Test mmm_dao is robust to outliers.

        Add some high outliers to Gaussian background.
        """
        np.random.seed(42)
        sky = np.random.normal(loc=100.0, scale=10.0, size=5000)
        # Add outliers
        sky[:50] = 500.0

        result = mmm_dao(sky)

        # Should still be close to 100 (robust estimator)
        assert_allclose(result, 100.0, atol=5.0)

    def test_mmm_dao_too_few_pixels(self):
        """Test mmm_dao raises error with too few pixels."""
        sky = np.array([100.0, 100.0, 100.0])  # only 3 pixels

        with pytest.raises(ValueError, match="must contain at least"):
            mmm_dao(sky, min_nsky=20)

    def test_mmm_dao_highbad(self):
        """
        Test mmm_dao with highbad parameter.

        Pixels above highbad should be rejected.
        """
        np.random.seed(42)
        sky = np.random.normal(loc=100.0, scale=10.0, size=5000)
        sky[:100] = 200.0  # These should be rejected if highbad=150

        result = mmm_dao(sky, highbad=150)

        # Should be close to 100
        assert_allclose(result, 100.0, atol=3.0)


# =============================================================================
# Analytical sky estimation tests
# =============================================================================
class TestSkyFitAnalytical:
    """Analytical tests for sky estimation methods."""

    def test_sex_estimator_formula(self):
        """
        Test SExtractor sky estimator formula.

        For |mean - median|/std < 0.3: use median
        Otherwise: use 2.5*median - 1.5*mean
        """
        # Create slightly skewed distribution
        np.random.seed(42)
        # Uniform data: mean = median, so should use median path
        sky = np.full(1000, 50.0)
        result = sky_fit(sky, annulus=None, method='sex')

        assert_allclose(result['msky'][0], 50.0, rtol=1e-10)

    def test_iraf_estimator_formula(self):
        """
        Test IRAF sky estimator formula (after sigma clipping).

        if mean < median: msky = mean
        else: msky = 3*median - 2*mean

        Note: sky_fit applies sigma clipping first, so we need to account for that.
        """
        # Create symmetric data where sigma clipping doesn't change much
        np.random.seed(42)
        sky = np.random.normal(loc=100.0, scale=5.0, size=1000)

        result = sky_fit(sky, annulus=None, method='iraf')

        # Apply same sigma clipping as sky_fit does internally
        from ..util import sigma_clipper
        sky_clipped = sigma_clipper(sky)
        sky_clipped = sky_clipped[~np.isnan(sky_clipped)]  # Remove NaN

        mean = np.mean(sky_clipped)
        median = np.median(sky_clipped)
        if mean < median:
            expected = mean
        else:
            expected = 3 * median - 2 * mean

        assert_allclose(result['msky'][0], expected, rtol=1e-5)

    def test_mmm_estimator_formula(self):
        """
        Test MMM sky estimator formula: 3*median - 2*mean (after sigma clipping)

        Note: sky_fit applies sigma clipping first, so we compute expected
        from clipped data.
        """
        np.random.seed(42)
        sky = np.random.normal(loc=100.0, scale=10.0, size=1000)

        result = sky_fit(sky, annulus=None, method='mmm')

        # Apply same sigma clipping as sky_fit does internally
        from ..util import sigma_clipper
        sky_clipped = sigma_clipper(sky)
        sky_clipped = sky_clipped[~np.isnan(sky_clipped)]  # Remove NaN

        mean = np.mean(sky_clipped)
        median = np.median(sky_clipped)
        expected = 3 * median - 2 * mean

        assert_allclose(result['msky'][0], expected, rtol=1e-5)
