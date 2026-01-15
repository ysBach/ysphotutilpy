"""
Tests for ysphotutilpy.util module.

All expected values are analytically derived.
"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from ..util import (
    sample_std,
    sigma_clipper,
    magsum,
    sqsum,
    err_prop,
    convert_pct,
    convert_deg,
    bezel_mask,
    gaussian_kernel,
    Gaussian2D_correct,
    normalize,
)


class TestSampleStd:
    """Tests for sample_std function."""

    def test_sample_std_ddof0(self, simple_array_for_std):
        """
        Test sample_std with ddof=0 (population std).

        Array: [1, 2, 3, 4, 5]
        Mean = 3
        Variance = ((1-3)^2 + (2-3)^2 + (3-3)^2 + (4-3)^2 + (5-3)^2) / 5
                 = (4 + 1 + 0 + 1 + 4) / 5 = 10/5 = 2
        Std = sqrt(2) = 1.4142135...
        """
        expected = np.sqrt(2.0)
        result = sample_std(simple_array_for_std, ddof=0)
        assert_allclose(result, expected, rtol=1e-10)

    def test_sample_std_ddof1(self, simple_array_for_std):
        """
        Test sample_std with ddof=1 (sample std).

        Array: [1, 2, 3, 4, 5], n=5
        Sample variance = 10 / (5-1) = 2.5
        Sample std = sqrt(2.5) = 1.5811388...

        But sample_std uses: sqrt(n/(n-ddof)) * nanstd
        nanstd with default ddof=0 gives sqrt(2)
        So: sqrt(5/4) * sqrt(2) = sqrt(5/4 * 2) = sqrt(2.5)
        """
        expected = np.sqrt(2.5)
        result = sample_std(simple_array_for_std, ddof=1)
        assert_allclose(result, expected, rtol=1e-10)

    def test_sample_std_with_nan(self, array_with_nan):
        """
        Test sample_std ignores NaN values.

        Array: [1, 2, NaN, 4, 5] -> effective [1, 2, 4, 5], n=4
        Mean = 12/4 = 3
        Variance = ((1-3)^2 + (2-3)^2 + (4-3)^2 + (5-3)^2) / 4
                 = (4 + 1 + 1 + 4) / 4 = 10/4 = 2.5
        """
        expected = np.sqrt(2.5)
        result = sample_std(array_with_nan, ddof=0)
        assert_allclose(result, expected, rtol=1e-10)

    def test_sample_std_empty_array(self):
        """Test sample_std returns empty array for empty input with large ddof."""
        arr = np.array([1.0])  # size=1
        result = sample_std(arr, ddof=1)  # division by zero case
        # The function returns empty array when ZeroDivisionError
        assert result.size == 0 or np.isinf(result) or np.isnan(result)


class TestSigmaClipper:
    """Tests for sigma_clipper function."""

    def test_sigma_clipper_no_outliers(self):
        """Array without outliers should remain unchanged."""
        arr = np.array([10.0, 10.1, 9.9, 10.0, 10.05])
        result = sigma_clipper(arr, sigma=3.0, maxiters=5)
        assert len(result) == len(arr)

    def test_sigma_clipper_with_outlier(self):
        """High outlier should be removed from array."""
        # Use larger array so outlier is clearly > 3 sigma from median
        arr = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 100.0])
        result = sigma_clipper(arr, sigma=3.0, maxiters=5)
        # With masked=False (hard-coded), clipped values are REMOVED
        assert len(result) < len(arr)  # Array should shrink
        assert 100.0 not in result  # Outlier should be removed

    def test_sigma_clipper_asymmetric(self):
        """Test asymmetric sigma bounds."""
        # Use array where 0.0 is clearly far below the median of ~10
        arr = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 0.0])
        # sigma_lower=2 should reject 0.0 (which is ~10 std below median when std is ~3)
        result = sigma_clipper(arr, sigma_lower=2.0, sigma_upper=10.0, maxiters=5)
        # 0.0 should be removed
        assert len(result) < len(arr)
        assert 0.0 not in result


class TestMagsum:
    """Tests for magsum function (sum of magnitudes)."""

    def test_magsum_equal_mags(self):
        """
        Two sources of equal magnitude.

        If m1 = m2 = 0, then flux1 = flux2 = 1
        Total flux = 2
        Total mag = -2.5 * log10(2) = -0.7525749891...
        """
        expected = -2.5 * np.log10(2)
        result = magsum(0, 0)
        assert_allclose(result, expected, rtol=1e-10)

    def test_magsum_different_mags(self):
        """
        Two sources with different magnitudes.

        m1 = 0 (flux = 1), m2 = 2.5 (flux = 10^(-1) = 0.1)
        Total flux = 1.1
        Total mag = -2.5 * log10(1.1) = -0.10379...
        """
        expected = -2.5 * np.log10(1.1)
        result = magsum(0, 2.5)
        assert_allclose(result, expected, rtol=1e-10)

    def test_magsum_three_sources(self):
        """
        Three sources: m = 0, 0, 0 -> flux = 1 + 1 + 1 = 3
        Total mag = -2.5 * log10(3) = -1.19279...
        """
        expected = -2.5 * np.log10(3)
        result = magsum(0, 0, 0)
        assert_allclose(result, expected, rtol=1e-10)


class TestSqsum:
    """Tests for sqsum function (sum of squares)."""

    def test_sqsum_3_4(self):
        """3^2 + 4^2 = 9 + 16 = 25"""
        assert sqsum(3, 4) == 25

    def test_sqsum_single(self):
        """5^2 = 25"""
        assert sqsum(5) == 25

    def test_sqsum_multiple(self):
        """1^2 + 2^2 + 3^2 = 1 + 4 + 9 = 14"""
        assert sqsum(1, 2, 3) == 14


class TestErrProp:
    """Tests for err_prop function (error propagation)."""

    def test_err_prop_3_4(self):
        """
        sqrt(3^2 + 4^2) = sqrt(25) = 5

        This is the 3-4-5 Pythagorean triple.
        """
        expected = 5.0
        result = err_prop(3, 4)
        assert_allclose(result, expected, rtol=1e-10)

    def test_err_prop_single(self):
        """sqrt(5^2) = 5"""
        assert_allclose(err_prop(5), 5.0, rtol=1e-10)

    def test_err_prop_multiple(self):
        """sqrt(1 + 4 + 9) = sqrt(14) = 3.7416..."""
        expected = np.sqrt(14)
        result = err_prop(1, 2, 3)
        assert_allclose(result, expected, rtol=1e-10)


class TestConvertPct:
    """Tests for convert_pct function (natural <-> percent)."""

    def test_natural_to_percent(self):
        """0.5 -> 50%"""
        result = convert_pct(0.5, already=False, convert2unit=True)
        assert_allclose(result, [50.0], rtol=1e-10)

    def test_percent_to_natural(self):
        """50% -> 0.5"""
        result = convert_pct(50.0, already=True, convert2unit=False)
        assert_allclose(result, [0.5], rtol=1e-10)

    def test_no_conversion(self):
        """Both True or both False: no change"""
        result = convert_pct(0.5, already=False, convert2unit=False)
        assert_allclose(result, [0.5], rtol=1e-10)

        result = convert_pct(50.0, already=True, convert2unit=True)
        assert_allclose(result, [50.0], rtol=1e-10)

    def test_multiple_values(self):
        """Multiple values converted at once."""
        result = convert_pct(0.1, 0.2, 0.3, already=False, convert2unit=True)
        expected = [10.0, 20.0, 30.0]
        assert_allclose(result, expected, rtol=1e-10)


class TestConvertDeg:
    """Tests for convert_deg function (radians <-> degrees)."""

    def test_rad_to_deg(self):
        """pi/2 rad -> 90 deg"""
        result = convert_deg(np.pi / 2, already=False, convert2unit=True)
        assert_allclose(result, [90.0], rtol=1e-10)

    def test_deg_to_rad(self):
        """90 deg -> pi/2 rad"""
        result = convert_deg(90.0, already=True, convert2unit=False)
        assert_allclose(result, [np.pi / 2], rtol=1e-10)

    def test_no_conversion(self):
        """Both True or both False: no change"""
        result = convert_deg(np.pi, already=False, convert2unit=False)
        assert_allclose(result, [np.pi], rtol=1e-10)

    def test_multiple_angles(self):
        """0, pi/4, pi/2, pi -> 0, 45, 90, 180 degrees"""
        result = convert_deg(0, np.pi/4, np.pi/2, np.pi, already=False, convert2unit=True)
        expected = [0.0, 45.0, 90.0, 180.0]
        assert_allclose(result, expected, rtol=1e-10)


class TestBezelMask:
    """Tests for bezel_mask function."""

    def test_bezel_mask_center(self):
        """Point at center should not be masked."""
        mask = bezel_mask(
            xvals=np.array([50]),
            yvals=np.array([50]),
            nx=100, ny=100,
            bezel_x=[10, 10],
            bezel_y=[10, 10]
        )
        assert not mask[0]

    def test_bezel_mask_edge_left(self):
        """Point at left edge should be masked."""
        mask = bezel_mask(
            xvals=np.array([5]),  # x < bezel_x[0] + 0.5 = 10.5
            yvals=np.array([50]),
            nx=100, ny=100,
            bezel_x=[10, 10],
            bezel_y=[10, 10]
        )
        assert mask[0]

    def test_bezel_mask_edge_right(self):
        """Point at right edge should be masked."""
        # x > (nx - bezel_x[1]) - 0.5 = 100 - 10 - 0.5 = 89.5
        mask = bezel_mask(
            xvals=np.array([95]),
            yvals=np.array([50]),
            nx=100, ny=100,
            bezel_x=[10, 10],
            bezel_y=[10, 10]
        )
        assert mask[0]

    def test_bezel_mask_corner(self):
        """Point at corner should be masked."""
        mask = bezel_mask(
            xvals=np.array([5]),
            yvals=np.array([5]),
            nx=100, ny=100,
            bezel_x=[10, 10],
            bezel_y=[10, 10]
        )
        assert mask[0]

    def test_bezel_mask_multiple_points(self):
        """Multiple points: some inside, some outside."""
        xvals = np.array([50, 5, 95, 50])
        yvals = np.array([50, 50, 50, 5])
        mask = bezel_mask(xvals, yvals, nx=100, ny=100, bezel_x=[10, 10], bezel_y=[10, 10])
        expected = np.array([False, True, True, True])
        assert_array_equal(mask, expected)


class TestNormalize:
    """Tests for normalize function."""

    def test_normalize_basic(self):
        """Basic normalization to [0, 360)"""
        assert normalize(370, 0, 360) == 10.0
        assert normalize(-10, 0, 360) == 350.0
        assert normalize(360, 0, 360) == 0.0

    def test_normalize_symmetric(self):
        """Normalization to [-180, 180)"""
        assert normalize(270, -180, 180) == -90.0
        assert normalize(-270, -180, 180) == 90.0

    def test_normalize_bounce(self):
        """Bounce mode (b=True)"""
        assert normalize(100, -90, 90, b=True) == 80.0
        assert normalize(-100, -90, 90, b=True) == -80.0


class TestGaussianKernel:
    """Tests for gaussian_kernel function."""

    def test_gaussian_kernel_sigma1(self):
        """
        Kernel with sigma=1.

        Note: The kernel is evaluated on a discrete grid, so the center value
        may not be exactly 1.0 even with amplitude=1. The grid offset from
        true center affects the peak value.
        """
        kernel = gaussian_kernel(sigma=1, nsigma=3, normalize_area=False)
        # Maximum should be at center and less than or equal to amplitude
        center_y, center_x = kernel.shape[0] // 2, kernel.shape[1] // 2
        # Center value should be the maximum
        assert kernel[center_y, center_x] == kernel.max()
        # Maximum should be <= 1 (amplitude)
        assert kernel.max() <= 1.0 + 1e-10

    def test_gaussian_kernel_normalized(self):
        """
        Normalized kernel should have sum close to 1 (for large enough nsigma).

        For nsigma=5 and larger sigma, more of the Gaussian is captured.
        """
        # Use larger sigma and nsigma to capture more of the distribution
        kernel = gaussian_kernel(sigma=5, nsigma=5, normalize_area=True)
        # Sum should be close to 1 (within ~1% due to truncation)
        assert_allclose(kernel.sum(), 1.0, rtol=0.02)

    def test_gaussian_kernel_fwhm(self):
        """
        Test kernel created with FWHM parameter.

        FWHM = 2 * sqrt(2 * ln(2)) * sigma ≈ 2.355 * sigma
        So FWHM=2.355 means sigma ≈ 1
        """
        fwhm = 2 * np.sqrt(2 * np.log(2))  # FWHM for sigma=1
        kernel = gaussian_kernel(fwhm=fwhm, nsigma=3, normalize_area=False)
        center_y, center_x = kernel.shape[0] // 2, kernel.shape[1] // 2
        # Center value should be the maximum
        assert kernel[center_y, center_x] == kernel.max()
        # Maximum should be <= 1 (amplitude)
        assert kernel.max() <= 1.0 + 1e-10

    def test_gaussian_kernel_asymmetric(self):
        """
        Asymmetric kernel with different x and y sigmas.
        """
        kernel = gaussian_kernel(sigma=[2, 1], nsigma=3, normalize_area=False)
        # Should be elongated along x (wider in x direction)
        assert kernel.shape[1] > kernel.shape[0]


class TestGaussian2DCorrect:
    """Tests for Gaussian2D_correct function."""

    def test_swap_axes_when_y_larger(self):
        """
        When y_stddev > x_stddev, axes should be swapped and theta adjusted.
        """
        from astropy.modeling.functional_models import Gaussian2D

        # Create Gaussian with y_stddev > x_stddev
        g = Gaussian2D(amplitude=1, x_mean=0, y_mean=0, x_stddev=1, y_stddev=2, theta=0)
        g_corrected = Gaussian2D_correct(g)

        # After correction, x_stddev should be the larger one
        assert g_corrected.x_stddev.value >= g_corrected.y_stddev.value

    def test_theta_normalized(self):
        """
        Theta should be normalized to [-pi/2, pi/2].
        """
        from astropy.modeling.functional_models import Gaussian2D

        g = Gaussian2D(amplitude=1, x_mean=0, y_mean=0, x_stddev=2, y_stddev=1, theta=5.0)
        g_corrected = Gaussian2D_correct(g)

        assert -np.pi / 2 <= g_corrected.theta.value <= np.pi / 2

    def test_preserves_shape(self):
        """
        Corrected Gaussian should produce identical output.
        """
        from astropy.modeling.functional_models import Gaussian2D

        g = Gaussian2D(amplitude=100, x_mean=5, y_mean=5, x_stddev=1, y_stddev=2, theta=0.5)
        g_corrected = Gaussian2D_correct(g)

        yy, xx = np.mgrid[:10, :10]
        original = g(xx, yy)
        corrected = g_corrected(xx, yy)

        assert_allclose(original, corrected, rtol=1e-10)
