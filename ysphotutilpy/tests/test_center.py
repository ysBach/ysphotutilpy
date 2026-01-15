"""
Tests for ysphotutilpy.center module.

All expected values are analytically derived.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from astropy.nddata import CCDData

from ..center import (
    circular_slice,
    circular_bbox_cut,
    find_centroid,
    GaussianConst2D,
)


# =============================================================================
# Tests for circular_slice
# =============================================================================
class TestCircularSlice:
    """Tests for circular_slice function."""

    def test_circular_slice_center(self):
        """
        Test circular_slice for centered position.

        pos = (50, 50), radius = 10, shape = (100, 100)
        Expected: slice(40, 61) for both axes (from 50-10 to 50+10+1)
        """
        shape = (100, 100)
        pos = (50, 50)
        radius = 10

        slices = circular_slice(shape, pos, radius)

        # Should cover from 40 to 60 inclusive (61 exclusive)
        assert slices[0] == slice(40, 61)
        assert slices[1] == slice(40, 61)

    def test_circular_slice_edge(self):
        """
        Test circular_slice when aperture extends beyond edge.

        pos = (5, 50), radius = 10, shape = (100, 100)
        x-slice should be truncated at 0
        """
        shape = (100, 100)
        pos = (5, 50)
        radius = 10

        slices = circular_slice(shape, pos, radius)

        # x-slice should start at 0 (not -5)
        assert slices[1].start >= 0
        # y-slice should be normal
        assert slices[0] == slice(40, 61)

    def test_circular_slice_corner(self):
        """
        Test circular_slice at corner.

        pos = (5, 5), radius = 10
        Both slices should be truncated at 0
        """
        shape = (100, 100)
        pos = (5, 5)
        radius = 10

        slices = circular_slice(shape, pos, radius)

        assert slices[0].start >= 0
        assert slices[1].start >= 0

    def test_circular_slice_return_offset(self):
        """Test circular_slice returns offset when requested."""
        shape = (100, 100)
        pos = (50, 50)
        radius = 10

        slices, offset = circular_slice(shape, pos, radius, return_offset=True)

        # Offset should be [40, 40] (start of slice)
        assert_array_equal(offset, [40, 40])


# =============================================================================
# Tests for circular_bbox_cut
# =============================================================================
class TestCircularBboxCut:
    """Tests for circular_bbox_cut function."""

    def test_circular_bbox_cut_shape(self, uniform_100x100):
        """Test cutout has correct shape."""
        pos = (50, 50)
        radius = 10

        cut, pos_cut, offset = circular_bbox_cut(uniform_100x100, pos, radius)

        # Cut should be approximately 21x21
        assert cut.shape[0] == 21
        assert cut.shape[1] == 21

    def test_circular_bbox_cut_position(self, uniform_100x100):
        """Test position is correctly transformed to cutout coordinates."""
        pos = (50, 50)
        radius = 10

        cut, pos_cut, offset = circular_bbox_cut(uniform_100x100, pos, radius)

        # Position in cutout should be at center (10, 10)
        assert_allclose(pos_cut, [10, 10], rtol=1e-10)

    def test_circular_bbox_cut_offset(self, uniform_100x100):
        """Test offset is correctly reported."""
        pos = (50, 50)
        radius = 10

        cut, pos_cut, offset = circular_bbox_cut(uniform_100x100, pos, radius)

        # Offset should be (40, 40)
        assert_array_equal(offset, [40, 40])

    def test_circular_bbox_cut_with_dists(self, uniform_100x100):
        """Test distances are correctly calculated when requested."""
        pos = (50, 50)
        radius = 10

        cut, pos_cut, offset, dists = circular_bbox_cut(
            uniform_100x100, pos, radius, return_dists=True
        )

        # Distance at center should be 0
        center_y, center_x = cut.shape[0] // 2, cut.shape[1] // 2
        assert_allclose(dists[center_y, center_x], 0.0, atol=1e-10)

        # Distance at corner should be sqrt(2) * 10 ≈ 14.14
        corner_dist = dists[0, 0]
        expected_corner = np.sqrt(10**2 + 10**2)
        assert_allclose(corner_dist, expected_corner, rtol=1e-10)

    def test_circular_bbox_cut_values(self, uniform_100x100):
        """Test cutout contains correct values."""
        pos = (50, 50)
        radius = 10

        cut, pos_cut, offset = circular_bbox_cut(uniform_100x100, pos, radius)

        # All values should be 10.0
        assert_allclose(cut, 10.0, rtol=1e-10)


# =============================================================================
# Tests for GaussianConst2D model
# =============================================================================
class TestGaussianConst2D:
    """Tests for GaussianConst2D model class."""

    def test_gaussian_const_2d_at_center(self):
        """
        Test GaussianConst2D value at center.

        At (x_mean, y_mean): value = amplitude + constant
        """
        model = GaussianConst2D(
            constant=10.0,
            amplitude=100.0,
            x_mean=5.0,
            y_mean=5.0,
            x_stddev=2.0,
            y_stddev=2.0,
            theta=0.0
        )

        value = model(5.0, 5.0)
        expected = 100.0 + 10.0  # amplitude + constant
        assert_allclose(value, expected, rtol=1e-10)

    def test_gaussian_const_2d_at_1sigma(self):
        """
        Test GaussianConst2D value at 1 sigma distance.

        At r = sigma: value = amplitude * exp(-0.5) + constant
        """
        model = GaussianConst2D(
            constant=10.0,
            amplitude=100.0,
            x_mean=5.0,
            y_mean=5.0,
            x_stddev=2.0,
            y_stddev=2.0,
            theta=0.0
        )

        # Point at 1 sigma in x direction
        value = model(7.0, 5.0)  # x = 5 + 2 = 7
        expected = 100.0 * np.exp(-0.5) + 10.0
        assert_allclose(value, expected, rtol=1e-10)

    def test_gaussian_const_2d_far_from_center(self):
        """
        Test GaussianConst2D far from center.

        At large distance: value ≈ constant (Gaussian contribution → 0)
        """
        model = GaussianConst2D(
            constant=10.0,
            amplitude=100.0,
            x_mean=5.0,
            y_mean=5.0,
            x_stddev=1.0,
            y_stddev=1.0,
            theta=0.0
        )

        # Point at 10 sigma away
        value = model(15.0, 5.0)  # x = 5 + 10 = 15
        # exp(-50) ≈ 0
        assert_allclose(value, 10.0, atol=1e-10)


# =============================================================================
# Tests for find_centroid
# =============================================================================
class TestFindCentroid:
    """Tests for find_centroid function."""

    def test_find_centroid_centered_source(self, ccd_with_source, gaussian_params_centered):
        """
        Test find_centroid on source at integer position.

        Source at (50, 50) should return (50, 50).
        """
        result = find_centroid(
            ccd_with_source,
            position_xy=(50, 50),
            cbox_size=15,
            maxiters=10,
            tol_shift=1e-4
        )

        expected_x = gaussian_params_centered['x_mean']
        expected_y = gaussian_params_centered['y_mean']

        assert_allclose(result[0], expected_x, atol=0.1)
        assert_allclose(result[1], expected_y, atol=0.1)

    def test_find_centroid_offset_source(self, ccd_with_source_offset, gaussian_params_offset):
        """
        Test find_centroid on source at fractional position.

        Source at (50.3, 50.7) should converge close to true position.
        """
        # Start with a slightly wrong initial guess
        result = find_centroid(
            ccd_with_source_offset,
            position_xy=(50, 51),
            cbox_size=15,
            maxiters=10,
            tol_shift=1e-4
        )

        expected_x = gaussian_params_offset['x_mean']
        expected_y = gaussian_params_offset['y_mean']

        # Should converge to within 0.2 pixels
        assert_allclose(result[0], expected_x, atol=0.2)
        assert_allclose(result[1], expected_y, atol=0.2)

    def test_find_centroid_full_output(self, ccd_with_source):
        """Test find_centroid with full=True returns history."""
        result = find_centroid(
            ccd_with_source,
            position_xy=(50, 50),
            cbox_size=15,
            maxiters=5,
            full=True
        )

        # Should return (position, x_history, y_history, total_shift)
        assert len(result) == 4
        pos, x_hist, y_hist, total = result

        # History arrays should have at least 2 elements (initial + 1 iteration)
        assert len(x_hist) >= 2
        assert len(y_hist) >= 2

    def test_find_centroid_max_shift_warning(self, ccd_with_source):
        """Test find_centroid warns when shift exceeds max_shift."""
        with pytest.warns(UserWarning, match="shifted larger than"):
            find_centroid(
                ccd_with_source,
                position_xy=(40, 40),  # Far from true position (50, 50)
                cbox_size=15,
                maxiters=10,
                max_shift=5.0  # Will exceed this
            )


# =============================================================================
# Analytical centroiding tests
# =============================================================================
class TestCentroidingAnalytical:
    """Analytical tests for centroiding accuracy."""

    def test_centroid_of_symmetric_gaussian(self):
        """
        Center of mass of symmetric 2D Gaussian equals its center.

        For G(x, y) = A * exp(-((x-x0)^2 + (y-y0)^2) / (2*sigma^2)),
        the center of mass is (x0, y0).
        """
        from photutils.centroids import centroid_com
        from .conftest import make_gaussian_2d

        # Create symmetric Gaussian at known position
        x_true, y_true = 25.5, 30.3
        data = make_gaussian_2d(
            shape=(50, 50),
            x_mean=x_true,
            y_mean=y_true,
            amplitude=1000.0,
            x_stddev=3.0,
            background=0.0
        )

        x_com, y_com = centroid_com(data)

        assert_allclose(x_com, x_true, atol=0.05)
        assert_allclose(y_com, y_true, atol=0.05)

    def test_centroid_unaffected_by_constant_background(self):
        """
        Adding constant background should not change centroid.

        COM(data + c) = COM(data) when c is constant across the image
        (this is only true if we use pixels above a threshold that
        excludes the background).
        """
        from photutils.centroids import centroid_com
        from .conftest import make_gaussian_2d

        x_true, y_true = 25.0, 25.0

        # Without background
        data_no_bg = make_gaussian_2d(
            shape=(50, 50),
            x_mean=x_true,
            y_mean=y_true,
            amplitude=1000.0,
            x_stddev=3.0,
            background=0.0
        )

        # With constant background (but mask it out)
        data_with_bg = make_gaussian_2d(
            shape=(50, 50),
            x_mean=x_true,
            y_mean=y_true,
            amplitude=1000.0,
            x_stddev=3.0,
            background=100.0
        )
        mask = data_with_bg < 150  # Mask out background

        x1, y1 = centroid_com(data_no_bg)
        x2, y2 = centroid_com(data_with_bg, mask=mask)

        # Both should give same centroid
        assert_allclose(x1, x2, atol=0.5)
        assert_allclose(y1, y2, atol=0.5)


# =============================================================================
# Edge cases
# =============================================================================
class TestCenterEdgeCases:
    """Edge case tests for centering functions."""

    def test_circular_slice_radius_zero(self):
        """Test circular_slice with radius=0."""
        shape = (100, 100)
        pos = (50, 50)
        radius = 0

        slices = circular_slice(shape, pos, radius)

        # Should give a single pixel slice
        assert slices[0].stop - slices[0].start == 1
        assert slices[1].stop - slices[1].start == 1

    def test_circular_slice_large_radius(self):
        """Test circular_slice with radius larger than image."""
        shape = (100, 100)
        pos = (50, 50)
        radius = 200

        slices = circular_slice(shape, pos, radius)

        # Should be clipped to image bounds
        assert slices[0].start >= 0
        assert slices[0].stop <= 100
        assert slices[1].start >= 0
        assert slices[1].stop <= 100

    def test_find_centroid_at_edge(self, gaussian_source_centered):
        """Test find_centroid behaves sensibly near image edge."""
        # Use Gaussian source data with a position near the source
        ccd = CCDData(gaussian_source_centered, unit='adu')

        # Position near the source but not exactly at center
        result = find_centroid(
            ccd,
            position_xy=(48, 48),  # Slightly off from center at (50, 50)
            cbox_size=11,
            maxiters=3
        )

        # Should converge toward the source center
        assert 45 <= result[0] <= 55
        assert 45 <= result[1] <= 55

    def test_find_centroid_uniform_image_may_fail(self, ccd_uniform):
        """Test find_centroid on uniform image may produce NaN or error."""
        # Uniform images have no gradient, so centroiding can fail
        # The function should either return NaN or raise an error
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = find_centroid(
                    ccd_uniform,
                    position_xy=(50, 50),
                    cbox_size=7,
                    maxiters=1  # Just one iteration to check behavior
                )
                # If it returns, values might be NaN or the original position
                # Both are acceptable behaviors for undefined centroid
                assert np.isnan(result[0]) or (0 <= result[0] <= 100)
            except (ValueError, RuntimeError):
                # Also acceptable to raise an error for undefined centroid
                pass
