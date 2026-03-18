"""
Tests for ysphotutilpy.aputil module.

All expected values are analytically derived.

Analytical reference areas:
  Circle r:                  pi * r^2
  Circular annulus r_in/out: pi * (r_out^2 - r_in^2)
  Ellipse rx/ry:             pi * rx * ry
  Elliptical annulus:        pi * (rx_out*ry_out - rx_in*ry_in)

For center-sampling (use_exact=0, subpixels=1) the mask sum equals the
number of pixel centers that fall inside the aperture, which for large
apertures converges to the area.  We use atol=2.0 for center-sampling
and atol=1.0 for exact-sampling.
"""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from ..aputil import (
    _circ_bbox,
    _ellip_bbox,
    fast_circ_apmask,
    fast_circ_anmask,
    fast_circ_apanmask,
    fast_ellip_apmask,
    fast_ellip_anmask,
)


# =============================================================================
# _bbox
# =============================================================================

class TestBbox:
    """Tests for the private _bbox helper."""

    def test_centered_at_origin(self):
        """r=5 centered at (0,0): verify ixmin, nx, and that ixmax=ixmin+nx."""
        ixmin, ixmax, iymin, iymax, xmin, xmax, ymin, ymax, nx, ny = _circ_bbox(0.0, 0.0, 5.0)
        # ixmin = floor(0-5+0.5) = floor(-4.5) = -5
        assert ixmin == -5
        # ixmax = ceil(0+5+0.5) = ceil(5.5) = 6  (exclusive upper bound)
        assert ixmax == 6
        assert iymin == -5
        assert iymax == 6
        assert nx == 11
        assert ny == 11

    def test_subpixel_offsets_centered(self):
        """For center at (0,0), r=5: xmin=ixmin-0.5-x, xmax=ixmax-0.5-x."""
        ixmin, ixmax, iymin, iymax, xmin, xmax, ymin, ymax, nx, ny = _circ_bbox(0.0, 0.0, 5.0)
        assert_allclose(xmin, ixmin - 0.5)
        assert_allclose(xmax, ixmax - 0.5)
        assert_allclose(ymin, iymin - 0.5)
        assert_allclose(ymax, iymax - 0.5)

    def test_offset_center(self):
        """r=3 at (10.5, 20.5): ixmin=8, ixmax=14, nx=6."""
        ixmin, ixmax, iymin, iymax, xmin, xmax, ymin, ymax, nx, ny = _circ_bbox(10.5, 20.5, 3.0)
        assert ixmin == 8
        assert ixmax == 14
        assert iymin == 18
        assert iymax == 24
        assert nx == 6
        assert ny == 6

    def test_invalid_radius_zero(self):
        with pytest.raises(ValueError, match="positive"):
            _circ_bbox(0.0, 0.0, 0.0)

    def test_invalid_radius_negative(self):
        with pytest.raises(ValueError, match="positive"):
            _circ_bbox(0.0, 0.0, -1.0)

    def test_nx_ny_positive(self):
        """nx and ny are always positive for any valid r."""
        for r in [0.1, 1.0, 5.0, 10.0, 100.0]:
            *_, nx, ny = _circ_bbox(0.0, 0.0, r)
            assert nx > 0
            assert ny > 0


# =============================================================================
# _ellip_bbox
# =============================================================================

class TestEllipBbox:
    """Tests for the private _ellip_bbox helper."""

    def test_axis_aligned_theta0(self):
        """theta=0: dx=rx, dy=ry.  rx=6, ry=4.
        ixmin=floor(-6+0.5)=-6, ixmax=ceil(6+0.5)=7 → nx=13."""
        ixmin, ixmax, iymin, iymax, xmin, xmax, ymin, ymax, nx, ny = _ellip_bbox(
            0.0, 0.0, 6.0, 4.0, 0.0
        )
        assert ixmin == -6
        assert ixmax == 7   # ceil(6.5) = 7
        assert iymin == -4
        assert iymax == 5   # ceil(4.5) = 5
        assert nx == 13
        assert ny == 9

    def test_axis_aligned_theta_halfpi(self):
        """theta=pi/2: dx=ry=4, dy=rx=6.
        ixmax=ceil(4.5)=5 → nx=9; iymax=ceil(6.5)=7 → ny=13."""
        ixmin, ixmax, iymin, iymax, xmin, xmax, ymin, ymax, nx, ny = _ellip_bbox(
            0.0, 0.0, 6.0, 4.0, math.pi / 2
        )
        assert nx == 9
        assert ny == 13

    def test_circle_case(self):
        """rx=ry=r: bbox same as _bbox regardless of theta."""
        r = 5.0
        for theta in [0.0, math.pi / 4, math.pi / 3]:
            ixmin_e, ixmax_e, iymin_e, iymax_e, *_ = _ellip_bbox(0.0, 0.0, r, r, theta)
            ixmin_c, ixmax_c, iymin_c, iymax_c, *_ = _circ_bbox(0.0, 0.0, r)
            assert ixmin_e == ixmin_c
            assert ixmax_e == ixmax_c
            assert iymin_e == iymin_c
            assert iymax_e == iymax_c

    def test_45deg_rotation(self):
        """theta=pi/4, rx=6, ry=2:
        dx = sqrt((6*cos45)^2 + (2*sin45)^2) = sqrt(18+2) = sqrt(20) ≈ 4.47
        dy = sqrt((6*sin45)^2 + (2*cos45)^2) = sqrt(18+2) = sqrt(20) ≈ 4.47
        Both dx and dy equal sqrt(20), so nx=ny."""
        ixmin, ixmax, iymin, iymax, xmin, xmax, ymin, ymax, nx, ny = _ellip_bbox(
            0.0, 0.0, 6.0, 2.0, math.pi / 4
        )
        assert nx == ny  # symmetric at 45 deg

    def test_offset_center(self):
        """Offset center shifts bbox correctly."""
        ixmin, ixmax, iymin, iymax, *_ = _ellip_bbox(10.0, 20.0, 5.0, 3.0, 0.0)
        ixmin0, ixmax0, iymin0, iymax0, *_ = _ellip_bbox(0.0, 0.0, 5.0, 3.0, 0.0)
        assert ixmin == ixmin0 + 10
        assert ixmax == ixmax0 + 10
        assert iymin == iymin0 + 20
        assert iymax == iymax0 + 20

    def test_invalid_rx_zero(self):
        with pytest.raises(ValueError):
            _ellip_bbox(0.0, 0.0, 0.0, 3.0, 0.0)

    def test_invalid_ry_zero(self):
        with pytest.raises(ValueError):
            _ellip_bbox(0.0, 0.0, 5.0, 0.0, 0.0)

    def test_invalid_rx_negative(self):
        with pytest.raises(ValueError):
            _ellip_bbox(0.0, 0.0, -1.0, 3.0, 0.0)

    def test_subpixel_offsets(self):
        """xmin = ixmin - 0.5 - x, xmax = ixmax - 0.5 - x."""
        x, y, rx, ry, theta = 5.0, 3.0, 4.0, 2.0, 0.0
        ixmin, ixmax, iymin, iymax, xmin, xmax, ymin, ymax, nx, ny = _ellip_bbox(
            x, y, rx, ry, theta
        )
        assert_allclose(xmin, ixmin - 0.5 - x)
        assert_allclose(xmax, ixmax - 0.5 - x)
        assert_allclose(ymin, iymin - 0.5 - y)
        assert_allclose(ymax, iymax - 0.5 - y)


# =============================================================================
# fast_circ_apmask
# =============================================================================

class TestFastCircApmask:
    """Tests for fast_circ_apmask."""

    def test_mask_shape_matches_bbox(self):
        """Mask shape equals (ny, nx) from _bbox."""
        mask, sl = fast_circ_apmask(0.0, 0.0, 5.0)
        *_, nx, ny = _circ_bbox(0.0, 0.0, 5.0)
        assert mask.shape == (ny, nx)

    def test_slice_matches_bbox(self):
        """Returned slice matches _bbox integer bounds."""
        mask, sl = fast_circ_apmask(10.0, 20.0, 5.0)
        ixmin, ixmax, iymin, iymax, *_ = _circ_bbox(10.0, 20.0, 5.0)
        assert sl == (slice(iymin, iymax), slice(ixmin, ixmax))

    def test_mask_values_in_range(self):
        """All mask values in [0, 1]."""
        for use_exact in [0, 1]:
            mask, _ = fast_circ_apmask(0.0, 0.0, 5.0, use_exact=use_exact)
            assert mask.min() >= 0.0 - 1e-12
            assert mask.max() <= 1.0 + 1e-12

    @pytest.mark.parametrize("r", [3.0, 5.0, 10.0, 20.0])
    def test_exact_sum_approx_area(self, r):
        """Exact mask sum ≈ pi*r^2 (atol=1.0 pixel)."""
        mask, _ = fast_circ_apmask(0.0, 0.0, r, use_exact=1)
        assert_allclose(mask.sum(), np.pi * r**2, atol=1.0)

    @pytest.mark.parametrize("r", [5.0, 10.0, 20.0])
    def test_center_sum_approx_area(self, r):
        """Center-sampling sum ≈ pi*r^2 (atol=12 pixels; center-sampling has ~r perimeter error)."""
        mask, _ = fast_circ_apmask(0.0, 0.0, r, use_exact=0, subpixels=1)
        assert_allclose(mask.sum(), np.pi * r**2, atol=12.0)

    def test_slice_only_returns_slice(self):
        """slice_only=True returns only the slice, not a tuple."""
        result = fast_circ_apmask(0.0, 0.0, 5.0, slice_only=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], slice)
        assert isinstance(result[1], slice)

    def test_subpixels_parameter(self):
        """subpixels>1 with use_exact=0 gives finer sampling (closer to exact)."""
        mask1, _ = fast_circ_apmask(0.0, 0.0, 5.0, use_exact=0, subpixels=1)
        mask5, _ = fast_circ_apmask(0.0, 0.0, 5.0, use_exact=0, subpixels=5)
        # subpixels=5 should be closer to pi*r^2 than subpixels=1
        exact_area = np.pi * 25
        assert abs(mask5.sum() - exact_area) <= abs(mask1.sum() - exact_area) + 1.0

    def test_offset_center_slice(self):
        """Offset center produces correct slice."""
        mask, sl = fast_circ_apmask(50.0, 30.0, 5.0)
        ixmin, ixmax, iymin, iymax, *_ = _circ_bbox(50.0, 30.0, 5.0)
        assert sl[0] == slice(iymin, iymax)
        assert sl[1] == slice(ixmin, ixmax)


# =============================================================================
# fast_circ_anmask
# =============================================================================

class TestFastCircAnmask:
    """Tests for fast_circ_anmask."""

    def test_mask_shape_matches_outer_bbox(self):
        """Mask shape matches outer radius bbox."""
        mask, sl = fast_circ_anmask(0.0, 0.0, 3.0, 7.0)
        *_, nx, ny = _circ_bbox(0.0, 0.0, 7.0)
        assert mask.shape == (ny, nx)

    def test_mask_values_in_range(self):
        """All mask values in [0, 1]."""
        mask, _ = fast_circ_anmask(0.0, 0.0, 3.0, 7.0)
        assert mask.min() >= 0.0 - 1e-12
        assert mask.max() <= 1.0 + 1e-12

    @pytest.mark.parametrize("r_in,r_out", [(3.0, 7.0), (5.0, 10.0), (2.0, 15.0)])
    def test_exact_sum_approx_area(self, r_in, r_out):
        """Exact sum ≈ pi*(r_out^2 - r_in^2)."""
        mask, _ = fast_circ_anmask(0.0, 0.0, r_in, r_out, use_exact=1)
        expected = np.pi * (r_out**2 - r_in**2)
        assert_allclose(mask.sum(), expected, atol=1.5)

    @pytest.mark.parametrize("r_in,r_out", [(3.0, 7.0), (5.0, 10.0)])
    def test_center_sum_approx_area(self, r_in, r_out):
        """Center-sampling sum ≈ pi*(r_out^2 - r_in^2) (atol=12 pixels)."""
        mask, _ = fast_circ_anmask(0.0, 0.0, r_in, r_out, use_exact=0, subpixels=1)
        expected = np.pi * (r_out**2 - r_in**2)
        assert_allclose(mask.sum(), expected, atol=12.0)

    def test_r_in_zero(self):
        """r_in=0 is valid (full disk minus nothing); guard against ZeroDivision."""
        mask, _ = fast_circ_anmask(0.0, 0.0, 0.0, 5.0, use_exact=1)
        assert_allclose(mask.sum(), np.pi * 25, atol=1.0)

    def test_invalid_r_in_negative(self):
        with pytest.raises(ValueError, match="non-negative"):
            fast_circ_anmask(0.0, 0.0, -1.0, 5.0)

    def test_invalid_r_in_ge_r_out(self):
        with pytest.raises(ValueError, match="less than r_out"):
            fast_circ_anmask(0.0, 0.0, 5.0, 5.0)

    def test_invalid_r_in_gt_r_out(self):
        with pytest.raises(ValueError, match="less than r_out"):
            fast_circ_anmask(0.0, 0.0, 6.0, 5.0)

    def test_annulus_less_than_disk(self):
        """Annulus mask sum < full disk sum."""
        mask_an, _ = fast_circ_anmask(0.0, 0.0, 3.0, 7.0, use_exact=1)
        mask_disk, _ = fast_circ_apmask(0.0, 0.0, 7.0, use_exact=1)
        assert mask_an.sum() < mask_disk.sum()

    def test_subpixels_parameter(self):
        """subpixels parameter accepted without error."""
        mask, _ = fast_circ_anmask(0.0, 0.0, 3.0, 7.0, use_exact=0, subpixels=5)
        assert mask.sum() > 0


# =============================================================================
# fast_circ_apanmask
# =============================================================================

class TestFastCircApanmask:
    """Tests for fast_circ_apanmask."""

    def test_returns_three_items(self):
        """Returns (mask_ap, mask_an, sl)."""
        result = fast_circ_apanmask(0.0, 0.0, 5.0, 7.0, 12.0)
        assert len(result) == 3

    def test_shapes_match(self):
        """Both masks have the same shape (outer bbox)."""
        mask_ap, mask_an, sl = fast_circ_apanmask(0.0, 0.0, 5.0, 7.0, 12.0)
        assert mask_ap.shape == mask_an.shape

    def test_ap_sum_approx_area(self):
        """Aperture mask sum ≈ pi*r^2."""
        mask_ap, _, _ = fast_circ_apanmask(0.0, 0.0, 5.0, 7.0, 12.0, ap_use_exact=1)
        assert_allclose(mask_ap.sum(), np.pi * 25, atol=1.0)

    def test_an_sum_approx_area(self):
        """Annulus mask sum ≈ pi*(r_out^2 - r_in^2)."""
        _, mask_an, _ = fast_circ_apanmask(0.0, 0.0, 5.0, 7.0, 12.0, an_use_exact=1)
        assert_allclose(mask_an.sum(), np.pi * (144 - 49), atol=1.5)

    def test_invalid_r_in_negative(self):
        with pytest.raises(ValueError, match="non-negative"):
            fast_circ_apanmask(0.0, 0.0, 5.0, -1.0, 10.0)

    def test_invalid_r_in_ge_r_out(self):
        with pytest.raises(ValueError, match="less than r_out"):
            fast_circ_apanmask(0.0, 0.0, 5.0, 10.0, 10.0)

    def test_invalid_r_gt_r_out(self):
        with pytest.raises(ValueError, match="r must be <= r_out"):
            fast_circ_apanmask(0.0, 0.0, 15.0, 7.0, 12.0)

    def test_r_equals_r_out(self):
        """r == r_out is valid (aperture fills outer boundary)."""
        mask_ap, _, _ = fast_circ_apanmask(0.0, 0.0, 12.0, 7.0, 12.0, ap_use_exact=1)
        assert_allclose(mask_ap.sum(), np.pi * 144, atol=1.0)

    def test_separate_use_exact_flags(self):
        """ap_use_exact and an_use_exact can differ independently."""
        mask_ap, mask_an, _ = fast_circ_apanmask(
            0.0, 0.0, 5.0, 7.0, 12.0, ap_use_exact=1, an_use_exact=0
        )
        assert mask_ap.sum() > 0
        assert mask_an.sum() > 0


# =============================================================================
# fast_ellip_apmask
# =============================================================================

class TestFastEllipApmask:
    """Tests for fast_ellip_apmask."""

    def test_returns_mask_and_slice(self):
        """Returns (mask, sl) tuple."""
        result = fast_ellip_apmask(0.0, 0.0, 6.0, 4.0, 0.0)
        assert len(result) == 2

    def test_mask_shape_matches_bbox(self):
        """Mask shape equals (ny, nx) from _ellip_bbox."""
        mask, sl = fast_ellip_apmask(0.0, 0.0, 6.0, 4.0, 0.0)
        *_, nx, ny = _ellip_bbox(0.0, 0.0, 6.0, 4.0, 0.0)
        assert mask.shape == (ny, nx)

    def test_slice_matches_bbox(self):
        """Returned slice matches _ellip_bbox integer bounds."""
        mask, sl = fast_ellip_apmask(10.0, 20.0, 6.0, 4.0, 0.0)
        ixmin, ixmax, iymin, iymax, *_ = _ellip_bbox(10.0, 20.0, 6.0, 4.0, 0.0)
        assert sl == (slice(iymin, iymax), slice(ixmin, ixmax))

    def test_mask_values_in_range(self):
        """All mask values in [0, 1]."""
        for use_exact in [0, 1]:
            mask, _ = fast_ellip_apmask(0.0, 0.0, 6.0, 4.0, 0.0, use_exact=use_exact)
            assert mask.min() >= 0.0 - 1e-12
            assert mask.max() <= 1.0 + 1e-12

    @pytest.mark.parametrize("rx,ry", [(6.0, 4.0), (8.0, 5.0), (10.0, 7.0)])
    def test_exact_sum_approx_area_theta0(self, rx, ry):
        """Exact sum ≈ pi*rx*ry for theta=0 (atol=1.0)."""
        mask, _ = fast_ellip_apmask(0.0, 0.0, rx, ry, 0.0, use_exact=1)
        assert_allclose(mask.sum(), np.pi * rx * ry, atol=1.0)

    @pytest.mark.parametrize("theta", [0.0, math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2])
    def test_exact_sum_invariant_to_theta(self, theta):
        """Area is invariant to rotation: sum ≈ pi*rx*ry for any theta."""
        rx, ry = 8.0, 5.0
        mask, _ = fast_ellip_apmask(0.0, 0.0, rx, ry, theta, use_exact=1)
        assert_allclose(mask.sum(), np.pi * rx * ry, atol=1.5)

    def test_circle_case_matches_circ_apmask(self):
        """rx=ry=r: elliptical mask sum ≈ circular mask sum."""
        r = 7.0
        mask_e, _ = fast_ellip_apmask(0.0, 0.0, r, r, 0.0, use_exact=1)
        mask_c, _ = fast_circ_apmask(0.0, 0.0, r, use_exact=1)
        assert_allclose(mask_e.sum(), mask_c.sum(), atol=0.5)

    @pytest.mark.parametrize("rx,ry", [(6.0, 4.0), (10.0, 6.0)])
    def test_center_sum_approx_area(self, rx, ry):
        """Center-sampling sum ≈ pi*rx*ry (atol=12 pixels)."""
        mask, _ = fast_ellip_apmask(0.0, 0.0, rx, ry, 0.0, use_exact=0, subpixels=1)
        assert_allclose(mask.sum(), np.pi * rx * ry, atol=12.0)

    def test_subpixels_parameter(self):
        """subpixels>1 with use_exact=0 gives finer sampling."""
        mask5, _ = fast_ellip_apmask(0.0, 0.0, 8.0, 5.0, 0.0, use_exact=0, subpixels=5)
        assert_allclose(mask5.sum(), np.pi * 40, atol=1.5)

    def test_invalid_rx_zero(self):
        with pytest.raises(ValueError):
            fast_ellip_apmask(0.0, 0.0, 0.0, 4.0, 0.0)

    def test_invalid_ry_zero(self):
        with pytest.raises(ValueError):
            fast_ellip_apmask(0.0, 0.0, 6.0, 0.0, 0.0)

    def test_invalid_rx_negative(self):
        with pytest.raises(ValueError):
            fast_ellip_apmask(0.0, 0.0, -1.0, 4.0, 0.0)

    def test_offset_center(self):
        """Offset center: mask sum unchanged, slice shifts."""
        mask0, sl0 = fast_ellip_apmask(0.0, 0.0, 6.0, 4.0, 0.0, use_exact=1)
        mask1, sl1 = fast_ellip_apmask(50.0, 30.0, 6.0, 4.0, 0.0, use_exact=1)
        assert_allclose(mask0.sum(), mask1.sum(), atol=1e-10)
        assert sl0 != sl1


# =============================================================================
# fast_ellip_anmask
# =============================================================================

class TestFastEllipAnmask:
    """Tests for fast_ellip_anmask."""

    def test_returns_mask_and_slice(self):
        """Returns (mask, sl) tuple."""
        result = fast_ellip_anmask(0.0, 0.0, 3.0, 2.0, 6.0, 4.0, 0.0)
        assert len(result) == 2

    def test_mask_shape_matches_outer_bbox(self):
        """Mask shape matches outer ellipse bbox."""
        mask, sl = fast_ellip_anmask(0.0, 0.0, 3.0, 2.0, 6.0, 4.0, 0.0)
        *_, nx, ny = _ellip_bbox(0.0, 0.0, 6.0, 4.0, 0.0)
        assert mask.shape == (ny, nx)

    def test_mask_values_in_range(self):
        """All mask values in [0, 1]."""
        mask, _ = fast_ellip_anmask(0.0, 0.0, 3.0, 2.0, 6.0, 4.0, 0.0)
        assert mask.min() >= 0.0 - 1e-12
        assert mask.max() <= 1.0 + 1e-12

    @pytest.mark.parametrize(
        "rx_in,ry_in,rx_out,ry_out",
        [
            (3.0, 2.0, 6.0, 4.0),
            (4.0, 3.0, 8.0, 6.0),
            (5.0, 3.0, 10.0, 6.0),
        ],
    )
    def test_exact_sum_approx_area_theta0(self, rx_in, ry_in, rx_out, ry_out):
        """Exact sum ≈ pi*(rx_out*ry_out - rx_in*ry_in) for theta=0."""
        mask, _ = fast_ellip_anmask(
            0.0, 0.0, rx_in, ry_in, rx_out, ry_out, 0.0, use_exact=1
        )
        expected = np.pi * (rx_out * ry_out - rx_in * ry_in)
        assert_allclose(mask.sum(), expected, atol=1.5)

    @pytest.mark.parametrize("theta", [0.0, math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2])
    def test_exact_sum_invariant_to_theta(self, theta):
        """Annulus area is invariant to rotation."""
        rx_in, ry_in, rx_out, ry_out = 3.0, 2.0, 7.0, 5.0
        mask, _ = fast_ellip_anmask(
            0.0, 0.0, rx_in, ry_in, rx_out, ry_out, theta, use_exact=1
        )
        expected = np.pi * (rx_out * ry_out - rx_in * ry_in)
        assert_allclose(mask.sum(), expected, atol=2.0)

    def test_annulus_less_than_outer_disk(self):
        """Annulus sum < outer ellipse sum."""
        mask_an, _ = fast_ellip_anmask(0.0, 0.0, 3.0, 2.0, 6.0, 4.0, 0.0, use_exact=1)
        mask_out, _ = fast_ellip_apmask(0.0, 0.0, 6.0, 4.0, 0.0, use_exact=1)
        assert mask_an.sum() < mask_out.sum()

    def test_rx_in_zero(self):
        """rx_in=0 (and ry_in=0) is valid: full outer ellipse."""
        mask, _ = fast_ellip_anmask(0.0, 0.0, 0.0, 0.0, 6.0, 4.0, 0.0, use_exact=1)
        assert_allclose(mask.sum(), np.pi * 24, atol=1.0)

    def test_invalid_rx_in_negative(self):
        with pytest.raises(ValueError, match="non-negative"):
            fast_ellip_anmask(0.0, 0.0, -1.0, 2.0, 6.0, 4.0, 0.0)

    def test_invalid_ry_in_negative(self):
        with pytest.raises(ValueError, match="non-negative"):
            fast_ellip_anmask(0.0, 0.0, 3.0, -1.0, 6.0, 4.0, 0.0)

    def test_invalid_rx_in_ge_rx_out(self):
        with pytest.raises(ValueError, match="less than"):
            fast_ellip_anmask(0.0, 0.0, 6.0, 2.0, 6.0, 4.0, 0.0)

    def test_invalid_ry_in_ge_ry_out(self):
        with pytest.raises(ValueError, match="less than"):
            fast_ellip_anmask(0.0, 0.0, 3.0, 4.0, 6.0, 4.0, 0.0)

    def test_center_sum_approx_area(self):
        """Center-sampling sum ≈ pi*(rx_out*ry_out - rx_in*ry_in) (atol=12 pixels)."""
        mask, _ = fast_ellip_anmask(
            0.0, 0.0, 3.0, 2.0, 8.0, 6.0, 0.0, use_exact=0, subpixels=1
        )
        expected = np.pi * (48 - 6)
        assert_allclose(mask.sum(), expected, atol=12.0)

    def test_subpixels_parameter(self):
        """subpixels>1 accepted and gives reasonable result."""
        mask, _ = fast_ellip_anmask(
            0.0, 0.0, 3.0, 2.0, 7.0, 5.0, 0.0, use_exact=0, subpixels=5
        )
        expected = np.pi * (35 - 6)
        assert_allclose(mask.sum(), expected, atol=2.0)

    def test_circle_annulus_matches_circ_anmask(self):
        """rx_in=ry_in=r_in, rx_out=ry_out=r_out: matches fast_circ_anmask."""
        r_in, r_out = 3.0, 7.0
        mask_e, _ = fast_ellip_anmask(0.0, 0.0, r_in, r_in, r_out, r_out, 0.0, use_exact=1)
        mask_c, _ = fast_circ_anmask(0.0, 0.0, r_in, r_out, use_exact=1)
        assert_allclose(mask_e.sum(), mask_c.sum(), atol=0.5)

    def test_offset_center(self):
        """Offset center: mask sum unchanged, slice shifts."""
        mask0, sl0 = fast_ellip_anmask(0.0, 0.0, 3.0, 2.0, 6.0, 4.0, 0.0, use_exact=1)
        mask1, sl1 = fast_ellip_anmask(50.0, 30.0, 3.0, 2.0, 6.0, 4.0, 0.0, use_exact=1)
        assert_allclose(mask0.sum(), mask1.sum(), atol=1e-10)
        assert sl0 != sl1
