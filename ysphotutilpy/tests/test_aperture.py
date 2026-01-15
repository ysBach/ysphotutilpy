"""
Tests for ysphotutilpy.aperture module.

All expected values are analytically derived.
"""

import numpy as np
from astropy.nddata import CCDData, Cutout2D
from numpy.testing import assert_allclose
from photutils.aperture import (
    CircularAperture,
    CircularAnnulus,
    EllipticalAperture,
    RectangularAperture,
)

from ..aperture import (
    circ_ap_an,
    ellip_ap_an,
    pill_ap_an,
    cutout_from_ap,
    ap_to_cutout_position,
    PillBoxAperture,
    PillBoxAnnulus,
)


class TestCircApAn:
    """Tests for circ_ap_an function."""

    def test_circ_ap_an_with_fwhm(self):
        """
        Test circular aperture/annulus with FWHM parameter.

        FWHM = 10, f_ap = 1.5, f_in = 4.0, f_out = 6.0
        Expected: r_ap = 15, r_in = 40, r_out = 60
        """
        positions = (50, 50)
        ap, an = circ_ap_an(positions, fwhm=10, f_ap=1.5, f_in=4.0, f_out=6.0)

        assert_allclose(ap.r, 15.0, rtol=1e-10)
        assert_allclose(an.r_in, 40.0, rtol=1e-10)
        assert_allclose(an.r_out, 60.0, rtol=1e-10)

    def test_circ_ap_an_with_explicit_radii(self):
        """
        Test circular aperture/annulus with explicit radii.

        r_ap = 5, r_in = 10, r_out = 15
        """
        positions = (50, 50)
        ap, an = circ_ap_an(positions, r_ap=5, r_in=10, r_out=15)

        assert_allclose(ap.r, 5.0, rtol=1e-10)
        assert_allclose(an.r_in, 10.0, rtol=1e-10)
        assert_allclose(an.r_out, 15.0, rtol=1e-10)

    def test_circ_ap_an_area(self):
        """
        Test aperture area calculation.

        CircularAperture area = pi * r^2
        For r = 5: area = pi * 25 = 78.5398...
        """
        positions = (50, 50)
        ap, an = circ_ap_an(positions, r_ap=5, r_in=10, r_out=15)

        expected_ap_area = np.pi * 5**2
        expected_an_area = np.pi * (15**2 - 10**2)  # pi * (225 - 100) = pi * 125

        assert_allclose(ap.area, expected_ap_area, rtol=1e-10)
        assert_allclose(an.area, expected_an_area, rtol=1e-10)

    def test_circ_ap_an_positions(self):
        """Test position is correctly set."""
        positions = (30.5, 40.7)
        ap, an = circ_ap_an(positions, r_ap=5, r_in=10, r_out=15)

        assert_allclose(ap.positions[0], 30.5, rtol=1e-10)
        assert_allclose(ap.positions[1], 40.7, rtol=1e-10)


class TestEllipApAn:
    """Tests for ellip_ap_an function."""

    def test_ellip_ap_an_with_fwhm(self):
        """
        Test elliptical aperture/annulus with FWHM parameter.

        FWHM = 10, f_ap = (1.5, 1.5), f_in = (4.0, 4.0), f_out = (6.0, 6.0)
        Expected: a_ap = b_ap = 15, a_in = b_in = 40, a_out = b_out = 60
        """
        positions = (50, 50)
        ap, an = ellip_ap_an(
            positions, fwhm=10,
            f_ap=(1.5, 1.5), f_in=(4.0, 4.0), f_out=(6.0, 6.0)
        )

        assert_allclose(ap.a, 15.0, rtol=1e-10)
        assert_allclose(ap.b, 15.0, rtol=1e-10)
        assert_allclose(an.a_in, 40.0, rtol=1e-10)
        assert_allclose(an.a_out, 60.0, rtol=1e-10)

    def test_ellip_ap_an_asymmetric(self):
        """
        Test elliptical aperture with asymmetric factors.

        FWHM = 10, f_ap = (2.0, 1.0)
        Expected: a_ap = 20, b_ap = 10
        """
        positions = (50, 50)
        ap, an = ellip_ap_an(
            positions, fwhm=10,
            f_ap=(2.0, 1.0), f_in=(4.0, 4.0), f_out=(6.0, 6.0)
        )

        assert_allclose(ap.a, 20.0, rtol=1e-10)
        assert_allclose(ap.b, 10.0, rtol=1e-10)

    def test_ellip_ap_an_area(self):
        """
        Test elliptical aperture area.

        EllipticalAperture area = pi * a * b
        For a = 10, b = 5: area = pi * 50 = 157.0796...
        """
        positions = (50, 50)
        ap, an = ellip_ap_an(positions, r_ap=(10, 5), r_in=(20, 10), r_out=(30, 15))

        expected_ap_area = np.pi * 10 * 5
        assert_allclose(ap.area, expected_ap_area, rtol=1e-10)

    def test_ellip_ap_an_theta(self):
        """Test rotation angle is correctly set."""
        positions = (50, 50)
        theta = np.pi / 4  # 45 degrees
        ap, an = ellip_ap_an(positions, r_ap=10, r_in=20, r_out=30, theta=theta)

        # theta is a Quantity with units of rad, extract value for comparison
        assert_allclose(ap.theta.value, theta, rtol=1e-10)
        assert_allclose(an.theta.value, theta, rtol=1e-10)


class TestPillBoxAperture:
    """Tests for PillBoxAperture class."""

    def test_pillbox_area(self):
        """
        Test PillBoxAperture area calculation.

        Area = w * h + pi * a * b
        where h = 2 * b

        For w = 10, a = 5, b = 3:
        h = 6
        Area = 10 * 6 + pi * 5 * 3 = 60 + 15*pi = 60 + 47.1238... = 107.1238...
        """
        pb = PillBoxAperture(positions=(50, 50), w=10, a=5, b=3, theta=0)

        expected_area = 10 * 6 + np.pi * 5 * 3
        assert_allclose(pb.area, expected_area, rtol=1e-10)

    def test_pillbox_h_parameter(self):
        """Test that h = 2 * b."""
        pb = PillBoxAperture(positions=(50, 50), w=10, a=5, b=3, theta=0)
        assert_allclose(pb.h, 6.0, rtol=1e-10)

    def test_pillbox_positions(self):
        """Test position is correctly set."""
        pb = PillBoxAperture(positions=(30.5, 40.7), w=10, a=5, b=3, theta=0)
        pos = np.atleast_2d(pb.positions)
        assert_allclose(pos[0, 0], 30.5, rtol=1e-10)
        assert_allclose(pos[0, 1], 40.7, rtol=1e-10)

    def test_pillbox_theta(self):
        """Test rotation angle."""
        theta = np.pi / 6  # 30 degrees
        pb = PillBoxAperture(positions=(50, 50), w=10, a=5, b=3, theta=theta)
        # theta is a Quantity with units of rad, extract value for comparison
        assert_allclose(pb.theta.value, theta, rtol=1e-10)


class TestPillBoxAnnulus:
    """Tests for PillBoxAnnulus class."""

    def test_pillbox_annulus_area(self):
        """
        Test PillBoxAnnulus area calculation.

        Area = w * (h_out - h_in) + pi * (a_out * b_out - a_in * b_in)

        For w = 10, a_in = 3, a_out = 6, b_out = 4:
        b_in = b_out * a_in / a_out = 4 * 3 / 6 = 2
        h_out = 2 * b_out = 8
        h_in = 2 * b_in = 4

        Area = 10 * (8 - 4) + pi * (6*4 - 3*2)
             = 10 * 4 + pi * (24 - 6)
             = 40 + 18*pi
             = 40 + 56.5486... = 96.5486...
        """
        pba = PillBoxAnnulus(positions=(50, 50), w=10, a_in=3, a_out=6, b_out=4, theta=0)

        b_in = 4 * 3 / 6  # = 2
        expected_area = 10 * (8 - 4) + np.pi * (6 * 4 - 3 * b_in)
        assert_allclose(pba.area, expected_area, rtol=1e-10)

    def test_pillbox_annulus_b_in(self):
        """Test that b_in is calculated correctly as b_out * a_in / a_out."""
        pba = PillBoxAnnulus(positions=(50, 50), w=10, a_in=3, a_out=6, b_out=4, theta=0)
        expected_b_in = 4 * 3 / 6  # = 2
        assert_allclose(pba.b_in, expected_b_in, rtol=1e-10)

    def test_pillbox_annulus_h_parameters(self):
        """Test h_out and h_in calculations."""
        pba = PillBoxAnnulus(positions=(50, 50), w=10, a_in=3, a_out=6, b_out=4, theta=0)

        assert_allclose(pba.h_out, 8.0, rtol=1e-10)  # 2 * b_out = 2 * 4
        assert_allclose(pba.h_in, 4.0, rtol=1e-10)   # 2 * b_in = 2 * 2


class TestPillApAn:
    """Tests for pill_ap_an convenience function."""

    def test_pill_ap_an_basic(self):
        """Test pill_ap_an creates PillBoxAperture and PillBoxAnnulus."""
        positions = (50, 50)
        fwhm = 5
        trail = 10

        ap, an = pill_ap_an(
            positions, fwhm=fwhm, trail=trail, theta=0,
            f_ap=(1.5, 1.5), f_in=(4.0, 4.0), f_out=(6.0, 6.0), f_w=1.0
        )

        assert isinstance(ap, PillBoxAperture)
        assert isinstance(an, PillBoxAnnulus)

    def test_pill_ap_an_dimensions(self):
        """
        Test pill_ap_an dimensions.

        fwhm = 5, trail = 10, f_ap = (1.5, 1.5), f_w = 1.0
        a_ap = 1.5 * 5 = 7.5
        b_ap = 1.5 * 5 = 7.5
        w = 1.0 * 10 = 10
        """
        positions = (50, 50)
        ap, an = pill_ap_an(
            positions, fwhm=5, trail=10, theta=0,
            f_ap=(1.5, 1.5), f_in=(4.0, 4.0), f_out=(6.0, 6.0), f_w=1.0
        )

        assert_allclose(ap.a, 7.5, rtol=1e-10)
        assert_allclose(ap.b, 7.5, rtol=1e-10)
        assert_allclose(ap.w, 10.0, rtol=1e-10)


class TestCutoutFromAp:
    """Tests for cutout_from_ap function."""

    def test_cutout_from_ap_shape(self, uniform_100x100):
        """Test cutout has correct shape from aperture bounding box."""
        ccd = CCDData(uniform_100x100, unit='adu')
        ap = CircularAperture((50, 50), r=10)

        cutout = cutout_from_ap(ap, ccd)

        # Bounding box for circular aperture with r=10 centered at (50,50)
        # should be approximately 21x21 (from 40 to 60 inclusive)
        assert cutout.data.shape[0] >= 20
        assert cutout.data.shape[1] >= 20

    def test_cutout_from_ap_values(self, uniform_100x100):
        """Test cutout contains correct values."""
        ccd = CCDData(uniform_100x100, unit='adu')
        ap = CircularAperture((50, 50), r=5)

        cutout = cutout_from_ap(ap, ccd)

        # All values should be 10.0 (from uniform array)
        assert_allclose(cutout.data, 10.0, rtol=1e-10)


class TestApToCutoutPosition:
    """Tests for ap_to_cutout_position function."""

    def test_position_update(self, uniform_100x100):
        """Test aperture position is correctly updated for cutout."""
        ccd = CCDData(uniform_100x100, unit='adu')
        ap = CircularAperture((50, 50), r=5)

        # Create cutout centered at (50, 50) with size 21
        cutout = Cutout2D(ccd.data, position=(50, 50), size=21)

        new_ap = ap_to_cutout_position(ap, cutout)

        # The new position should be at center of cutout (10, 10) for 21x21
        pos = np.atleast_2d(new_ap.positions)
        assert_allclose(pos[0], [10, 10], rtol=1e-10)


class TestApertureAreaFormulas:
    """Verify aperture area formulas against analytical calculations."""

    def test_circular_aperture_area(self):
        """
        CircularAperture area = pi * r^2

        For r = 7: area = pi * 49 = 153.9380...
        """
        ap = CircularAperture((0, 0), r=7)
        expected = np.pi * 49
        assert_allclose(ap.area, expected, rtol=1e-10)

    def test_circular_annulus_area(self):
        """
        CircularAnnulus area = pi * (r_out^2 - r_in^2)

        For r_in = 5, r_out = 10: area = pi * (100 - 25) = 75 * pi = 235.6194...
        """
        an = CircularAnnulus((0, 0), r_in=5, r_out=10)
        expected = np.pi * 75
        assert_allclose(an.area, expected, rtol=1e-10)

    def test_elliptical_aperture_area(self):
        """
        EllipticalAperture area = pi * a * b

        For a = 8, b = 4: area = pi * 32 = 100.5309...
        """
        ap = EllipticalAperture((0, 0), a=8, b=4, theta=0)
        expected = np.pi * 32
        assert_allclose(ap.area, expected, rtol=1e-10)

    def test_rectangular_aperture_area(self):
        """
        RectangularAperture area = w * h

        For w = 10, h = 5: area = 50
        """
        ap = RectangularAperture((0, 0), w=10, h=5, theta=0)
        expected = 50.0
        assert_allclose(ap.area, expected, rtol=1e-10)
