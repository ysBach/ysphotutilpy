"""
Tests for ysphotutilpy.aperture module.

All expected values are analytically derived or computed from known geometry.
"""

import numpy as np
import pytest
from astropy.nddata import CCDData, Cutout2D
from numpy.testing import assert_allclose
from photutils.aperture import (
    CircularAnnulus,
    CircularAperture,
    EllipticalAnnulus,
    EllipticalAperture,
    RectangularAperture,
)

from ..aperture import (
    PillBoxAnnulus,
    PillBoxAperture,
    SkyPillBoxAnnulus,
    SkyPillBoxAperture,
    ap_to_cutout_position,
    circ_ap_an,
    cutout_from_ap,
    ellip_ap_an,
    pa2xytheta,
    pill_ap_an,
)


# =============================================================================
# Helpers
# =============================================================================

def _uniform_ccd(shape=(100, 100), value=10.0):
    return CCDData(np.full(shape, value, dtype=np.float64), unit="adu")


def _make_wcs(naxis1=100, naxis2=100, cdelt=-0.5 / 3600, flip_ra=True):
    """Return a simple TAN WCS with East left (standard astronomical orientation)."""
    from astropy.wcs import WCS

    w = WCS(naxis=2)
    w.wcs.crpix = [naxis1 / 2 + 0.5, naxis2 / 2 + 0.5]
    w.wcs.crval = [180.0, 0.0]
    # flip_ra=True → RA increases to the left (standard), so CD1_1 < 0
    w.wcs.cdelt = [-abs(cdelt) if flip_ra else abs(cdelt), abs(cdelt)]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w._naxis = [naxis1, naxis2]
    return w


# =============================================================================
# circ_ap_an
# =============================================================================

class TestCircApAn:
    """Tests for circ_ap_an."""

    def test_fwhm_factors(self):
        """r = factor * fwhm for each of ap, in, out."""
        ap, an = circ_ap_an((50, 50), fwhm=10, f_ap=1.5, f_in=4.0, f_out=6.0)
        assert_allclose(ap.r, 15.0)
        assert_allclose(an.r_in, 40.0)
        assert_allclose(an.r_out, 60.0)

    def test_explicit_radii(self):
        """Explicit r_ap / r_in / r_out override fwhm."""
        ap, an = circ_ap_an((50, 50), r_ap=5, r_in=10, r_out=15)
        assert_allclose(ap.r, 5.0)
        assert_allclose(an.r_in, 10.0)
        assert_allclose(an.r_out, 15.0)

    def test_area_ap(self):
        """CircularAperture area = pi * r^2.  r=5 → 25*pi."""
        ap, _ = circ_ap_an((50, 50), r_ap=5, r_in=10, r_out=15)
        assert_allclose(ap.area, np.pi * 25)

    def test_area_an(self):
        """CircularAnnulus area = pi*(r_out^2 - r_in^2).  r_in=10, r_out=15 → 125*pi."""
        _, an = circ_ap_an((50, 50), r_ap=5, r_in=10, r_out=15)
        assert_allclose(an.area, np.pi * 125)

    def test_positions_scalar(self):
        """Single position stored correctly."""
        ap, an = circ_ap_an((30.5, 40.7), r_ap=5, r_in=10, r_out=15)
        assert_allclose(ap.positions, [30.5, 40.7])
        assert_allclose(an.positions, [30.5, 40.7])

    def test_positions_multi(self):
        """Multiple positions stored correctly."""
        pos = [(10, 20), (30, 40)]
        ap, an = circ_ap_an(pos, r_ap=3, r_in=5, r_out=8)
        assert np.atleast_2d(ap.positions).shape == (2, 2)

    def test_default_factors(self):
        """Default f_ap=1.5, f_in=4.0, f_out=6.0."""
        ap, an = circ_ap_an((50, 50), fwhm=8)
        assert_allclose(ap.r, 12.0)
        assert_allclose(an.r_in, 32.0)
        assert_allclose(an.r_out, 48.0)


# =============================================================================
# ellip_ap_an
# =============================================================================

class TestEllipApAn:
    """Tests for ellip_ap_an."""

    def test_fwhm_symmetric_factors(self):
        """Symmetric f_ap=(1.5,1.5) → a=b=1.5*fwhm."""
        ap, an = ellip_ap_an((50, 50), fwhm=10, f_ap=(1.5, 1.5), f_in=(4.0, 4.0), f_out=(6.0, 6.0))
        assert_allclose(ap.a, 15.0)
        assert_allclose(ap.b, 15.0)
        assert_allclose(an.a_in, 40.0)
        assert_allclose(an.a_out, 60.0)

    def test_fwhm_asymmetric_factors(self):
        """Asymmetric f_ap=(2.0,1.0) → a=20, b=10."""
        ap, _ = ellip_ap_an((50, 50), fwhm=10, f_ap=(2.0, 1.0), f_in=(4.0, 4.0), f_out=(6.0, 6.0))
        assert_allclose(ap.a, 20.0)
        assert_allclose(ap.b, 10.0)

    def test_explicit_radii(self):
        """Explicit r_ap=(10,5) → a=10, b=5."""
        ap, _ = ellip_ap_an((50, 50), r_ap=(10, 5), r_in=(20, 10), r_out=(30, 15))
        assert_allclose(ap.a, 10.0)
        assert_allclose(ap.b, 5.0)

    def test_area_ap(self):
        """EllipticalAperture area = pi*a*b.  a=10, b=5 → 50*pi."""
        ap, _ = ellip_ap_an((50, 50), r_ap=(10, 5), r_in=(20, 10), r_out=(30, 15))
        assert_allclose(ap.area, np.pi * 50)

    def test_area_an(self):
        """EllipticalAnnulus area = pi*(a_out*b_out - a_in*b_in).
        a_in=20,b_in=10, a_out=30,b_out=15 → pi*(450-200)=250*pi."""
        _, an = ellip_ap_an((50, 50), r_ap=(10, 5), r_in=(20, 10), r_out=(30, 15))
        assert_allclose(an.area, np.pi * 250)

    def test_theta_stored(self):
        """theta is stored correctly (as Quantity in rad)."""
        import astropy.units as u
        theta = np.pi / 4
        ap, an = ellip_ap_an((50, 50), r_ap=10, r_in=20, r_out=30, theta=theta)
        assert_allclose(ap.theta.to_value(u.rad), theta)
        assert_allclose(an.theta.to_value(u.rad), theta)

    def test_scalar_factor_broadcast(self):
        """Scalar f_ap=1.5 broadcasts to both a and b."""
        ap, _ = ellip_ap_an((50, 50), fwhm=10, f_ap=1.5, f_in=4.0, f_out=6.0)
        assert_allclose(ap.a, 15.0)
        assert_allclose(ap.b, 15.0)


# =============================================================================
# PillBoxAperture
# =============================================================================

class TestPillBoxAperture:
    """Tests for PillBoxAperture geometry and mask."""

    # --- construction ---

    def test_h_equals_2b(self):
        """h = 2*b always."""
        pb = PillBoxAperture((50, 50), w=10, a=5, b=3, theta=0)
        assert_allclose(pb.h, 6.0)

    def test_area(self):
        """Area = w*h + pi*a*b = w*(2b) + pi*a*b.
        w=10, a=5, b=3 → 60 + 15*pi."""
        pb = PillBoxAperture((50, 50), w=10, a=5, b=3, theta=0)
        assert_allclose(pb.area, 60.0 + np.pi * 15)

    def test_positions_stored(self):
        """Positions stored correctly."""
        pb = PillBoxAperture((30.5, 40.7), w=10, a=5, b=3, theta=0)
        pos = np.atleast_2d(pb.positions)
        assert_allclose(pos[0], [30.5, 40.7])

    def test_theta_stored(self):
        """theta stored correctly (as Quantity in rad)."""
        import astropy.units as u
        theta = np.pi / 6
        pb = PillBoxAperture((50, 50), w=10, a=5, b=3, theta=theta)
        assert_allclose(pb.theta.to_value(u.rad), theta)

    def test_multi_position(self):
        """Multiple positions accepted."""
        pb = PillBoxAperture([(10, 20), (30, 40)], w=5, a=3, b=2, theta=0)
        assert np.atleast_2d(pb.positions).shape == (2, 2)
        assert not pb.isscalar

    def test_isscalar_single(self):
        """Single position → isscalar=True."""
        pb = PillBoxAperture((50, 50), w=5, a=3, b=2, theta=0)
        assert pb.isscalar

    # --- to_mask ---

    def test_to_mask_returns_aperturemask(self):
        """to_mask returns ApertureMask for scalar aperture."""
        from photutils.aperture import ApertureMask
        pb = PillBoxAperture((50, 50), w=10, a=5, b=3, theta=0)
        msk = pb.to_mask(method="center")
        assert isinstance(msk, ApertureMask)

    def test_to_mask_values_in_range(self):
        """Mask values are in [0, 1]."""
        pb = PillBoxAperture((50, 50), w=10, a=5, b=3, theta=0)
        for method in ("center", "subpixel", "exact"):
            msk = pb.to_mask(method=method, subpixels=5)
            arr = msk.data
            assert arr.min() >= 0.0 - 1e-12
            assert arr.max() <= 1.0 + 1e-12

    def test_to_mask_sum_approx_area(self):
        """Sum of exact mask ≈ aperture area (within ~1 pixel tolerance)."""
        pb = PillBoxAperture((50.0, 50.0), w=10, a=5, b=3, theta=0)
        msk = pb.to_mask(method="exact")
        assert_allclose(msk.data.sum(), pb.area, atol=1.0)

    def test_to_mask_multi_returns_list(self):
        """Multi-position → list of ApertureMask."""
        from photutils.aperture import ApertureMask
        pb = PillBoxAperture([(10, 20), (30, 40)], w=5, a=3, b=2, theta=0)
        masks = pb.to_mask(method="center")
        assert isinstance(masks, list)
        assert len(masks) == 2
        for m in masks:
            assert isinstance(m, ApertureMask)

    def test_to_mask_theta_rotates(self):
        """Rotating theta=pi/2 swaps the trail direction (bbox shape changes)."""
        pb0 = PillBoxAperture((50, 50), w=20, a=3, b=3, theta=0)
        pb90 = PillBoxAperture((50, 50), w=20, a=3, b=3, theta=np.pi / 2)
        msk0 = pb0.to_mask(method="center")
        msk90 = pb90.to_mask(method="center")
        # bbox width and height should swap
        h0, w0 = msk0.data.shape
        h90, w90 = msk90.data.shape
        assert w0 == h90
        assert h0 == w90

    # --- _to_patch ---

    def test_to_patch_returns_patch(self):
        """_to_patch returns a matplotlib PathPatch."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.patches as mpatches
        pb = PillBoxAperture((50, 50), w=10, a=5, b=3, theta=0)
        patch = pb._to_patch()
        assert isinstance(patch, mpatches.PathPatch)

    def test_to_patch_multi_returns_list(self):
        """Multi-position _to_patch returns list of PathPatch."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.patches as mpatches
        pb = PillBoxAperture([(10, 20), (30, 40)], w=5, a=3, b=2, theta=0)
        patches = pb._to_patch()
        assert isinstance(patches, list)
        assert len(patches) == 2


# =============================================================================
# PillBoxAnnulus
# =============================================================================

class TestPillBoxAnnulus:
    """Tests for PillBoxAnnulus geometry and mask."""

    # --- construction ---

    def test_b_in_formula(self):
        """b_in = b_out * a_in / a_out.  a_in=3, a_out=6, b_out=4 → b_in=2."""
        pba = PillBoxAnnulus((50, 50), w=10, a_in=3, a_out=6, b_out=4, theta=0)
        assert_allclose(pba.b_in, 2.0)

    def test_h_out_h_in(self):
        """h_out=2*b_out, h_in=2*b_in."""
        pba = PillBoxAnnulus((50, 50), w=10, a_in=3, a_out=6, b_out=4, theta=0)
        assert_allclose(pba.h_out, 8.0)
        assert_allclose(pba.h_in, 4.0)

    def test_area(self):
        """Area = w*(h_out-h_in) + pi*(a_out*b_out - a_in*b_in).
        w=10, a_in=3, a_out=6, b_out=4, b_in=2:
        = 10*(8-4) + pi*(24-6) = 40 + 18*pi."""
        pba = PillBoxAnnulus((50, 50), w=10, a_in=3, a_out=6, b_out=4, theta=0)
        assert_allclose(pba.area, 40.0 + 18.0 * np.pi)

    def test_area_thin_annulus(self):
        """Very thin annulus: a_in close to a_out → area is small but positive."""
        pba = PillBoxAnnulus((50, 50), w=0.001, a_in=5.0, a_out=5.001, b_out=3.0, theta=0)
        assert pba.area >= 0

    def test_positions_stored(self):
        """Positions stored correctly."""
        pba = PillBoxAnnulus((30.5, 40.7), w=10, a_in=3, a_out=6, b_out=4, theta=0)
        pos = np.atleast_2d(pba.positions)
        assert_allclose(pos[0], [30.5, 40.7])

    def test_theta_stored(self):
        """theta stored correctly (as Quantity in rad)."""
        import astropy.units as u
        theta = np.pi / 3
        pba = PillBoxAnnulus((50, 50), w=10, a_in=3, a_out=6, b_out=4, theta=theta)
        assert_allclose(pba.theta.to_value(u.rad), theta)

    def test_multi_position(self):
        """Multiple positions accepted."""
        pba = PillBoxAnnulus([(10, 20), (30, 40)], w=5, a_in=2, a_out=4, b_out=3, theta=0)
        assert np.atleast_2d(pba.positions).shape == (2, 2)
        assert not pba.isscalar

    # --- to_mask ---

    def test_to_mask_returns_aperturemask(self):
        """to_mask returns ApertureMask for scalar annulus."""
        from photutils.aperture import ApertureMask
        pba = PillBoxAnnulus((50, 50), w=10, a_in=3, a_out=6, b_out=4, theta=0)
        msk = pba.to_mask(method="center")
        assert isinstance(msk, ApertureMask)

    def test_to_mask_values_in_range(self):
        """Mask values are in [0, 1]."""
        pba = PillBoxAnnulus((50, 50), w=10, a_in=3, a_out=6, b_out=4, theta=0)
        for method in ("center", "subpixel", "exact"):
            msk = pba.to_mask(method=method, subpixels=5)
            arr = msk.data
            assert arr.min() >= 0.0 - 1e-12
            assert arr.max() <= 1.0 + 1e-12

    def test_to_mask_sum_approx_area(self):
        """Sum of exact mask ≈ annulus area (within ~2 pixel tolerance)."""
        pba = PillBoxAnnulus((50.0, 50.0), w=10, a_in=3, a_out=6, b_out=4, theta=0)
        msk = pba.to_mask(method="exact")
        assert_allclose(msk.data.sum(), pba.area, atol=2.0)

    def test_to_mask_inner_excluded(self):
        """Annulus mask sum < aperture mask sum (inner region excluded)."""
        pos = (50.0, 50.0)
        pba = PillBoxAnnulus(pos, w=10, a_in=3, a_out=6, b_out=4, theta=0)
        pb = PillBoxAperture(pos, w=10, a=6, b=4, theta=0)
        msk_an = pba.to_mask(method="exact")
        msk_ap = pb.to_mask(method="exact")
        assert msk_an.data.sum() < msk_ap.data.sum()

    def test_to_mask_multi_returns_list(self):
        """Multi-position → list of ApertureMask."""
        from photutils.aperture import ApertureMask
        pba = PillBoxAnnulus([(10, 20), (30, 40)], w=5, a_in=2, a_out=4, b_out=3, theta=0)
        masks = pba.to_mask(method="center")
        assert isinstance(masks, list)
        assert len(masks) == 2

    # --- _to_patch ---

    def test_to_patch_returns_patch(self):
        """_to_patch returns a matplotlib PathPatch."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.patches as mpatches
        pba = PillBoxAnnulus((50, 50), w=10, a_in=3, a_out=6, b_out=4, theta=0)
        patch = pba._to_patch()
        assert isinstance(patch, mpatches.PathPatch)

    def test_to_patch_multi_returns_list(self):
        """Multi-position _to_patch returns list."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.patches as mpatches
        pba = PillBoxAnnulus([(10, 20), (30, 40)], w=5, a_in=2, a_out=4, b_out=3, theta=0)
        patches = pba._to_patch()
        assert isinstance(patches, list)
        assert len(patches) == 2


# =============================================================================
# pill_ap_an convenience function
# =============================================================================

class TestPillApAn:
    """Tests for pill_ap_an."""

    def test_returns_correct_types(self):
        """Returns (PillBoxAperture, PillBoxAnnulus)."""
        ap, an = pill_ap_an((50, 50), fwhm=5, trail=10)
        assert isinstance(ap, PillBoxAperture)
        assert isinstance(an, PillBoxAnnulus)

    def test_dimensions_scalar_fwhm(self):
        """fwhm=5, trail=10, f_ap=(1.5,1.5), f_w=1.0:
        a_ap=7.5, b_ap=7.5, w=10."""
        ap, an = pill_ap_an((50, 50), fwhm=5, trail=10, f_ap=(1.5, 1.5), f_w=1.0)
        assert_allclose(ap.a, 7.5)
        assert_allclose(ap.b, 7.5)
        assert_allclose(ap.w, 10.0)

    def test_dimensions_asymmetric_fwhm(self):
        """fwhm=[6,4], f_ap=(2.0,1.5): a_ap=12, b_ap=6."""
        ap, _ = pill_ap_an((50, 50), fwhm=[6, 4], trail=10, f_ap=(2.0, 1.5), f_w=1.0)
        assert_allclose(ap.a, 12.0)
        assert_allclose(ap.b, 6.0)

    def test_annulus_dimensions(self):
        """fwhm=5, f_in=(4,4), f_out=(6,6): a_in=20, a_out=30."""
        _, an = pill_ap_an(
            (50, 50), fwhm=5, trail=10,
            f_ap=(1.5, 1.5), f_in=(4.0, 4.0), f_out=(6.0, 6.0), f_w=1.0
        )
        assert_allclose(an.a_in, 20.0)
        assert_allclose(an.a_out, 30.0)

    def test_f_w_scales_trail(self):
        """f_w=2.0 doubles the trail width."""
        _, an1 = pill_ap_an((50, 50), fwhm=5, trail=10, f_w=1.0)
        _, an2 = pill_ap_an((50, 50), fwhm=5, trail=10, f_w=2.0)
        assert_allclose(an2.w, 2 * an1.w)

    def test_theta_passed_through(self):
        """theta is forwarded to both aperture and annulus."""
        import astropy.units as u
        theta = np.pi / 4
        ap, an = pill_ap_an((50, 50), fwhm=5, trail=10, theta=theta)
        assert_allclose(ap.theta.to_value(u.rad), theta)
        assert_allclose(an.theta.to_value(u.rad), theta)

    def test_scalar_f_factors_broadcast(self):
        """Scalar f_ap=1.5 broadcasts to both a and b."""
        ap, _ = pill_ap_an((50, 50), fwhm=5, trail=10, f_ap=1.5, f_w=1.0)
        assert_allclose(ap.a, 7.5)
        assert_allclose(ap.b, 7.5)


# =============================================================================
# to_sky / to_pixel round-trip (photutils 2.x compatibility)
# =============================================================================

class TestPillBoxSkyPixelRoundTrip:
    """
    Tests for PillBoxAperture.to_sky / SkyPillBoxAperture.to_pixel round-trip.

    These specifically exercise the photutils 2.x API where _to_sky_params and
    _to_pixel_params no longer accept a `mode` keyword argument.
    """

    @pytest.fixture
    def wcs(self):
        return _make_wcs()

    def test_pillbox_aperture_to_sky_no_mode_error(self, wcs):
        """PillBoxAperture.to_sky(wcs) must not raise TypeError from mode= kwarg."""
        pb = PillBoxAperture((50, 50), w=10, a=5, b=3, theta=0)
        sky_ap = pb.to_sky(wcs)
        assert isinstance(sky_ap, SkyPillBoxAperture)

    def test_pillbox_annulus_to_sky_no_mode_error(self, wcs):
        """PillBoxAnnulus.to_sky(wcs) must not raise TypeError from mode= kwarg."""
        pba = PillBoxAnnulus((50, 50), w=10, a_in=3, a_out=6, b_out=4, theta=0)
        sky_an = pba.to_sky(wcs)
        assert isinstance(sky_an, SkyPillBoxAnnulus)

    def test_sky_pillbox_aperture_to_pixel_no_mode_error(self, wcs):
        """SkyPillBoxAperture.to_pixel(wcs) must not raise TypeError."""
        pb = PillBoxAperture((50, 50), w=10, a=5, b=3, theta=0)
        sky_ap = pb.to_sky(wcs)
        pix_ap = sky_ap.to_pixel(wcs)
        assert isinstance(pix_ap, PillBoxAperture)

    def test_sky_pillbox_annulus_to_pixel_no_mode_error(self, wcs):
        """SkyPillBoxAnnulus.to_pixel(wcs) must not raise TypeError."""
        pba = PillBoxAnnulus((50, 50), w=10, a_in=3, a_out=6, b_out=4, theta=0)
        sky_an = pba.to_sky(wcs)
        pix_an = sky_an.to_pixel(wcs)
        assert isinstance(pix_an, PillBoxAnnulus)

    def test_aperture_roundtrip_position(self, wcs):
        """Pixel → sky → pixel position round-trip is self-consistent."""
        pos = (50.0, 50.0)
        pb = PillBoxAperture(pos, w=10, a=5, b=3, theta=0)
        sky_ap = pb.to_sky(wcs)
        pix_ap = sky_ap.to_pixel(wcs)
        assert_allclose(np.atleast_2d(pix_ap.positions), np.atleast_2d(pb.positions), atol=1e-6)

    def test_annulus_roundtrip_position(self, wcs):
        """Pixel → sky → pixel position round-trip is self-consistent."""
        pos = (50.0, 50.0)
        pba = PillBoxAnnulus(pos, w=10, a_in=3, a_out=6, b_out=4, theta=0)
        sky_an = pba.to_sky(wcs)
        pix_an = sky_an.to_pixel(wcs)
        assert_allclose(np.atleast_2d(pix_an.positions), np.atleast_2d(pba.positions), atol=1e-6)


# =============================================================================
# cutout_from_ap
# =============================================================================

class TestCutoutFromAp:
    """Tests for cutout_from_ap."""

    def test_returns_cutout2d(self):
        """Returns a Cutout2D object for scalar aperture."""
        ccd = _uniform_ccd()
        ap = CircularAperture((50, 50), r=10)
        cut = cutout_from_ap(ap, ccd)
        assert isinstance(cut, Cutout2D)

    def test_returns_list_for_multi(self):
        """Returns list of Cutout2D for multi-position aperture."""
        ccd = _uniform_ccd()
        ap = CircularAperture([(30, 30), (60, 60)], r=5)
        cuts = cutout_from_ap(ap, ccd)
        assert isinstance(cuts, list)
        assert len(cuts) == 2
        for c in cuts:
            assert isinstance(c, Cutout2D)

    def test_bbox_method_shape(self):
        """method='bbox': cutout shape matches aperture bounding box.
        CircularAperture r=10 at (50,50): bbox is 21×21."""
        ccd = _uniform_ccd()
        ap = CircularAperture((50, 50), r=10)
        cut = cutout_from_ap(ap, ccd, method="bbox")
        # photutils bbox for r=10: floor(50-10+0.5)=40, ceil(50+10+0.5)=61 → 21 pixels
        assert cut.data.shape == (21, 21)

    def test_bbox_method_values_uniform(self):
        """method='bbox' on uniform array: all cutout values equal fill value."""
        ccd = _uniform_ccd(value=7.0)
        ap = CircularAperture((50, 50), r=5)
        cut = cutout_from_ap(ap, ccd, method="bbox")
        assert_allclose(cut.data, 7.0)

    def test_center_method_returns_cutout(self):
        """method='center' returns Cutout2D with data attribute."""
        ccd = _uniform_ccd()
        ap = CircularAperture((50, 50), r=5)
        cut = cutout_from_ap(ap, ccd, method="center")
        assert isinstance(cut, Cutout2D)
        assert cut.data is not None

    def test_elliptical_aperture(self):
        """Works with EllipticalAperture."""
        ccd = _uniform_ccd()
        ap = EllipticalAperture((50, 50), a=8, b=4, theta=0)
        cut = cutout_from_ap(ap, ccd, method="bbox")
        assert isinstance(cut, Cutout2D)

    def test_pillbox_aperture(self):
        """Works with PillBoxAperture."""
        ccd = _uniform_ccd()
        ap = PillBoxAperture((50, 50), w=10, a=5, b=3, theta=0)
        cut = cutout_from_ap(ap, ccd, method="bbox")
        assert isinstance(cut, Cutout2D)


# =============================================================================
# ap_to_cutout_position
# =============================================================================

class TestApToCutoutPosition:
    """Tests for ap_to_cutout_position."""

    def test_position_updated(self):
        """Position is correctly remapped into cutout coordinates."""
        ccd = _uniform_ccd()
        ap = CircularAperture((50, 50), r=5)
        cutout = Cutout2D(ccd.data, position=(50, 50), size=21)
        new_ap = ap_to_cutout_position(ap, cutout)
        # center of a 21×21 cutout centered at (50,50) is (10,10)
        assert_allclose(np.atleast_2d(new_ap.positions)[0], [10, 10], atol=1e-10)

    def test_original_unchanged(self):
        """Original aperture is not mutated (deep copy)."""
        ccd = _uniform_ccd()
        ap = CircularAperture((50, 50), r=5)
        cutout = Cutout2D(ccd.data, position=(50, 50), size=21)
        _ = ap_to_cutout_position(ap, cutout)
        assert_allclose(np.atleast_2d(ap.positions)[0], [50, 50])

    def test_multi_position(self):
        """Multi-position aperture: each position remapped correctly."""
        ccd = _uniform_ccd()
        ap = CircularAperture([(30, 30), (70, 70)], r=5)
        # cutout centered at (50,50), size=81 → positions shift by -10
        cutout = Cutout2D(ccd.data, position=(50, 50), size=81)
        new_ap = ap_to_cutout_position(ap, cutout)
        pos = np.atleast_2d(new_ap.positions)
        # (30,30) in original → (30-10, 30-10) = (20,20) in cutout
        assert_allclose(pos[0], [20, 20], atol=1e-10)
        assert_allclose(pos[1], [60, 60], atol=1e-10)

    def test_radius_unchanged(self):
        """Aperture radius is not changed by position update."""
        ccd = _uniform_ccd()
        ap = CircularAperture((50, 50), r=7)
        cutout = Cutout2D(ccd.data, position=(50, 50), size=21)
        new_ap = ap_to_cutout_position(ap, cutout)
        assert_allclose(new_ap.r, 7.0)


# =============================================================================
# pa2xytheta
# =============================================================================

class TestPa2XyTheta:
    """Tests for pa2xytheta.

    Analytical expectations for the test WCS (_make_wcs with flip_ra=True):
    - cdelt = [-0.5/3600, +0.5/3600]: RA increases left, Dec increases up.
    - pa_x = PA of the +x axis = 270° (West, since RA increases left → +x points West).
    - pa_y = PA of the +y axis = 0° (North).
    - CCW orientation: theta = pa_x - pa = 270 - pa.
    """

    @pytest.fixture
    def wcs_standard(self):
        """Standard WCS: RA left, Dec up.  pa_x=270°, pa_y=0°."""
        return _make_wcs(flip_ra=True)

    def test_pa_north_gives_270deg(self, wcs_standard):
        """PA=0 (North) → theta=270°.  pa_x=270, CCW: theta = 270 - 0 = 270."""
        theta = pa2xytheta(0.0, wcs_standard, location="crpix")
        assert_allclose(theta, 270.0, atol=0.1)

    def test_pa_east_gives_180deg(self, wcs_standard):
        """PA=90 (East) → theta=180°.  CCW: theta = 270 - 90 = 180."""
        theta = pa2xytheta(90.0, wcs_standard, location="crpix")
        assert_allclose(theta, 180.0, atol=0.1)

    def test_pa_south_gives_90deg(self, wcs_standard):
        """PA=180 (South) → theta=90°.  CCW: theta = 270 - 180 = 90."""
        theta = pa2xytheta(180.0, wcs_standard, location="crpix")
        assert_allclose(theta, 90.0, atol=0.1)

    def test_pa_west_gives_0deg(self, wcs_standard):
        """PA=270 (West) → theta=0°.  CCW: theta = 270 - 270 = 0."""
        theta = pa2xytheta(270.0, wcs_standard, location="crpix")
        assert_allclose(abs(theta), 0.0, atol=0.1)

    def test_location_crpix(self, wcs_standard):
        """location='crpix' does not raise and returns a float."""
        theta = pa2xytheta(0.0, wcs_standard, location="crpix")
        assert np.isfinite(theta)

    def test_location_center(self, wcs_standard):
        """location='center' does not raise and returns a float."""
        theta = pa2xytheta(0.0, wcs_standard, location="center")
        assert np.isfinite(theta)

    def test_location_tuple(self, wcs_standard):
        """location=(x,y) tuple does not raise and returns a float."""
        theta = pa2xytheta(0.0, wcs_standard, location=(40, 40))
        assert np.isfinite(theta)

    def test_crpix_vs_center_close(self, wcs_standard):
        """crpix and center locations give nearly identical theta for simple WCS."""
        t_crpix = pa2xytheta(45.0, wcs_standard, location="crpix")
        t_center = pa2xytheta(45.0, wcs_standard, location="center")
        assert_allclose(t_crpix, t_center, atol=0.5)


# =============================================================================
# Photometric consistency: aperture sum on uniform image
# =============================================================================

class TestAperturePhotometry:
    """Verify aperture sums on analytically known images."""

    def test_pillbox_aperture_sum_uniform(self):
        """Sum of PillBoxAperture mask * uniform image ≈ area * value.
        w=10, a=5, b=3, value=2.0 → sum ≈ (60+15*pi)*2."""
        value = 2.0
        im = np.full((200, 200), value)
        pb = PillBoxAperture((100, 100), w=10, a=5, b=3, theta=0)
        msk = pb.to_mask(method="exact")
        apsum = np.sum(msk.multiply(im))
        assert_allclose(apsum, pb.area * value, atol=1.0)

    def test_pillbox_annulus_sum_uniform(self):
        """Sum of PillBoxAnnulus mask * uniform image ≈ area * value."""
        value = 3.0
        im = np.full((200, 200), value)
        pba = PillBoxAnnulus((100, 100), w=10, a_in=3, a_out=6, b_out=4, theta=0)
        msk = pba.to_mask(method="exact")
        apsum = np.sum(msk.multiply(im))
        assert_allclose(apsum, pba.area * value, atol=2.0)

    def test_circ_aperture_sum_uniform(self):
        """CircularAperture sum on uniform image = pi*r^2 * value.
        r=10, value=5 → 500*pi."""
        value = 5.0
        im = np.full((200, 200), value)
        ap = CircularAperture((100, 100), r=10)
        msk = ap.to_mask(method="exact")
        apsum = np.sum(msk.multiply(im))
        assert_allclose(apsum, np.pi * 100 * value, atol=1.0)

    def test_pillbox_annulus_less_than_aperture(self):
        """Annulus sum < outer aperture sum (inner region excluded)."""
        im = np.full((200, 200), 1.0)
        pos = (100, 100)
        pb = PillBoxAperture(pos, w=10, a=6, b=4, theta=0)
        pba = PillBoxAnnulus(pos, w=10, a_in=3, a_out=6, b_out=4, theta=0)
        sum_ap = np.sum(pb.to_mask(method="exact").multiply(im))
        sum_an = np.sum(pba.to_mask(method="exact").multiply(im))
        assert sum_an < sum_ap
