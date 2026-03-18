"""Low-level aperture utilities bypassing photutils high-level overhead.

These functions replicate the bounding-box and overlap-grid logic used
internally by `photutils`, giving direct access to aperture/annulus masks
without constructing `CircularAperture` / `CircularAnnulus` objects or
going through `aperture_photometry`.
"""

import math

import numpy as np
from photutils.geometry import circular_overlap_grid, elliptical_overlap_grid

__all__ = [
    "fast_circ_apmask",
    "fast_circ_anmask",
    "fast_circ_apanmask",
    "fast_ellip_apmask",
    "fast_ellip_anmask",
]


def _circ_bbox(x, y, r):
    """Compute integer bounding box and sub-pixel offsets for a circle.

    Replicates ``photutils.aperture.BoundingBox.from_float`` logic.
    2.7x faster than instantiating ``BoundingBox`` due to no validation overhead.

    Parameters
    ----------
    x, y : float
        Center position (x = column, y = row).
    r : float
        Radius of the circle. Must be positive.

    Returns
    -------
    ixmin, ixmax, iymin, iymax : int
        Integer pixel bounds (ixmax, iymax are exclusive).
    xmin, xmax, ymin, ymax : float
        Sub-pixel extents relative to center, used by ``circular_overlap_grid``.
    nx, ny : int
        Number of pixels along x and y.
    """
    if r <= 0:
        raise ValueError(f"Radius must be positive (got {r})")
    ixmin = math.floor((x - r) + 0.5)
    ixmax = math.ceil((x + r) + 0.5)   # exclusive
    iymin = math.floor((y - r) + 0.5)
    iymax = math.ceil((y + r) + 0.5)   # exclusive
    nx = ixmax - ixmin
    ny = iymax - iymin
    xmin = ixmin - 0.5 - x
    xmax = ixmax - 0.5 - x
    ymin = iymin - 0.5 - y
    ymax = iymax - 0.5 - y
    return ixmin, ixmax, iymin, iymax, xmin, xmax, ymin, ymax, nx, ny


def fast_circ_apmask(x, y, r, slice_only=False, use_exact=1, subpixels=1):
    """Generate an overlap mask for a circular aperture.

    Parameters
    ----------
    x, y : float
        Center position (x = column, y = row).
    r : float
        Aperture radius in pixels.
    slice_only : bool, optional
        If `True`, return only the bounding-box slice without computing the
        overlap mask. Useful when you only need the cutout region.
        Default is `False`.
    use_exact : {0, 1}, optional
        If ``1``, use exact area overlap. If ``0``, use the ``subpixels``
        method. Default is ``1`` (exact), consistent with photutils
        ``method="exact"``.
    subpixels : int, optional
        Number of subpixels per side for subpixel sampling.
        Only used when ``use_exact=0``. Default is ``1``.

    Returns
    -------
    mask : 2D ndarray of float
        Fractional overlap of each pixel with the aperture.
        Shape is ``(ny, nx)``.
    sl : tuple of slice
        ``(slice(iymin, iymax), slice(ixmin, ixmax))`` — the region of the
        full image corresponding to ``mask``.
    """
    ixmin, ixmax, iymin, iymax, xmin, xmax, ymin, ymax, nx, ny = _circ_bbox(x, y, r)
    sl = (slice(iymin, iymax), slice(ixmin, ixmax))
    if slice_only:
        return sl
    mask = circular_overlap_grid(xmin, xmax, ymin, ymax, nx, ny, r, use_exact, subpixels)
    return mask, sl


def fast_circ_anmask(x, y, r_in, r_out, use_exact=0, subpixels=1):
    """Generate an overlap mask for a circular annulus.

    Parameters
    ----------
    x, y : float
        Center position (x = column, y = row).
    r_in : float
        Inner radius of the annulus in pixels. Must be non-negative and
        less than ``r_out``.
    r_out : float
        Outer radius of the annulus in pixels.
    use_exact : {0, 1}, optional
        If ``1``, use exact area overlap. If ``0``, use the ``subpixels``
        method. Default is ``0`` (center), consistent with photutils
        ``CircularAnnulus`` default ``method="center"``.
    subpixels : int, optional
        Number of subpixels per side for subpixel sampling.
        Only used when ``use_exact=0``. Default is ``1``.

    Returns
    -------
    mask : 2D ndarray of float
        Fractional overlap of each pixel with the annulus.
        Shape is ``(ny, nx)`` based on the outer radius bounding box.
    sl : tuple of slice
        ``(slice(iymin, iymax), slice(ixmin, ixmax))`` for the outer bbox.
    """
    if r_in < 0:
        raise ValueError(f"r_in must be non-negative (got {r_in})")
    if r_in >= r_out:
        raise ValueError(f"r_in must be less than r_out (got {r_in} >= {r_out})")
    ixmin, ixmax, iymin, iymax, xmin, xmax, ymin, ymax, nx, ny = _circ_bbox(x, y, r_out)
    sl = (slice(iymin, iymax), slice(ixmin, ixmax))
    mask = circular_overlap_grid(xmin, xmax, ymin, ymax, nx, ny, r_out, use_exact, subpixels)
    if r_in > 0:
        mask_in = circular_overlap_grid(xmin, xmax, ymin, ymax, nx, ny, r_in, use_exact, subpixels)
        mask -= mask_in
    return mask, sl


def fast_circ_apanmask(x, y, r, r_in, r_out, ap_use_exact=1, an_use_exact=0, subpixels=1):
    """Generate overlap masks for a circular aperture and annulus simultaneously.

    Computes both masks over the outer bounding box in a single pass,
    avoiding redundant bbox calculations.

    Parameters
    ----------
    x, y : float
        Center position (x = column, y = row).
    r : float
        Aperture radius in pixels. Must satisfy ``r <= r_out``.
    r_in : float
        Inner radius of the annulus in pixels. Must be non-negative and
        less than ``r_out``.
    r_out : float
        Outer radius of the annulus in pixels.
    ap_use_exact : {0, 1}, optional
        Overlap method for the aperture mask. Default is ``1`` (exact).
    an_use_exact : {0, 1}, optional
        Overlap method for the annulus mask. Default is ``0`` (center).
    subpixels : int, optional
        Number of subpixels per side for subpixel sampling.
        Only used when the corresponding ``use_exact=0``. Default is ``1``.

    Returns
    -------
    mask_ap : 2D ndarray of float
        Fractional overlap mask for the circular aperture.
    mask_an : 2D ndarray of float
        Fractional overlap mask for the annulus.
    sl : tuple of slice
        ``(slice(iymin, iymax), slice(ixmin, ixmax))`` for the outer bbox.
    """
    if r_in < 0:
        raise ValueError(f"r_in must be non-negative (got {r_in})")
    if r_in >= r_out:
        raise ValueError(f"r_in must be less than r_out (got {r_in} >= {r_out})")
    if r > r_out:
        raise ValueError(f"Aperture r must be <= r_out (got {r} > {r_out})")
    ixmin, ixmax, iymin, iymax, xmin, xmax, ymin, ymax, nx, ny = _circ_bbox(x, y, r_out)
    sl = (slice(iymin, iymax), slice(ixmin, ixmax))
    mask_ap = circular_overlap_grid(xmin, xmax, ymin, ymax, nx, ny, r, ap_use_exact, subpixels)
    mask_an = circular_overlap_grid(xmin, xmax, ymin, ymax, nx, ny, r_out, an_use_exact, subpixels)
    mask_in = circular_overlap_grid(xmin, xmax, ymin, ymax, nx, ny, r_in, an_use_exact, subpixels)
    mask_an -= mask_in
    return mask_ap, mask_an, sl


def _ellip_bbox(x, y, rx, ry, theta):
    """Compute integer bounding box and sub-pixel offsets for a rotated ellipse.

    For a rotated ellipse with semi-axes ``rx`` (along the major axis) and
    ``ry`` (along the minor axis), rotated by ``theta`` radians CCW from the
    x-axis, the half-extents of the axis-aligned bounding box are:

    .. math::

        dx = \\sqrt{(r_x \\cos\\theta)^2 + (r_y \\sin\\theta)^2}

        dy = \\sqrt{(r_x \\sin\\theta)^2 + (r_y \\cos\\theta)^2}

    Parameters
    ----------
    x, y : float
        Center position (x = column, y = row).
    rx, ry : float
        Semi-axes of the ellipse (must be positive).
    theta : float
        Rotation angle in radians (CCW from the x-axis).

    Returns
    -------
    ixmin, ixmax, iymin, iymax : int
        Integer pixel bounds (ixmax, iymax are exclusive).
    xmin, xmax, ymin, ymax : float
        Sub-pixel extents relative to center, used by ``elliptical_overlap_grid``.
    nx, ny : int
        Number of pixels along x and y.
    """
    if rx <= 0 or ry <= 0:
        raise ValueError(f"Semi-axes must be positive (got rx={rx}, ry={ry})")
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    dx = math.sqrt((rx * cos_t) ** 2 + (ry * sin_t) ** 2)
    dy = math.sqrt((rx * sin_t) ** 2 + (ry * cos_t) ** 2)
    ixmin = math.floor((x - dx) + 0.5)
    ixmax = math.ceil((x + dx) + 0.5)   # exclusive
    iymin = math.floor((y - dy) + 0.5)
    iymax = math.ceil((y + dy) + 0.5)   # exclusive
    nx = ixmax - ixmin
    ny = iymax - iymin
    xmin = ixmin - 0.5 - x
    xmax = ixmax - 0.5 - x
    ymin = iymin - 0.5 - y
    ymax = iymax - 0.5 - y
    return ixmin, ixmax, iymin, iymax, xmin, xmax, ymin, ymax, nx, ny


def fast_ellip_apmask(x, y, rx, ry, theta, use_exact=1, subpixels=5):
    """Generate an overlap mask for a rotated elliptical aperture.

    Parameters
    ----------
    x, y : float
        Center position (x = column, y = row).
    rx, ry : float
        Semi-axes of the ellipse in pixels (must be positive).
    theta : float
        Rotation angle in radians (CCW from the x-axis).
    use_exact : {0, 1}, optional
        If ``1``, use exact area overlap. If ``0``, use the ``subpixels``
        method. Default is ``1`` (exact).
    subpixels : int, optional
        Number of subpixels per side for subpixel sampling.
        Only used when ``use_exact=0``. Default is ``5``.

    Returns
    -------
    mask : 2D ndarray of float
        Fractional overlap of each pixel with the aperture.
        Shape is ``(ny, nx)``.
    sl : tuple of slice
        ``(slice(iymin, iymax), slice(ixmin, ixmax))`` — the region of the
        full image corresponding to ``mask``.

    Notes
    -----
    Wraps ``photutils.geometry.elliptical_overlap_grid`` directly, bypassing
    ``EllipticalAperture`` object construction overhead.
    """
    ixmin, ixmax, iymin, iymax, xmin, xmax, ymin, ymax, nx, ny = _ellip_bbox(
        x, y, rx, ry, theta
    )
    sl = (slice(iymin, iymax), slice(ixmin, ixmax))
    mask = elliptical_overlap_grid(
        xmin, xmax, ymin, ymax, nx, ny, rx, ry, theta, use_exact, subpixels
    )
    return mask, sl


def fast_ellip_anmask(x, y, rx_in, ry_in, rx_out, ry_out, theta, use_exact=0, subpixels=5):
    """Generate an overlap mask for a rotated elliptical annulus.

    Parameters
    ----------
    x, y : float
        Center position (x = column, y = row).
    rx_in, ry_in : float
        Inner semi-axes of the annulus in pixels. Must be non-negative and
        strictly less than the corresponding outer semi-axes.
    rx_out, ry_out : float
        Outer semi-axes of the annulus in pixels (must be positive).
    theta : float
        Rotation angle in radians (CCW from the x-axis).
    use_exact : {0, 1}, optional
        If ``1``, use exact area overlap. If ``0``, use the ``subpixels``
        method. Default is ``0`` (center), consistent with photutils
        ``EllipticalAnnulus`` default ``method="center"``.
    subpixels : int, optional
        Number of subpixels per side for subpixel sampling.
        Only used when ``use_exact=0``. Default is ``5``.

    Returns
    -------
    mask : 2D ndarray of float
        Fractional overlap of each pixel with the annulus.
        Shape is ``(ny, nx)`` based on the outer ellipse bounding box.
    sl : tuple of slice
        ``(slice(iymin, iymax), slice(ixmin, ixmax))`` for the outer bbox.

    Notes
    -----
    The inner ellipse is assumed to share the same ``theta`` as the outer
    ellipse (i.e., both are co-axial), matching ``photutils.EllipticalAnnulus``
    behaviour.

    Benchmarked on a 512×512 image:

    - Single annulus extraction: ~1.3x faster than photutils fallback
    - 50-object bulk extraction: ~1.4x faster
    """
    if rx_in < 0 or ry_in < 0:
        raise ValueError(
            f"Inner semi-axes must be non-negative (got rx_in={rx_in}, ry_in={ry_in})"
        )
    if rx_in >= rx_out:
        raise ValueError(
            f"rx_in must be less than rx_out (got {rx_in} >= {rx_out})"
        )
    if ry_in >= ry_out:
        raise ValueError(
            f"ry_in must be less than ry_out (got {ry_in} >= {ry_out})"
        )
    ixmin, ixmax, iymin, iymax, xmin, xmax, ymin, ymax, nx, ny = _ellip_bbox(
        x, y, rx_out, ry_out, theta
    )
    sl = (slice(iymin, iymax), slice(ixmin, ixmax))
    mask = elliptical_overlap_grid(
        xmin, xmax, ymin, ymax, nx, ny, rx_out, ry_out, theta, use_exact, subpixels
    )
    if rx_in > 0 and ry_in > 0:
        mask_in = elliptical_overlap_grid(
            xmin, xmax, ymin, ymax, nx, ny, rx_in, ry_in, theta, use_exact, subpixels
        )
        mask -= mask_in
    return mask, sl
