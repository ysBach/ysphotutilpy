"""Low-level aperture utilities bypassing photutils high-level overhead.

These functions replicate the bounding-box and overlap-grid logic used
internally by `photutils`, giving direct access to aperture/annulus masks
without constructing `CircularAperture` / `CircularAnnulus` objects or
going through `aperture_photometry`.
"""

import math

import numpy as np
from photutils.geometry import circular_overlap_grid

__all__ = [
    "fast_circ_apmask",
    "fast_circ_anmask",
    "fast_circ_apanmask",
]


def _bbox(x, y, r):
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
    ixmin, ixmax, iymin, iymax, xmin, xmax, ymin, ymax, nx, ny = _bbox(x, y, r)
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
    ixmin, ixmax, iymin, iymax, xmin, xmax, ymin, ymax, nx, ny = _bbox(x, y, r_out)
    sl = (slice(iymin, iymax), slice(ixmin, ixmax))
    mask = circular_overlap_grid(xmin, xmax, ymin, ymax, nx, ny, r_out, use_exact, subpixels)
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
    ixmin, ixmax, iymin, iymax, xmin, xmax, ymin, ymax, nx, ny = _bbox(x, y, r_out)
    sl = (slice(iymin, iymax), slice(ixmin, ixmax))
    mask_ap = circular_overlap_grid(xmin, xmax, ymin, ymax, nx, ny, r, ap_use_exact, subpixels)
    mask_an = circular_overlap_grid(xmin, xmax, ymin, ymax, nx, ny, r_out, an_use_exact, subpixels)
    mask_in = circular_overlap_grid(xmin, xmax, ymin, ymax, nx, ny, r_in, an_use_exact, subpixels)
    mask_an -= mask_in
    return mask_ap, mask_an, sl
