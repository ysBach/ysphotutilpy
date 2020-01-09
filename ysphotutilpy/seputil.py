'''
A collection of convenience functions used with the package sep.

Why use sep, not photutils?
===========================
Many times the purpose of SExtractor is to mearly **find** objects, not
detailed photometry of them. Thus,  the calculational speed is an
important factor.

sep is about 100 times faster in background estimation than photutils
from many tests (see sep/bench.py of the sep repo):

```
$ python bench.py
test image shape: (1024, 1024)
test image dtype: float32
measure background:  12.34 ms
subtract background:   3.38 ms
background array:   6.94 ms
rms array:   5.21 ms
extract:  88.87 ms  [1167 objects]

sep version:       1.10.0
photutils version: 0.7.1

| test                    | sep             | photutils       | ratio  |
|-------------------------|-----------------|-----------------|--------|
| 1024^2 image background |        11.68 ms |       851.76 ms |  72.94 |
| circles  r= 5  subpix=5 |    3.40 us/aper |   45.30 us/aper |  13.31 |
| circles  r= 5  exact    |    3.89 us/aper |   45.76 us/aper |  11.76 |
| ellipses r= 5  subpix=5 |    3.89 us/aper |   75.29 us/aper |  19.37 |
| ellipses r= 5  exact    |   10.88 us/aper |   63.71 us/aper |   5.85 |
```

Maybe more benchmark for extraction is needed (photutils's
detect_sources, source_properties), but it's already tempting to use sep
over photutils given the purpose is only to extract the objects, not
photometry.

Pixel convention
================
Note that sep also uses same pixel notation as photutils: pixel 0 covers
-0.5 to +0.5. Simple test code:
>>> test = np.zeros((15, 15))
>>> test[7, 8] = 1
>>> test[8, 9] = 2
>>> obj = pd.DataFrame(sep.extract(test, 0))
>>> for c in obj.columns:
>>>     print(f"{c}: {obj[c].values}")
>>> # x: [8.66666667]
>>> # y: [7.66666667]
'''
from warnings import warn

import numpy as np
import pandas as pd

from .util import bezel_mask

try:
    import sep
except ImportError:
    warn("Package sep is not installed. Some functions will not work.")


__all__ = ['sep_back', 'sep_extract', 'sep_flux_auto']

sep_default_kernel = np.array([[1.0, 2.0, 1.0],
                               [2.0, 4.0, 2.0],
                               [1.0, 2.0, 1.0]], dtype=np.float32)


def sep_back(data, mask=None, maskthresh=0.0, filter_threshold=0.0,
             box_size=(64, 64), filter_size=(3, 3)):
    '''
    Note
    ----
    This includes ``sep``'s ``Background``. Equivalent processes in
    photutils may include ``Background2D``.

    Parameters
    ----------
    data : CCDData or array-like
        The 2D array from which to estimate the background.

    mask : 2-d `~numpy.ndarray`, optional
        Mask array.

    maskthresh : float, optional
        Only in sep. The effective mask will be ``m = (mask <
        maskthresh)``.
        sep:
        Mask threshold. This is the inclusive upper limit on the mask
        value in order for the corresponding pixel to be unmasked. For
        boolean arrays, False and True are interpreted as 0 and 1,
        respectively. Thus, given a threshold of zero, True corresponds
        to masked and False corresponds to unmasked.
        photutils:
        In photutils, sigma clipping is used (need check).

    filter_threshold : int, optional
        Name in photutils; ``fthresh`` in the oritinal sep. Default is
        0.0.
        sep:
        Filter threshold. Default is 0.0.
        photutils:
        The threshold value for used for selective median filtering of
        the low-resolution 2D background map.  The median filter will be
        applied to only the background meshes with values larger than
        ``filter_threshold``.  Set to `None` to filter all meshes
        (default).

    box_size : int or array_like (int)
        Name in photutils; ``bh, bw`` in sep. Default is ``(64, 64)``.
        sep:
        Size of background boxes in pixels. Default is 64.
        photutils:
        The box size along each axis. If ``box_size`` is a scalar then a
        square box of size ``box_size`` will be used.  If ``box_size``
        has two elements, they should be in ``(ny, nx)`` order.  For
        best results, the box shape should be chosen such that the
        ``data`` are covered by an integer number of boxes in both
        dimensions. When this is not the case, see the ``edge_method``
        keyword for more options.

    filter_size : int or array_like (int), optional
        Name in photutils; ``bh, bw`` in sep. Default is ``(64, 64)``.
        sep:
        Filter width and height in boxes. Default is 3.
        photutils:
        The window size of the 2D median filter to apply to the
        low-resolution background map. If ``filter_size`` is a scalar
        then a square box of size ``filter_size`` will be used.  If
        ``filter_size`` has two elements, they should be in ``(ny, nx)``
        order.  A filter size of ``1`` (or ``(1, 1)``) means no
        filtering.

    Returns
    -------
    bkg : sep.Background
        Use ``bkg.back()`` and ``bkg.rms()`` to get the background and
        rms error. All other methods/attributes include
        ``bkg.subfrom()``, ``bkg.globalback``, and ``bkg.globalrms``.
    '''
    # asarrya is needed because sep gives Error for C-contiguous array.
    data = np.asarray(data)
    if mask is not None:
        mask = np.asarray(mask).astype(bool)

    box_size = np.atleast_1d(box_size)
    if len(box_size) == 1:
        box_size = np.repeat(box_size, 2)
    box_size = (min(box_size[0], data.shape[0]),
                min(box_size[1], data.shape[1]))

    filter_size = np.atleast_1d(filter_size)
    if len(filter_size) == 1:
        filter_size = np.repeat(filter_size, 2)

    bkg = sep.Background(data, mask=mask, bw=box_size[1], bh=box_size[0],
                         fw=filter_size[1], fh=filter_size[0],
                         maskthresh=maskthresh, fthresh=filter_threshold)

    return bkg


def sep_extract(data, thresh, bkg, mask=None, maskthresh=0.0,
                err=None, var=None,
                bezel_x=[0, 0], bezel_y=[0, 0],
                gain=None, minarea=5, filter_kernel=sep_default_kernel,
                filter_type='matched', deblend_nthresh=32,
                deblend_cont=0.005, clean=True, clean_param=1.0):
    """
    Note
    ----
    This includes ``sep``'s `extract``. Equivalent processes in
    photutils may include ``detect_sources`` and ``source_properties``.
    Maybe we can use ``extract(data=data, err=err, thresh=3)``
    for a snr > 3 extraction.

    Parameters
    ----------
    data : CCDData or array-like
        The 2D array from which to estimate the background.

    thresh : float, optional.
        Only in sep;
        sep:
        Threshold pixel value for detection. If an ``err`` or ``var``
        array is **not** given, this is interpreted as an absolute
        threshold. If ``err`` or ``var`` is given, this is interpreted
        as a relative threshold: the absolute threshold at pixel (j, i)
        will be ``thresh * err[j, i]`` or ``thresh * sqrt(var[j, i])``.
        Note: If you want to give pixel-wise threshold, make the ``err``
        with such threshold values and set ``thresh = 1``.

    bkg : sep.Background object
        The ``sep.Background`` object used to extract sky and sky rms.

    mask : `~numpy.ndarray`, optional
        Mask array. ``True`` values, or numeric values greater than
        ``maskthresh``, are considered masked. Masking a pixel is equivalent
        to setting data to zero and noise (if present) to infinity.

    maskthresh : float, optional
        Threshold for a pixel to be masked. Default is ``0.0``.

    err, var : float or `~numpy.ndarray`, optional
        Error *or* variance (specify at most one). This can be used to
        specify a pixel-by-pixel detection threshold; see ``thresh``.

    bezel_x, bezel_y : int, float, 2-array-like, optional
        The bezel (border width) for x and y axes. If array-like, it
        should be ``(lower, upper)``. Mathematically put, only objects
        with center ``(bezel_x[0] + 0.5 < center_x) & (center_x < nx -
        bezel_x[1] - 0.5)`` (similar for y) will be selected. If you
        want to keep some stars outside the edges, put negative values
        (e.g., ``-5``).

    gain : float, optional
        Conversion factor between data array units and poisson counts. This
        does not affect detection; it is used only in calculating Poisson
        noise contribution to uncertainty parameters such as ``errx2``. If
        not given, no Poisson noise will be added.

    minarea : int, optional
        Minimum number of pixels required for an object. Default is 5.
    filter_kernel : `~numpy.ndarray` or None, optional
        Filter kernel used for on-the-fly filtering (used to
        enhance detection). Default is a 3x3 array:
        [[1,2,1], [2,4,2], [1,2,1]]. Set to ``None`` to skip
        convolution.

    filter_type : {'matched', 'conv'}, optional
        Filter treatment. This affects filtering behavior when a noise
        array is supplied. ``'matched'`` (default) accounts for
        pixel-to-pixel noise in the filter kernel. ``'conv'`` is
        simple convolution of the data array, ignoring pixel-to-pixel
        noise across the kernel.  ``'matched'`` should yield better
        detection of faint sources in areas of rapidly varying noise
        (such as found in coadded images made from semi-overlapping
        exposures).  The two options are equivalent when noise is
        constant.

    deblend_nthresh : int, optional
        Number of thresholds used for object deblending. Default is 32.

    deblend_cont : float, optional
        Minimum contrast ratio used for object deblending. Default is 0.005.
        To entirely disable deblending, set to 1.0.

    clean : bool, optional
        Perform cleaning? Default is True.

    clean_param : float, optional
        Cleaning parameter (see SExtractor manual). Default is 1.0.

    Returns
    -------

    Example
    -------
    To test:
    >>> def bkg(data, mask, th):
    >>>     return sep.Background(data, mask=mask, maskthresh=th).back()
    >>> np.random.seed(1234)
    >>> data = np.random.normal(scale=100, size=(1000, 1000))
    >>> bs = []
    >>> bs.append(bkg(data, mask=None, th=0))
    >>> bs.append(bkg(data, mask=data, th=-1.e+5))
    >>> bs.append(bkg(data, mask=data, th=0))
    >>> bs.append(bkg(data, mask=data, th=+1.e+5))
    >>> bs.append(bkg(data, mask=data.astype(bool), th=+1.e+5))
    >>> np.testing.assert_array_almost_equal(bs[1], bs[2])
    >>> fig, axs = plt.subplots(2, 4, figsize=(8, 5))
    >>> for idx, b in enumerate(bs):
    >>>     ax = axs[idx%2, idx//2]
    >>>     ax.imshow(b, origin='lower')
    >>>     ax.set(title=idx)
    >>> plt.tight_layout()
    >>> plt.show()
    """
    # asarrya is needed because sep gives Error for C-contiguous array.
    data = np.asarray(data)

    if mask is not None:
        mask = np.asarray(mask).astype(bool)

    if err is None:
        err = np.zeros_like(data)
    else:
        err = np.asarray(err)

    sky = bkg.back()
    err = np.sqrt(err**2 + bkg.rms()**2)

    obj, segm = sep.extract(data - sky, thresh=thresh, err=err,
                            mask=mask, maskthresh=maskthresh,
                            minarea=minarea, filter_kernel=filter_kernel,
                            filter_type=filter_type,
                            deblend_nthresh=deblend_nthresh,
                            deblend_cont=deblend_cont, clean=clean,
                            clean_param=clean_param, segmentation_map=True)
    obj = pd.DataFrame(obj)

    ny, nx = data.shape
    mask = bezel_mask(obj['x'], obj['y'], nx, ny,
                      bezel_x=bezel_x, bezel_y=bezel_y)
    obj = obj[~mask]
    obj = obj.reset_index()
    obj = obj.rename(columns={'index': 'segm_label'})
    obj['segm_label'] += 1
    return obj, segm


def sep_flux_auto(data, sepext, err=None, phot_autoparams=(2.5, 3.5)):
    # Calculate FLUX_AUTO
    # https://sep.readthedocs.io/en/v1.0.x/apertures.html#equivalent-of-flux-auto-e-g-mag-auto-in-source-extractor

    sepx, sepy = sepext['x'], sepext['y']
    sepa, sepb = sepext['a'], sepext['b']
    septh = sepext['theta']

    r_kron, nrej_k = sep.kron_radius(data, sepx, sepy, sepa, sepb, septh, 6.0)
    fl, dfl, nrej = sep.sum_ellipse(data, sepx, sepy, sepa, sepb, septh,
                                    r=phot_autoparams[0]*r_kron, err=err,
                                    subpix=1)
    nrej |= nrej_k  # combine flags into 'flag'

    r_min = phot_autoparams[1]  # R_min = 3.5
    use_circle = r_kron * np.sqrt(sepa * sepb) < r_min
    cfl, cdfl, nrej_c = sep.sum_circle(
        data, sepx[use_circle], sepy[use_circle], r_min,
        err=err, subpix=1)
    fl[use_circle] = cfl
    dfl[use_circle] = cdfl
    nrej[use_circle] = nrej_c
    return fl, dfl, nrej
