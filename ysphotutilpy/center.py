from warnings import warn

import numpy as np
import pandas as pd
from astropy.modeling import Fittable2DModel, Parameter
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import Const2D, Gaussian2D
from astropy.nddata import CCDData, Cutout2D, support_nddata
from photutils.centroids import centroid_com
from scipy import ndimage
from scipy.optimize import curve_fit

from .background import sky_fit
from .seputil import (_sanitize_byteorder, _sep_extract, sep_default_kernel,
                      sep_extract)
from .util import Gaussian2D_correct

__all__ = ["circular_slice", "circular_bbox_cut",
           "center_sep",
           "find_center_2dg", "find_centroid"]


def circular_slice(shape, pos, radius, resturn_offset=False):
    offset = np.array([max(0, _p - radius) for _p in pos])
    slices = [slice(int(_o), min(_n, int(_p+radius)+1))
              for _o, _p, _n in zip(offset[::-1], pos[::-1], shape)]
    #         flooring by `int`, ceiling by "`int` and +1"
    if resturn_offset:
        return tuple(slices), np.array([sl.start for sl in slices[::-1]])
    return tuple(slices)


# TODO: use "min_fraction" to mask any pixel that has that fraction as inside
# the circle. If it is 0, then it is the same as the current implementation. If
# "center", photutils's aperture center-like.
def circular_bbox_cut(img, pos, radius, return_dists=False):
    """
    Any pixel that has a non-zero fraction within the circle OR touches the
    circle are extracted.
    """
    img = np.array(img)
    # position difference between the current position and cutout position
    sl, offset = circular_slice(img.shape, pos, radius, resturn_offset=True)
    cut = img[sl]
    pos_cut = np.array(pos) - offset
    if return_dists:
        grids = np.meshgrid(*[np.arange(_n) for _n in cut.shape])
        dists = np.sqrt(np.sum(
            [(g - p)**2 for g, p in zip(grids, pos_cut[::-1])],
            axis=0
        )).T
        return cut, pos_cut, offset, dists
    return cut, pos_cut, offset


def _scaling_shift(pos_old, pos_new_raw, max_shift_step=None, verbose=False):
    dxdy = np.array(pos_new_raw) - np.array(pos_old)
    shift = np.linalg.norm(dxdy)

    if max_shift_step is None:
        return dxdy, shift

    if shift > max_shift_step:
        if verbose:
            print(f"\tshift({shift:.3f}) > max_shift_step({max_shift_step:.3f}): "
                  + f"shift truncated to {max_shift_step:.3f} pixels.")
        scale = max_shift_step / shift
        shift *= scale
        dxdy *= scale

    return dxdy, shift


def _background(data, bkg="min"):
    """Estimate the background level from the data
    Parameters
    ----------
    data : array_like
        The data array. It is expected `data` is a very small array (optimally
        side length 2-4 times FWHM) containing the object.

    bkg: str or float or array_like or sep.Background
        The background estimation method::

         * "min": use the minimum value of the data.
         * "mean": use the mean value of the data.
         * "median": use the median value of the data.
         * float: use the given value as the background level.
         * array_like: use the given array as the background level.
    """
    if bkg == "min":
        # min/nanmin takes short time.
        # An example of 7x7 data, min/nanmin=1.6/2.4 us.
        return np.nanmin(data)
    elif bkg == "mean":
        # Mostly it will have no NaN.
        # An example of 7x7 data, mean/nanmean=3/12 us.
        bkg = np.mean(data)
        if np.isnan(bkg):
            bkg = np.nanmean(data)
        return bkg
    elif bkg == "median":
        # median will anyway take long time.
        # An example of 7x7 data, median/nanmedian=9/12 us.
        return np.nanmedian(data)
    elif isinstance(bkg, (int, float, np.ndarray)):
        return bkg
    elif bkg is None:
        return 0
    else:  # it must be sep.Background
        try:
            return bkg.back()
        except AttributeError:
            raise TypeError("bkg must be one of 'min', 'mean', 'median', "
                            + "float, array_like, or sep.Background.")


def resample_psf(data, zoom=5, order=3, mode='constant',
                 cval=0.0, prefilter=True, *, grid_mode=False):
    data_min, data_max = data.min(), data.max()
    y_max, x_max = np.argwhere(data==data_max)
    _data = data - data_min
    _GAUSS_AMP = _data.max() - _data.min()

    def _gaussx(x, fwhm):
        return _GAUSS_AMP * np.exp(-4 * np.log(2) * (x - x_max)**2 / fwhm**2)
    def _gaussy(y, fwhm):
        return _GAUSS_AMP * np.exp(-4 * np.log(2) * (y - y_max)**2 / fwhm**2)

    poptx = curve_fit(_gaussx, np.arange(_data.shape[1]), _data[y_max, :])[0]
    popty = curve_fit(_gaussy, np.arange(_data.shape[0]), _data[:, x_max])[0]
    pos = np.array([poptx[0], popty[0]])
    fwhm = np.mean([min(max(1, poptx[1]), data.shape[1]),
                    min(max(1, popty[1]), data.shape[0])])
    gauss = (_gaussx(np.arange(_data.shape[1]), *poptx)
             * _gaussy(np.arange(_data.shape[0]), *popty)[:, None])
    gauss = ndimage.zoom(gauss, zoom, order=order, mode=mode,
                         cval=cval, prefilter=prefilter, grid_mode=grid_mode)
    spl = ndimage.zoom(_data - gauss, zoom, order=order, mode=mode,
                       cval=cval, prefilter=prefilter, grid_mode=grid_mode)
    return gauss + spl + data_min, pos, fwhm

def _center_sep(
    data,
    position,
    crad=3,
    move2="",
    bkg="min",
    thresh=0,
    mask=None,
    err=None,
    var=None,
    minarea=5,
    max_shift_step=None,
    full=False,
    verbose=0,
    **kwargs
):
    """ Centroiding by SEP
    Parameters
    ----------
    data : array_like
        The data array.

    thresh : float
        The detection threshold. It is the thresh value **AFTER** backgroud
        subtraction, so ``0`` is default. Otherwise, a possibility is to use 3
        times the sky standard deviation, etc.

    move2 : {"", "peak", "cpeak"}
        The suffix to "x" and "y" to move to. If "peak", it moves to the
        brightest pixel. If "cpeak", it moves to the brightest pixel within the
        convolved box. This is useful for the very first iteration.

    Notes
    -----
    If more than one (cbox_size is too large), it automatically selects the
    brightest object.

    """
    cut_slices, cut_offset = circular_slice(data.shape, position, crad, resturn_offset=True)
    cut, pos_cut, offset, dists = circular_bbox_cut(data, position, crad, return_dists=True)
    in_circ = dists <= crad
    mask_cut = mask[cut_slices] | ~in_circ if mask is not None else ~in_circ
    err_cut = err[cut_slices] if err is not None else None
    var_cut = var[cut_slices] if var is not None else None

    if isinstance(bkg, np.ndarray):
        bkg = bkg[cut_slices]

    data_skysub = cut - _background(cut, bkg=bkg)
    obj, seg = _sep_extract(
        data_skysub,
        thresh=thresh,
        mask=_sanitize_byteorder(mask_cut),
        err=_sanitize_byteorder(err_cut),
        var=_sanitize_byteorder(var_cut),
        minarea=minarea,
        seg_remove_mask=False,
        **kwargs
    )

    nobj = len(obj)
    if nobj == 0:
        if verbose:
            print("No object found in the cutout.")
        if full:
            return position, np.array([0, 0]), obj, seg
        return position, np.array([0, 0])

    if nobj > 1:
        if verbose:
            print(f"{nobj} objects found in the cutout. Selecting the brightest...")
        obj = obj[obj["flux"] == obj["flux"].max()]

    dxdy, shift = _scaling_shift(
        position,
        (np.array([float(obj[0][f"x{move2}"]), float(obj[0][f"y{move2}"])])
         + cut_offset),
        max_shift_step=max_shift_step,
        verbose=verbose
    )
    pos_new = position + dxdy

    n_circ = np.sum(in_circ)
    n_src = np.sum(seg == 1)

    if verbose:
        print(f"\t{n_circ} / {cut.size} pixels used for SEP, {n_src} pixels "
              "identified as the source.")

    overflow = True if n_src > n_circ else False

    if full:
        return pos_new, shift, obj, seg
    return pos_new, shift


def center_sep(
    data,
    position,
    err=None,
    var=None,
    crad=3,
    bkg="min",
    mask=None,
    maskthresh=0.0,
    minarea=5,
    filter_kernel=sep_default_kernel,
    filter_type='matched',
    deblend_nthresh=32,
    deblend_cont=1,
    clean=False,
    clean_param=1.,
    maxiters=50,
    tol_shift=1.e-3,
    max_shift=1,
    max_shift_step=1,
    verbose=False,
    full=False
):
    """Find the center of the object by SEP.
    Parameters
    ----------
    data : CCDData, array_like
        The data array.

    position_xy : array-like
        The position of the initial guess in image XY coordinate.

    crad : float
        The radius of the circular cutout from data for center finding.
        Approximately 2-4 times FWHM is recommended, but not too large so that
        only one single object is contained in the cutout. Approximately
        ``2*crad+1`` is the side length of the cutout array called **cbox**
        (stands for "centering box", identical to bounding box)

    bkg : {'min', 'mean', 'median', float, array_like, sep.Background}, optional
        The background estimation method within the cbox::

         * "min": use the minimum value of the data.
         * "mean": use the mean value of the data.
         * "median": use the median value of the data.
         * float: use the given value as the background level.
         * array_like: use the given array as the background level.

    tol_shift : float
        The absolute tolerance for the shift. If the shift in centroid after
        iteration is smaller than this, iteration stops.

    max_shift: float
        The maximum acceptable shift. If shift is larger than this, raises
        warning.

    max_shift_step : float, None, optional
        The maximum acceptable shift for each iteration. If the shift (call it
        ``shift_raw``) is larger than this, the actual shift will be
        `max_shift_step` towards the direction identical to ``shift_raw``. If
        `None` (default), no such truncation is done.

    deblend_cont = 1: no deblending (sep default = 0.005)
    clean = False (sep default = True)
    crad : float
        The radius of the circular cutout from data.
        It is expected `crad` cutout array is a very small array (optimally
        side length 2-4 times FWHM) containing only a single object.

    """
    pos = np.array(position)
    positions = [pos]
    shifts, objs, segs = [], [], []
    crad = max(crad, 1)

    if verbose >= 1:
        print("*** Centering by SEP ***")
        print(f"Initial xy: ({positions[0][0]}, {positions[0][1]}) [0-index]"
              + f"\n\t{crad=:.1f}, {maxiters=:d}, {tol_shift=:.1e}")
        if bkg in ["min", "mean", "median"]:
            print(f"\tBackgound = **{bkg}** value inside the cutout")
        elif isinstance(bkg, (int, float)):
            print(f"\tConstant backgound = {bkg}")
        elif isinstance(bkg, np.ndarray):
            print(f"\tBackgound array: {bkg.shape}")
        else:
            print(f"\tBackgound: {bkg} (sep.Background)")

    try:
        bkg = bkg.back()  # convert to ndarray
    except AttributeError:
        pass

    for i_iter in range(maxiters):
        if verbose >= 2:
            print(f"Iteration {i_iter+1:d} / {maxiters:d}: ")

        pos, d, obj, seg = _center_sep(
            data,
            pos,
            crad=crad,
            move2="",
            bkg=bkg,
            mask=mask,
            maskthresh=maskthresh,
            err=err,
            var=var,
            minarea=minarea,
            filter_kernel=filter_kernel,
            filter_type=filter_type,
            deblend_nthresh=deblend_nthresh,
            deblend_cont=deblend_cont,
            clean=clean,
            clean_param=clean_param,
            max_shift_step=max_shift_step,
            full=True,
            verbose=verbose>=2,
        )
        positions.append(pos)
        shifts.append(d)
        objs.append(obj)
        segs.append(seg)

        if verbose >= 2:
            print(f"\t({pos[0]:.2f}, {pos[1]:.2f}) [dx = {pos[0] - positions[-2][0]:.2f}, "
                  + f"dy = {pos[1] - positions[-2][1]:.2f} --> shift = {d:.2f}]")
        if d < tol_shift:
            if verbose >= 2:
                print(f"*** Finished at iteration {i_iter+1}. ***")
            break

    # `pos` is the final position
    dxdy = pos - positions[0]
    total = np.linalg.norm(dxdy)

    if verbose >= 1:
        print(f"   (x, y) = ({positions[0][0]:8.2f}, {positions[0][1]:8.2f})")
        print(f"        --> ({positions[-1][0]:8.2f}, {positions[-1][1]:8.2f}) [0-index]")
        print(f" (dx, dy) = ({dxdy[0]:+8.2f}, {dxdy[1]:+8.2f})")
        print(f" total shift {total:8.2f} pixels")

    if total > max_shift:
        warn(f"Object with initial position ({positions[0]}) "
             + f"shifted larger than {max_shift} ({total:.2f}).")

    if full:
        return pos, dxdy, total, objs, segs

    return pos


# TODO: add mask
def _centroiding_iteration(
        ccd, position_xy,
        centroider=centroid_com,
        cbox_size=5.,
        csigma=3,
        max_shift_step=None,
        msky=None,
        error=0,
        verbose=False
):
    ''' Find the intensity-weighted centroid of the image iteratively

    Returns
    -------
    ccd : `~astropy.nddata.CCDData`
        The full CCD image.

    position_xy : float
        The position of the initial guess in image XY coordinate. It is assumed
        that the `position_xy` is already quite close to the centroid, so both
        `msky` and `error`(if constant) are not needed to be updated at every
        iteration.

    cbox_size : float or int, optional
        The size of the box to find the centroid. Recommended to use 2.5 to 4.0
        times the seeing FWHM. Minimally about 5 pixel is recommended. If
        extended source (e.g., comet), recommend larger cbox.
        See:
        https://iraf.readthedocs.io/en/latest/tasks/noao/digiphot/apphot/centerpars.html

    csigma : float or int, optional
        The parameter to use in sigma-clipping. If `None`, pixels ABOVE (not
        equal to) the minimum pixel value within the `cbox_size` will be used.
        If `0` (actually if < 1.e-6), pixels ABOVE the "mean pixel value within
        the `cbox_size`" will be used (IRAF default?). Otherwise, pixels above
        ``msky + csigma*error`` will be used.
        Default is 0

    msky : float, optional
        The approximate value of the sky or background. If `None` (default),
        it is the minimum pixel value within the `cbox_size`.

    error : float, ndarray, optional
        The 1-sigma error-bar for each pixel. If float, a constant error-bar
        (e.g., sample standard deviation of sky) is assumed. If array-like, it
        should be in the shape of ccd. Ignored if `csigma` is
        `None` or `0`.
        Default is 0.

    max_shift_step : float, None, optional
        The maximum acceptable shift for each iteration. If the shift (call it
        ``shift_raw``) is larger than this, the actual shift will be
        `max_shift_step` towards the direction identical to `shift_raw`. If
        `None` (default), no such truncation is done.

    verbose : bool
        Whether to print how many iterations were needed for the centroiding.

    Returns
    -------
    pos_new : ndarray
        The new centroid position in image XY coordinate.

    shift : float
        The total distance between the initial guess and the fitted centroid,
        i.e., the distance between `(xc_img, yc_img)` and `position_xy`.
    '''

    # x_init, y_init = position_xy
    _cutkw = dict(position=position_xy, size=cbox_size)
    cutccd = Cutout2D(ccd.data, **_cutkw)

    _mindata = np.min(cutccd.data)  # TODO: use np.nanmin?

    if csigma is None:
        cthresh = _mindata
        if verbose:
            info = f"minimum value within {cbox_size = }"
    elif csigma < 1.e-6:
        cthresh = np.mean(cutccd.data)  # TODO: use np.nanmean?
        if verbose:
            info = f"mean value within {cbox_size = }"
    else:
        msky = _mindata if msky is None else msky
        error = Cutout2D(error, **_cutkw).data if isinstance(error, np.ndarray) else error
        cthresh = msky + csigma * error
        if (cthresh < _mindata):
            if verbose:
                print(f"\tthreshold = {cthresh:.3f} < (min of pixels in cbox).")
                print(f"\t∴ threshold = {_mindata:.3f} (min of pixels in cbox)...")
            # msky = _mindata
            cthresh = _mindata

    mask = (cutccd.data <= (cthresh+1.e-10))
    if ccd.mask is not None:
        mask += Cutout2D(ccd.mask, **_cutkw).data

    if verbose:
        n_all = np.size(mask)
        n_rej = np.count_nonzero(mask.astype(int))
        if isinstance(cthresh, np.ndarray):
            info = f"\t{n_rej} / {n_all} rejected [threshold = "
        else:
            info = f"\t{n_rej} / {n_all} rejected [threshold = {cthresh:.3f} = "

        if csigma is None:
            info += f"minimum within {cbox_size = }]"
        elif csigma < 1.e-6:
            info += f"mean within {cbox_size = }]"
        else:
            if not isinstance(error, np.ndarray):
                info += f"({msky=:.3f}) + ({csigma=}) * {error=:.3f}]"
            else:
                info += f"({msky=:.3f}) + ({csigma=}) * (error=ndarray)]"
        print(info)

    x_c_cut, y_c_cut = centroider(data=cutccd.data, mask=mask)
    # The position is in the cutout image coordinate, e.g., (3, 3).

    # x_c, y_c = cutccd.to_original_position((x_c_cut, y_c_cut))
    # convert the cutout image coordinate to original coordinate.
    # e.g., (3, 3) becomes something like (137, 189)

    pos_new_raw = cutccd.to_original_position((x_c_cut, y_c_cut))
    # convert the cutout image coordinate to original coordinate.
    # e.g., (3, 3) becomes something like (137, 189)

    dxdy, shift = _scaling_shift(
        position_xy,
        pos_new_raw,
        max_shift_step=max_shift_step,
        verbose=verbose
    )
    pos_new = position_xy + dxdy

    return pos_new, shift


def _fit_2dgaussian(data, error=None, mask=None):
    """
    Fit a 2D Gaussian plus a constant to a 2D image.
    Non-finite values (e.g., NaN or inf) in the ``data`` or ``error``
    arrays are automatically masked. These masks are combined.

    Parameters
    ----------
    data : array_like
        The 2D array of the image.
    error : array_like, optional
        The 2D array of the 1-sigma errors of the input ``data``.
    mask : array_like (bool), optional
        A boolean mask, with the same shape as ``data``, where a `True`
        value indicates the corresponding element of ``data`` is masked.

    Returns
    -------
    result : A `GaussianConst2D` model instance.
        The best-fitting Gaussian 2D model.
    """
    data = np.ma.asanyarray(data)

    if mask is not None and mask is not np.ma.nomask:
        mask = np.asanyarray(mask)
        if data.shape != mask.shape:
            raise ValueError('data and mask must have the same shape.')
        data.mask |= mask

    if np.any(~np.isfinite(data)):
        data = np.ma.masked_invalid(data)
        warn('Input data contains non-finite values (e.g., NaN or infs) that were '
             + 'automatically masked.', UserWarning)

    if error is not None:
        error = np.ma.masked_invalid(error)
        if data.shape != error.shape:
            raise ValueError('data and error must have the same shape.')
        data.mask |= error.mask
        weights = 1.0 / error.clip(min=1.e-30)
    else:
        weights = np.ones(data.shape)

    if np.ma.count(data) < 7:
        raise ValueError('Input data must have a least 7 unmasked values to '
                         'fit a 2D Gaussian plus a constant.')

    # assign zero weight to masked pixels
    if data.mask is not np.ma.nomask:
        weights[data.mask] = 0.

    mask = data.mask
    data.fill_value = 0.
    data = data.filled()

    # Subtract the minimum of the data as a rough background estimate.
    # This will also make the data values positive, preventing issues with
    # the moment estimation in data_properties. Moments from negative data
    # values can yield undefined Gaussian parameters, e.g., x/y_stddev.
    props = sep_extract(
        data - np.min(data),
        thresh=0.0,              # Use all data points
        mask=mask,
        filter_kernel=None,      # No convolution
        deblend_cont=1.0,        # No deblending
        clean=False,             # No cleaning
        segmentation_map=False,  # No segmentation map
    )[0]

    init_const = 0.  # subtracted data minimum above
    init_amplitude = np.ptp(data)
    g_init = GaussianConst2D(constant=init_const,
                             amplitude=init_amplitude,
                             x_mean=props["x"].iloc[0],
                             y_mean=props["y"].iloc[0],
                             x_stddev=props["a"].iloc[0],
                             y_stddev=props["b"].iloc[0],
                             theta=props["theta"].iloc[0])
    fitter = LevMarLSQFitter()
    y, x = np.indices(data.shape)
    gfit = fitter(g_init, x, y, data, weights=weights)

    return gfit


class GaussianConst2D(Fittable2DModel):
    """
    A model for a 2D Gaussian plus a constant.
    Parameters
    ----------
    constant : float
        Value of the constant.
    amplitude : float
        Amplitude of the Gaussian.
    x_mean : float
        Mean of the Gaussian in x.
    y_mean : float
        Mean of the Gaussian in y.
    x_stddev : float
        Standard deviation of the Gaussian in x. ``x_stddev`` and
        ``y_stddev`` must be specified unless a covariance matrix
        (``cov_matrix``) is input.
    y_stddev : float
        Standard deviation of the Gaussian in y. ``x_stddev`` and
        ``y_stddev`` must be specified unless a covariance matrix
        (``cov_matrix``) is input.
    theta : float, optional
        Rotation angle in radians. The rotation angle increases
        counterclockwise.
    """

    constant = Parameter(default=0)
    amplitude = Parameter(default=1)
    x_mean = Parameter(default=0)
    y_mean = Parameter(default=0)
    x_stddev = Parameter(default=1)
    y_stddev = Parameter(default=1)
    theta = Parameter(default=0)

    @staticmethod
    def evaluate(x, y, constant, amplitude, x_mean, y_mean, x_stddev, y_stddev, theta):
        """Two dimensional Gaussian plus constant function."""
        return (Const2D(constant)(x, y)
                + Gaussian2D(amplitude, x_mean, y_mean, x_stddev, y_stddev, theta)(x, y))


def find_center_2dg(
        ccd,
        position_xy,
        cbox_size=5.,
        csigma=3.,
        msky=None,
        ssky=0,
        sky_annulus=None,
        sky_kw={},
        maxiters=5,
        error=None,
        atol_shift=1.e-4,
        max_shift=1,
        max_shift_step=None,
        verbose=False,
        full=True,
        full_history=False
):
    ''' Find the center of the image by 2D Gaussian fitting.

    Parameters
    ----------
    ccd : CCDData or ndarray
        The whole image which the `position_xy` is calculated.

    position_xy : array-like
        The position of the initial guess in image XY coordinate.

    cbox_size : float or int, optional
        The size of the box to find the centroid. Recommended to use 2.5 to 4.0
        times the seeing FWHM. Minimally about 5 pixel is recommended. If
        extended source (e.g., comet), recommend larger cbox.
        See:
        https://iraf.readthedocs.io/en/latest/tasks/noao/digiphot/apphot/centerpars.html

    csigma : float or int, optional
        The parameter to use in sigma-clipping. Using pixels only above 3-simga
        level for centroiding is recommended. See Ma+2009, Optics Express, 17,
        8525.

    ssky : float, optional
        The sample standard deviation of the sky or background. It will be used
        instead of `sky_annulus` if `sky_annulus` is `None`. The pixels above
        the local minima (of the array of size `cbox_size`) plus
        ``csigma*ssky`` will be used for centroid, following the default of
        IRAF's ``noao.digiphot.apphot``:
        https://iraf.readthedocs.io/en/latest/tasks/noao/digiphot/apphot/centerpars.html

    sky_annulus : `~photutils.Aperture` annulus object
        The annulus to estimate the sky. All `_shape_params` of the object will
        be kept, while positions will be updated according to the new
        centroids. The initial input, therefore, does not need to have the
        position information (automatically initialized by `position_xy`). If
        `None` (default), the constant `ssky` value will be used instead.

    sky_kw : dict, optional.
        The keyword arguments of `.backgroud.sky_fit`.

    tol_shift : float
        The absolute tolerance for the shift. If the shift in centroid after
        iteration is smaller than this, iteration stops.

    max_shift: float
        The maximum acceptable shift. If shift is larger than this, raises
        warning.

    max_shift_step : float, None, optional
        The maximum acceptable shift for each iteration. If the shift (call it
        ``shift_raw``) is larger than this, the actual shift will be
        `max_shift_step` towards the direction identical to `shift_raw`. If
        `None` (default), no such truncation is done.

    error : CCDData or ndarray
        The 1-sigma uncertainty map used for fitting.

    verbose : bool
        Whether to print how many iterations were needed for the centroiding.

    full : bool
        Whether to return the original and final cutout images.

    full_history : bool, optional
        Whether to return all the history of memory-heavy objects, including
        cutout ccd, cutout of the error frame, evaluated array of the fitted
        models. Most likely this is turned on only to check whether the
        centroiding process works correctly (i.e., kind of debugging purpose).

    Returns
    -------
    newpos : tuple
        The iteratively found centroid position.

    shift : float
        Total shift from the initial position.

    fulldict : dict
        The ``dict`` when returned if ``full=True``:
            * ``positions``: `Nx2` numpy.array
        The history of ``x`` and ``y`` positions. The 0-th element is the
        initial position and the last element is the final fitted position.
    '''
    if sky_annulus is not None:
        import copy
        ANNULUS = copy.deepcopy(sky_annulus)
    else:
        ANNULUS = None

    def _center_2dg_iteration(ccd, position_xy, cbox_size=5., csigma=3.,
                              max_shift_step=None, msky=None, ssky=0,
                              error=None, verbose=False):
        ''' Find the intensity-weighted centroid of the image iteratively

        Returns
        -------
        position_xy : float
            The centroided location in the original image coordinate in
            image XY.

        shift : float
            The total distance between the initial guess and the fitted
            centroid, i.e., the distance between `(xc_img, yc_img)` and
            `position_xy`.
        '''

        cut = Cutout2D(ccd.data, position=position_xy, size=cbox_size)
        e_cut = Cutout2D(error.data, position=position_xy, size=cbox_size)

        if ANNULUS is not None:
            ANNULUS.positions = position_xy
            _sky = sky_fit(ccd=ccd, annulus=ANNULUS, **sky_kw, to_table=False)
            ssky = _sky["ssky"]
            msky = _sky["msky"]
        elif msky is None:
            msky = np.min(cut.data)

        cthresh = msky + csigma * ssky
        mask = (cut.data < cthresh)
        # using pixels only above med + 3*std for centroiding is recommended.
        # See Ma+2009, Optics Express, 17, 8525
        # -- I doubt this... YPBach 2019-07-08 10:43:54 (KST: GMT+09:00)

        if verbose:
            n_all = np.size(mask)
            n_rej = np.count_nonzero(mask.astype(int))
            print(f"\t{n_rej} / {n_all} rejected [threshold = {cthresh:.3f} from min "
                  + f"({np.min(cut.data):.3f}) + ({csigma = }) * ({ssky = :.3f})]")

        if ccd.mask is not None:
            cutmask = Cutout2D(ccd.mask, position=position_xy, size=cbox_size)
            mask += cutmask

        yy, xx = np.mgrid[:cut.data.shape[0], :cut.data.shape[1]]
        g_fit = _fit_2dgaussian(data=cut.data, error=e_cut.data, mask=mask)
        g_fit = Gaussian2D_correct(g_fit)
        # The position is in the cutout image coordinate, e.g., (3, 3).

        dxdy, shift = _scaling_shift(
            position_xy,
            cut.to_original_position((g_fit.x_mean.value, g_fit.y_mean.value)),
            max_shift_step=max_shift_step,
            verbose=verbose
        )
        # ^ convert the cutout image coordinate to original coordinate.
        #   e.g., (3, 3) becomes something like (137, 189)

        pos_new = position_xy + dxdy

        return pos_new, shift, g_fit, cut, e_cut, g_fit(xx, yy)

    position_init = np.array(position_xy)
    if position_init.shape != (2,):
        raise TypeError("position_xy must have two and only two (xy) values.")

    _ccd = ccd.copy() if isinstance(ccd, CCDData) else CCDData(ccd, unit='adu')

    if error is not None:
        _error = error.copy() if isinstance(error, CCDData) else CCDData(error, unit='adu')
    else:
        _error = np.ones_like(_ccd.data)

    positions = [position_init]

    if full:
        mods = []
        shift = []
        cuts = []
        e_cuts = []
        fits = []
        fit_params = {}
        for k in GaussianConst2D.param_names:
            fit_params[k] = []

    if verbose:
        print(f"Initial xy: {position_init} [0-indexing]\n"
              + f"\t(max iteration {maxiters:d}, "
              + f"shift tolerance {atol_shift} pixel)")

    for i_iter in range(maxiters):
        xy_old = positions[-1]

        if verbose:
            print(f"Iteration {i_iter+1:d} / {maxiters:d}: ")

        res = _center_2dg_iteration(ccd=_ccd,
                                    position_xy=xy_old,
                                    cbox_size=cbox_size,
                                    csigma=csigma,
                                    msky=msky,
                                    ssky=ssky,
                                    error=_error,
                                    max_shift_step=max_shift_step,
                                    verbose=verbose)
        newpos, d, g_fit, cut, e_cut, fit = res

        if d < atol_shift:
            if verbose:
                print(f"Finishing iteration (shift {d:.5f} < tol_shift).")
            break

        positions.append(newpos)

        if full:
            for k, v in zip(g_fit.param_names, g_fit.parameters):
                fit_params[k].append(v)
            shift.append(d)
            mods.append(g_fit)
            cuts.append(cut)
            e_cuts.append(e_cut)
            fits.append(fit)

        if verbose:
            print(f"\t({newpos[0]:.2f}, {newpos[1]:.2f}), shifted {d:.2f}")

    total_dx_dy = positions[-1] - positions[0]
    total_shift = np.sqrt(np.sum(total_dx_dy**2))

    if verbose:
        print(f"Final shift: dx={total_dx_dy[0]:+.2f}, dy={total_dx_dy[1]:+.2f}, "
              + f"total_shift={total_shift:.2f}")

    if total_shift > max_shift:
        warn(f"Object with initial position {position_xy} "
             + f"shifted larger than {max_shift = } ({total_shift:.2f}).")

    if full:
        if full_history:
            fulldict = dict(
                positions=np.atleast_2d(positions),
                shifts=np.atleast_1d(shift),
                fit_models=mods,
                fit_params=fit_params,
                cuts=cuts,
                e_cuts=e_cuts,
                fits=fits
            )
        else:
            fulldict = dict(
                positions=np.atleast_2d(positions),
                shifts=np.atleast_1d(shift),
                fit_models=mods[-1],
                fit_params=fit_params,
                cuts=cuts[-1],
                e_cuts=e_cuts[-1],
                fits=fits[-1]
            )
        return positions[-1], total_shift, fulldict

    return positions[-1], total_shift


# TODO: Add error-bar of the centroids by accepting error-map
def find_centroid(
        ccd,
        position_xy,
        centroider=centroid_com,
        maxiters=5,
        cbox_size=5.,
        csigma=0,
        msky=None,
        error=0,
        tol_shift=1.e-4,
        max_shift=1,
        max_shift_step=None,
        verbose=False,
        full=False
):
    ''' Find the intensity-weighted centroid iteratively.

    Notes
    -----
    Cut out small box region (`cbox_size`) around the initial position, use
    pixels above certain threshold, and find the intensity-weighted centroid of
    the box after subtracting "background"::

      * If `csigma=0` (or < 1.e-6), use pixels only above the mean of the box.
        Ignore `msky` and `error`. (IRAF default?)
      * If `csigma=None`, use pixels only above the minimum of the box. Ignore
        `msky` and `error`.
      * If `csigma` is a positive number, use pixels above ``msky +
        csigma*error``.
        * If `msky` is not given, use the minimum pixel value within the box.

    Simply run `_centroiding_iteration` function iteratively for `maxiters`
    times. Given the initial guess of centroid position in image xy coordinate,
    it finds the intensity-weighted centroid (center of mass) after rejecting
    pixels by sigma-clipping.

    https://iraf.readthedocs.io/en/latest/tasks/noao/digiphot/daophot/centerpars.html

    Parameters
    ----------
    ccd : CCDData or ndarray
        The whole image which the `position_xy` is calculated.

    position_xy : array-like
        The position of the initial guess in image XY coordinate. It is assumed
        that the `position_xy` is already quite close to the centroid, so both
        `msky` and `ssky` are not needed to be updated at every iteration.

    centroider : callable
        The centroider function (Dafault uses `photutils.centroid_com`).

    cbox_size : float or int, optional
        The size of the box to find the centroid. Recommended to use 2.5 to 4.0
        times the seeing FWHM.  Minimally about 5 pixel is recommended. If
        extended source (e.g., comet), recommend larger cbox.
        See:
        https://iraf.readthedocs.io/en/latest/tasks/noao/digiphot/apphot/centerpars.html

    csigma : float or int, optional
        The parameter to use in sigma-clipping. If `None`, pixels ABOVE (not
        equal to) the minimum pixel value within the `cbox_size` will be used.
        If `0` (actually if < 1.e-6), pixels ABOVE the "mean pixel value within
        the `cbox_size`" will be used (IRAF default?). Otherwise, pixels above
        ``msky + csigma*error`` will be used.
        Default is 0

    msky : float, optional
        The approximate value of the sky or background. If `None` (default),
        it is the minimum pixel value within the `cbox_size`.

    error : float, ndarray, optional
        The 1-sigma error-bar for each pixel. If float, a constant error-bar
        (e.g., sample standard deviation of sky) is assumed. If array-like, it
        should be in the shape of ccd. Ignored if `csigma` is
        `None` or `0`.
        Default is 0.

    tol_shift : float
        The absolute tolerance for the shift. If the shift in centroid after
        iteration is smaller than this, iteration stops.

    max_shift: float
        The maximum acceptable shift. If shift is larger than this, raises
        warning.

    max_shift_step : float, None, optional
        The maximum acceptable shift for each iteration. If the shift (call it
        ``shift_raw``) is larger than this, the actual shift will be
        `max_shift_step` towards the direction identical to ``shift_raw``. If
        `None` (default), no such truncation is done.

    verbose : int
        Whether to print how many iterations were needed for the centroiding.

          * 0: No information is printed.
          * 1: Print the initial and final centroid information.
          * 2: Also print the information at each iteration.

    full : bool
        Whether to return the original and final cutout images.

    Returns
    -------
    com_xy : list
        The iteratively found centroid position.
    '''
    _ccd = CCDData(ccd, unit='adu') if not isinstance(ccd, CCDData) else ccd.copy()

    x, y = position_xy
    xc_iter = [x]
    yc_iter = [y]
    shift = []

    if verbose >= 1:
        print(f"Initial xy: ({xc_iter[0]}, {yc_iter[0]}) [0-index]"
              + f"\n\t({maxiters = :d}, {tol_shift = :.2e})")

    for i_iter in range(maxiters):
        if verbose >= 2:
            print(f"Iteration {i_iter+1:d} / {maxiters:d}: ")
        (x, y), d = _centroiding_iteration(
            ccd=_ccd,
            position_xy=(x, y),
            centroider=centroider,
            cbox_size=cbox_size,
            csigma=csigma,
            msky=msky,
            error=error,
            max_shift_step=max_shift_step,
            verbose=verbose >= 2
        )
        xc_iter.append(x)
        yc_iter.append(y)
        shift.append(d)
        if verbose >= 2:
            print(f"\t({x:.2f}, {y:.2f}) [dx = {x - xc_iter[-2]:.2f}, "
                  + f"dy = {y - yc_iter[-2]:.2f} --> shift = {d:.2f}]")
        if d < tol_shift:
            if verbose >= 2:
                print(f"*** Finished at {i_iter}th-iteration. ***")
            break

    newpos = [xc_iter[-1], yc_iter[-1]]
    dx = x - position_xy[0]
    dy = y - position_xy[1]
    total = np.sqrt(dx**2 + dy**2)

    if verbose >= 1:
        print(f"   (x, y) = ({xc_iter[0]:8.2f}, {yc_iter[0]:8.2f})")
        print(f"        --> ({xc_iter[-1]:8.2f}, {yc_iter[-1]:8.2f}) [0-index]")
        print(f" (dx, dy) = ({dx:+8.2f}, {dy:+8.2f})")
        print(f" total shift {total:8.2f} pixels")

    if total > max_shift:
        warn(f"Object with initial position ({xc_iter[-1]}, {yc_iter[-1]}) "
             + f"shifted larger than {max_shift} ({total:.2f}).")

    # if verbose:
    #     print('Found centroid after {} iterations'.format(i_iter))
    #     print('Initially {}'.format(position_xy))
    #     print('Converged ({}, {})'.format(xc_iter[i_iter], yc_iter[i_iter]))
    #     shift = position_xy - np.array([xc_iter[i_iter], yc_iter[i_iter]])
    #     print('(Python/C-like indexing, not IRAF/FITS/Fortran)')
    #     print()
    #     print('Shifted to {}'.format(shift))
    #     print('\tShift tolerance was {}'.format(tol_shift))

    # if full:
    #     original_cut = Cutout2D(data=ccd.data,
    #                             position=position_xy,
    #                             size=cbox_size)
    #     final_cut = Cutout2D(data=ccd.data,
    #                          position=newpos,
    #                          size=cbox_size)
    #     return newpos, original_cut, final_cut

    if full:
        return newpos, np.array(xc_iter), np.array(yc_iter), total

    return newpos


"""
def find_centroid_com(ccd, position_xy, maxiters=5, cbox_size=5., csigma=3.,
                      ssky=0, tol_shift=1.e-4, max_shift=1,
                      max_shift_step=None, verbose=False, full=False):
    ''' Find the intensity-weighted centroid iteratively.
    Simply run `centroiding_iteration` function iteratively for `maxiters`
    times. Given the initial guess of centroid position in image xy coordinate,
    it finds the intensity-weighted centroid (center of mass) after rejecting
    pixels by sigma-clipping.

    Parameters
    ----------
    ccd : CCDData or ndarray
        The whole image which the `position_xy` is calculated.

    position_xy : array-like
        The position of the initial guess in image XY coordinate.

    cbox_size : float or int, optional
        The size of the box to find the centroid. Recommended to use 2.5 to 4.0
        times the seeing FWHM. Minimally about 5 pixel is recommended. If
        extended source (e.g., comet), recommend larger cbox.
        See:
        http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?centerpars

    csigma : float or int, optional
        The parameter to use in sigma-clipping. Using pixels only above 3-simga
        level for centroiding is recommended. See Ma+2009, Optics Express, 17,
        8525.

    ssky : float, optional
        The sample standard deviation of the sky or background. It will be used
        instead of `sky_annulus` if `sky_annulus` is `None`. The pixels above
        the local minima (of the array of size `cbox_size`) plus
        ``csigma*ssky`` will be used for centroid, following the default of
        IRAF's ``noao.digiphot.apphot``:
        http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?centerpars.hlp

    tol_shift : float
        The absolute tolerance for the shift. If the shift in centroid after
        iteration is smaller than this, iteration stops.

    max_shift: float
        The maximum acceptable shift. If shift is larger than this, raises
        warning.

    max_shift_step : float, None, optional
        The maximum acceptable shift for each iteration. If the shift (call it
        ``shift_raw``) is larger than this, the actual shift will be
        `max_shift_step` towards the direction identical to `shift_raw`. If
        `None` (default), no such truncation is done.

    verbose : bool
        Whether to print how many iterations were needed for the centroiding.

    full : bool
        Whether to return the original and final cutout images.

    Returns
    -------
    com_xy : list
        The iteratively found centroid position.
    '''
    warn("DEPRECATED -- use find_centroid or find_center_2dg",
         AstropyDeprecationWarning)

    def _centroiding_iteration(ccd, position_xy, centroider=centroid_com,
                               cbox_size=5., csigma=3.,
                               max_shift_step=None, ssky=0, verbose=False):
        ''' Find the intensity-weighted centroid of the image iteratively

        Returns
        -------
        position_xy : float
            The centroided location in the original image coordinate
            in image XY.

        shift : float
            The total distance between the initial guess and the
            fitted centroid, i.e., the distance between `(xc_img,
            yc_img)` and `position_xy`.
        '''

        x_init, y_init = position_xy
        cutccd = Cutout2D(ccd.data, position=position_xy, size=cbox_size)
        cthresh = np.min(cutccd.data) + csigma * ssky
        # using pixels only above med + 3*std for centroiding is recommended.
        # See Ma+2009, Optics Express, 17, 8525
        # -- I doubt this... YPBach 2019-07-08 10:43:54 (KST: GMT+09:00)

        mask = (cutccd.data < cthresh)

        if verbose:
            n_all = np.size(mask)
            n_rej = np.count_nonzero(mask.astype(int))
            print(f"\t{n_rej} / {n_all} rejected [threshold = {cthresh:.3f} "
                  + f"from min ({np.min(cutccd.data):.3f}) "
                  + f"+ csigma ({csigma}) * ssky ({ssky:.3f})]")

        if ccd.mask is not None:
            mask += ccd.mask

        x_c_cut, y_c_cut = centroider(data=cutccd.data, mask=mask)
        # The position is in the cutout image coordinate, e.g., (3, 3).

        x_c, y_c = cutccd.to_original_position((x_c_cut, y_c_cut))
        # convert the cutout image coordinate to original coordinate.
        # e.g., (3, 3) becomes something like (137, 189)

        pos_new_raw = cutccd.to_original_position((x_c_cut, y_c_cut))
        # convert the cutout image coordinate to original coordinate.
        # e.g., (3, 3) becomes something like (137, 189)

        dxdy, shift = _scaling_shift(position_xy, pos_new_raw,
                                       max_shift_step=max_shift_step,
                                       verbose=verbose)
        pos_new = (position_xy[0] + dx, position_xy[1] + dy)

        return pos_new, shift

    if not isinstance(ccd, CCDData):
        _ccd = CCDData(ccd, unit='adu')  # Just a dummy
    else:
        _ccd = ccd.copy()

    i_iter = 0
    xc_iter = [position_xy[0]]
    yc_iter = [position_xy[1]]
    shift = []
    d = 0
    if verbose:
        print(f"Initial xy: ({xc_iter[0]}, {yc_iter[0]}) [0-index]")
        print(f"\t(max iteration {maxiters:d}, shift tolerance {tol_shift})")

    for i_iter in range(maxiters):
        xy_old = (xc_iter[-1], yc_iter[-1])
        if verbose:
            print(f"Iteration {i_iter+1:d} / {maxiters:d}: ")
        (x, y), d = _centroiding_iteration(ccd=_ccd,
                                           position_xy=xy_old,
                                           cbox_size=cbox_size,
                                           csigma=csigma,
                                           ssky=ssky,
                                           max_shift_step=max_shift_step,
                                           verbose=verbose)
        xc_iter.append(x)
        yc_iter.append(y)
        shift.append(d)
        if d < tol_shift:
            if verbose:
                print(f"Finishing iteration (shift {d:.5f} < tol_shift).")
            break
        if verbose:
            print(f"\t({x:.2f}, {y:.2f}), shifted {d:.2f}")

    newpos = [xc_iter[-1], yc_iter[-1]]
    dx = x - position_xy[0]
    dy = y - position_xy[1]
    total = np.sqrt(dx**2 + dy**2)

    if verbose:
        print(f"Final shift: dx={dx:+.2f}, dy={dy:+.2f}, total={total:.2f}")

    if total > max_shift:
        warn(f"Object with initial position ({xc_iter[-1]}, {yc_iter[-1]}) "
             + f"shifted larger than {max_shift} ({total:.2f}).")

    # if verbose:
    #     print('Found centroid after {} iterations'.format(i_iter))
    #     print('Initially {}'.format(position_xy))
    #     print('Converged ({}, {})'.format(xc_iter[i_iter], yc_iter[i_iter]))
    #     shift = position_xy - np.array([xc_iter[i_iter], yc_iter[i_iter]])
    #     print('(Python/C-like indexing, not IRAF/FITS/Fortran)')
    #     print()
    #     print('Shifted to {}'.format(shift))
    #     print('\tShift tolerance was {}'.format(tol_shift))

    # if full:
    #     original_cut = Cutout2D(data=ccd.data,
    #                             position=position_xy,
    #                             size=cbox_size)
    #     final_cut = Cutout2D(data=ccd.data,
    #                          position=newpos,
    #                          size=cbox_size)
    #     return newpos, original_cut, final_cut

    if full:
        return newpos, np.array(xc_iter), np.array(yc_iter), total

    return newpos
"""
