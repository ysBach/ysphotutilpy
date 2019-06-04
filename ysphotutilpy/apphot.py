from warnings import warn
import numpy as np

from photutils import aperture_photometry as apphot
from photutils.centroids import centroid_com
# from photutils.detection import find_peaks

from astropy.table import hstack
from astropy.nddata import CCDData, Cutout2D
from astropy.stats import sigma_clipped_stats
from astropy import units as u

from .background import sky_fit

__all__ = ["apphot_annulus", "find_centroid_com"]

# TODO: Put centroiding into this apphot_annulus ?


def apphot_annulus(ccd, aperture, annulus, t_exposure=None,
                   exposure_key="EXPTIME", error=None, mask=None,
                   sky_keys={}, t_exposure_unit=u.s, verbose=False,
                   **kwargs):
    ''' Do aperture photometry using annulus.
    Parameters
    ----------
    ccd: CCDData
        The data to be photometried. Preferably in ADU.
    aperture, annulus: photutils aperture and annulus object
        The aperture and annulus to be used for aperture photometry.
    exposure_key: str
        The key for exposure time. Together with ``t_exposure_unit``, the
        function will normalize the signal to exposure time. If ``t_exposure``
        is not None, this will be ignored.
    error: array-like or Quantity, optional
        See ``photutils.aperture_photometry`` documentation.
        The pixel-wise error map to be propagated to magnitued error.
    sky_keys: dict
        kwargs of ``sky_fit``. Mostly one doesn't change the default setting,
        so I intentionally made it to be dict rather than usual kwargs, etc.
    **kwargs:
        kwargs for ``photutils.aperture_photometry``.

    Returns
    -------
    phot_f: astropy.table.Table
        The photometry result.
    '''
    _ccd = ccd.copy()

    if t_exposure is None:
        t_exposure = ccd.header[exposure_key]

    if error is not None:
        if verbose:
            print("Ignore any uncertainty extension in the original CCD "
                  + "and use provided error.")
        err = error.copy()
        if isinstance(err, CCDData):
            err = err.data

    else:
        try:
            err = ccd.uncertainty.array
        except AttributeError:
            if verbose:
                warn("Couldn't find Uncertainty extension in ccd. Will not calculate errors.")
            err = np.zeros_like(_ccd.data)

    if mask is not None:
        if _ccd.mask is not None:
            if verbose:
                warn("ccd contains mask, so given mask will be added to it.")
            _ccd.mask += mask
        else:
            _ccd.mask = mask

    skys = sky_fit(_ccd, annulus, **sky_keys)
    n_ap = aperture.area()
    phot = apphot(_ccd.data, aperture, mask=_ccd.mask, error=err, **kwargs)
    # If we use ``_ccd``, photutils deal with the unit, and the lines below
    # will give a lot of headache for units. It's not easy since aperture
    # can be pixel units or angular units (Sky apertures).
    # ysBach 2018-07-26

    phot_f = hstack([phot, skys])

    phot_f["source_sum"] = phot_f["aperture_sum"] - n_ap * phot_f["msky"]

    # see, e.g., http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?radprof.hlp
    # Poisson + RDnoise + dark + digitization noise:
    var_errmap = phot_f["aperture_sum_err"]**2
    # Sum of n_ap Gaussians (kind of random walk):
    var_skyrand = n_ap * phot_f["ssky"]**2
    # "systematic" uncertainty in the msky value:
    var_skysyst = (n_ap * phot_f['ssky'])**2 / phot_f['nsky']

    phot_f["source_sum_err"] = np.sqrt(var_errmap + var_skyrand + var_skysyst)

    phot_f["mag"] = -2.5 * np.log10(phot_f['source_sum'] / t_exposure)
    phot_f["merr"] = (2.5 / np.log(10)
                      * phot_f["source_sum_err"] / phot_f['source_sum'])

    return phot_f


def centroiding_iteration(ccd, position_xy, cbox_size=5., csigma=3.):
    ''' Find the intensity-weighted centroid of the image iteratively

    Returns
    -------
    xc_img, yc_img : float
        The centroided location in the original image coordinate in image XY.

    shift : float
        The total distance between the initial guess and the fitted centroid,
        i.e., the distance between `(xc_img, yc_img)` and `position_xy`.
    '''

    imgX, imgY = position_xy
    cutccd = Cutout2D(ccd.data, position=position_xy, size=cbox_size)
    avg, med, std = sigma_clipped_stats(cutccd.data, sigma=3, maxiters=5)
    cthresh = med + csigma * std
    # using pixels only above med + 3*std for centroiding is recommended.
    # See Ma+2009, Optics Express, 17, 8525
    mask = (cutccd.data < cthresh)
    if ccd.mask is not None:
        mask += ccd.mask
    xc_cut, yc_cut = centroid_com(data=cutccd.data, mask=mask)
    # Find the centroid with pixels have values > 3sigma, by center of mass
    # method. The position is in the cutout image coordinate, e.g., (3, 3).

    xc_img, yc_img = cutccd.to_original_position((xc_cut, yc_cut))
    # convert the cutout image coordinate to original coordinate.
    # e.g., (3, 3) becomes something like (137, 189)

    dx = xc_img - imgX
    dy = yc_img - imgY
    shift = np.sqrt(dx**2 + dy**2)
    return xc_img, yc_img, shift


def find_centroid_com(ccd, position_xy, maxiters=5, cbox_size=5., csigma=3.,
                      tol_shift=1.e-4, max_shift=1, verbose=False, full=False):
    ''' Find the intensity-weighted centroid iteratively.
    Simply run `centroiding_iteration` function iteratively for `maxiters` times.
    Given the initial guess of centroid position in image xy coordinate, it
    finds the intensity-weighted centroid (center of mass) after rejecting
    pixels by sigma-clipping.

    Parameters
    ----------
    ccd : CCDData or ndarray
        The whole image which the `position_xy` is calculated.

    position_xy : array-like
        The position of the initial guess in image XY coordinate.

    cbox_size : float or int, optional
        The size of the box to find the centroid. Recommended to use 2.5 to
        4.0 `FWHM`. See: http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?centerpars
        Minimally about 5 pixel is recommended. If extended source (e.g.,
        comet), recommend larger cbox.

    csigma : float or int, optional
        The parameter to use in sigma-clipping. Using pixels only above 3-simga
        level for centroiding is recommended. See Ma+2009, Optics Express, 17,
        8525.

    tol_shift : float
        The tolerance for the shift. If the shift in centroid after iteration
        is smaller than this, iteration stops.

    max_shift: float
        The maximum acceptable shift. If shift is larger than this, raises
        warning.

    verbose : bool
        Whether to print how many iterations were needed for the centroiding.

    full : bool
        Whether to return the original and final cutout images.
    Returns
    -------
    com_xy : list
        The iteratively found centroid position.
    '''
    if not isinstance(ccd, CCDData):
        ccd = CCDData(ccd, unit='adu')  # Just a dummy

    i_iter = 0
    xc_iter = [position_xy[0]]
    yc_iter = [position_xy[1]]
    shift = []
    d = 0
    if verbose:
        print(f"Initial xy: ({xc_iter[0]}, {yc_iter[0]}) [0-index]")
        print(f"With max iteration {maxiters:d}, shift tolerance {tol_shift}")

    while (i_iter < maxiters) and (d < tol_shift):
        xy_old = [xc_iter[-1], yc_iter[-1]]

        x, y, d = centroiding_iteration(ccd=ccd,
                                        position_xy=xy_old,
                                        cbox_size=cbox_size,
                                        csigma=csigma)
        xc_iter.append(x)
        yc_iter.append(y)
        shift.append(d)
        i_iter += 1
        if verbose:
            print(f"Iteration {i_iter:d} / {maxiters:d}: "
                  + f"({x:.2f}, {y:.2f}), shifted {d:.2f}")

    newpos = [xc_iter[-1], yc_iter[-1]]
    dx = x - position_xy[0]
    dy = y - position_xy[1]
    total = np.sqrt(dx**2 + dy**2)

    if verbose:
        print(f"Final shift: dx={dx:+.2f}, dy={dy:+.2f}, total={total:.2f}")

    if total > max_shift:
        warn(f"Shift is larger than {max_shift} ({total:.2f}).")

    # if verbose:
    #     print('Found centroid after {} iterations'.format(i_iter))
    #     print('Initially {}'.format(position_xy))
    #     print('Converged ({}, {})'.format(xc_iter[i_iter], yc_iter[i_iter]))
    #     shift = position_xy - np.array([xc_iter[i_iter], yc_iter[i_iter]])
    #     print('(Python/C-like indexing, not IRAF/FITS/Fortran)')
    #     print()
    #     print('Shifted to {}'.format(shift))
    #     print('\tShift tolerance was {}'.format(tol_shift))

    if full:
        original_cut = Cutout2D(data=ccd.data,
                                position=position_xy,
                                size=cbox_size)
        final_cut = Cutout2D(data=ccd.data,
                             position=newpos,
                             size=cbox_size)
        return newpos, original_cut, final_cut

    return newpos
