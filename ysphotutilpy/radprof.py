""" Radial profiles
"""

from itertools import repeat

import numpy as np
import pandas as pd
from astropy.nddata import CCDData, Cutout2D
from photutils.aperture import CircularAnnulus, CircularAperture
from scipy.ndimage import center_of_mass

from .background import sky_fit
from .center import circular_bbox_cut

__all__ = [
    "moffat_r",
    "gauss_r",
    "bivt_r",
    "radial_profile",
    "radprof_pix",
]


# def psfmeasure(img, pos, level=0.5, radius=5, maxiters=3, sky_buffer=5, sky_width=5,
#                saturation=None, verbose=False, **kwargs):
#     if isinstance(img, CCDData):
#         img = img.data
#     elif not isinstance(img, np.ndarray):
#         raise TypeError(f'img must be a CCDData or ndarray (now {type(img) = })')

#     pos_old = pos
#     crad = radius + sky_buffer + sky_width
#     for i in range(maxiters):
#         cut, dists, pos_cut = circular_cut(img, pos, radius=crad)


def gauss_r(r, amp=1, sig=1, const=0):
    return amp * np.exp(-0.5 * (r / sig) ** 2) + const


def moffat_r(r, amp=1, core=1, power=2.5, const=0):
    return amp * (1 + (r / core) ** 2) ** (-power) + const


def bivt_r(r, amp=1, num2=1, sig=1, const=0):
    """Bivariate t distribution (generalized Moffat).

    Parameters
    ----------
    r : array-like
        Radial distance.
    amp : float, optional
        Amplitude.
        Default is ``1``.
    num2 : float, optional
        The degrees of freedom (nu) minus 2.
        Default is ``1``.
    sig : float, optional
        Scale parameter (sigma).
        Default is ``1``.
    const : float, optional
        Constant background.
        Default is ``0``.

    Returns
    -------
    float or array-like
        The bivariate t profile values.
    """
    return amp * (1 + (r**2 / (num2 * sig**2))) ** (-(num2 + 4) / 2) + const


def fwhm_r(popt, fun):
    if fun == "gauss":
        return 2 * np.sqrt(2 * np.log(2)) * popt[1]
    elif fun == "moffat":
        return 2 * popt[1] * np.sqrt(2 ** (1 / popt[2]) - 1)
    elif fun == "bivt":
        # 2*sig*np.sqrt((nu-2)*(2**(2/(nu+2)) - 1))
        return 2 * popt[2] * np.sqrt(popt[1] * (2 ** (2 / (popt[1] + 4)) - 1))
    else:
        raise ValueError("Unknown function: {}".format(fun))


def radial_profile(
    im, center, radii=1, thickness=1, mask=None, norm_by_center=False, add_center=False, **kwargs
):
    """Calculate radial profile of the image.

    Parameters
    ----------
    im : 2D array
        The image data.
    center : tuple
        The (x, y) coordinates of the center.
    radii : 1D array
        The radii of the annulus.
    mask : 2D array, optional
        A mask to apply to the image. Pixels with True values will be ignored.
        Default is `None`.
    thickness : int or array of int, optional
        The thickness(es) of the annulus for the radial profile.
        Default is ``1``.
    norm_by_center : bool, optional
        If `True`, normalize the profile by the value at the center position.
        Default is `False`.
    add_center : bool, optional
        If `True`, include the center pixel value as the first entry in the
        profile (r = 0, mpix = center pixel value, spix = 0, npix = 1).
        Default is `False`.
    **kwargs : dict, optional
        Additional keyword arguments to pass to the `sky_fit` function.

    Returns
    -------
    profs : pandas.DataFrame
        A DataFrame containing the radial profile data, with columns for radius and sky fit parameters.
    center_val : float
        The value of the pixel at the center position.

    Example
    -------
    >>> profs, center_val = radial_profile(im, center=(50, 50), radii=np.arange(1, 11), thickness=2)
    >>> print(profs)
         r      mpix      spix  npix    spix_n
    0   0.0  1500.0000   0.0000     1   0.000000
    1   1.0   1450.1234  10.5678   12   3.045678
    2   2.0   1400.5678  15.6789   28   2.962345
    ...
    >>> profs["spix_n"] = profs["spix"] / np.sqrt(profs["npix"])
    >>> plt.errorbar(profs['r'], profs['mpix'], yerr=profs['spix_n'], fmt='o', color="k", capsize=3)
    >>> plt.errorbar(profs['r'], profs['mpix'], yerr=profs['spix'], fmt='', color="k", capsize=0, elinewidth=0.5)
    """
    radii = np.asarray(radii).ravel()
    if np.isscalar(thickness):
        thickness = repeat(thickness, len(radii))
    else:
        thickness = np.asarray(thickness).ravel()

    center_val = im[
        *(np.round(center).astype(int)[::-1])
    ]  # reverse for (y, x) indexing

    if add_center:
        # use original names (names for sky_fit before renaming)
        profs = [{"r": 0, "msky": center_val, "ssky": 0, "nsky": 1, "nrej": 0}]
    else:
        profs = []
    for r, _thick in zip(radii, thickness):
        an = CircularAnnulus(
            center, r_in=max(0.01, r - _thick / 2), r_out=r + _thick / 2
        )
        _skyfit = sky_fit(im, an, mask=mask, to_table=False, **kwargs)[0]
        _skyfit["r"] = r
        profs.append(_skyfit)

    profs = pd.DataFrame.from_dict(profs)
    profs = profs.rename(columns={"msky": "mpix", "ssky": "spix", "nsky": "npix"})
    profs["spix_n"] = profs["spix"] / np.sqrt(profs["npix"])

    if norm_by_center:
        _cval = np.abs(center_val)
        profs["mpix"] /= _cval
        profs["spix"] /= _cval
        profs["spix_n"] /= _cval
    return profs, center_val


def radprof_pix(img, pos, mask=None, rmax=10, sort_dist=False, fitfunc=None, refit=1):
    """Get radial profile (pixel values) of an object from n-D image.

    Parameters
    ----------
    img : CCDData or ndarray
        The image to be profiled.
    pos : array_like
        The xy coordinates of the center of the object (0-indexing).
    rmax : int, optional
        The maximum radius to be profiled.
        Default is ``10``.
    refit : float, None, optional
        If not None, refit the profile for pixels within max(refit*FWHM, 3). It this exceeds rmax, refit is ignored.
        Default is ``1``.

    Example
    -------
    >>> _r, _i, _f, popt, fwhm = ypu.radprof_pix(ccd, pos=ap_t.positions, fit="gauss")
    >>> rr = np.arange(0, _r.max()+1, 0.1)
    >>> axs.plot(_r, _i, "s", mfc="none")
    >>> axs.plot(rr, _f(rr, *popt), "-")
    >>> axs.axvline(fwhm/2)
    >>> axs.axhline(popt[-1])
    >>> axs.axhline(popt[0] + popt[-1])
    """

    if isinstance(img, CCDData):
        img = img.data
    elif not isinstance(img, np.ndarray):
        raise TypeError(f"img must be a CCDData or ndarray (now {type(img) = })")

    cut, _, _, dists = circular_bbox_cut(img, pos, radius=rmax, return_dists=True)
    mask = (dists > rmax) if mask is None else ((dists > rmax) | mask)

    # cut = Cutout2D(img, pos, 2*rmax + 1).data
    # pos_cut = cut.
    # mask_c, dists = circular_mask(img.shape, pos, radius=rmax)
    # mask = (mask_c) if mask is None else (mask_c | mask)
    # mask = Cutout2D(mask, pos, 2*rmax + 1).data
    # dists = Cutout2D(dists, pos, 2*rmax + 1).data
    # cut = img[mask]

    if sort_dist:
        sort_idx = np.argsort(dists[~mask])
        return dists[~mask][sort_idx], cut[~mask][sort_idx]

    if fitfunc is not None:
        from scipy.optimize import curve_fit

        _r = dists[~mask]
        _i = cut[~mask]
        _imin, _imax = _i.min(), _i.max()
        if fitfunc == "gauss":
            fitter = gauss_r
            p0 = [_i[_r == _r.min()][0] - _imin, max(1, rmax / 6), _imin]
            bounds = np.array([[0, 0, _imin - 1], [np.inf, rmax / 2, _imax + 1]])
            # assuming the user gave ~ (2-3)x FWHM, and we want sigma ~ 0.4xFWHM
        elif fitfunc == "moffat":
            fitter = moffat_r
            p0 = [_i[_r == _r.min()][0] - _imin, 1, 2.5, _imin]
            bounds = np.array(
                [[0, 0.1, 1.0, _imin - 1], [np.inf, rmax, np.inf, _imax + 1]]
            )
        elif fitfunc == "bivt":
            fitter = bivt_r
            p0 = [_i[_r == _r.min()][0] - _imin, 1, max(1, rmax / 6), _imin]
            bounds = np.array(
                [[0, -2, 1.0e-10, _imin - 1], [np.inf, 15, 2 * rmax, _imax + 1]]
            )
            # assuming the user gave ~ (2-3)x FWHM, and we want sigma ~ 0.4xFWHM
        else:
            raise ValueError("Unknown function: {}".format(fitfunc))

        popt, _ = curve_fit(fitter, _r, _i, p0=p0, bounds=bounds)
        if refit is not None:
            fitrad = max(refit * fwhm_r(popt, fun=fitfunc), 3)
            if fitrad < rmax:
                mask = _r > fitrad
                # upper bound of sky as the profile at fitrad
                bounds[1][-1] = fitter(fitrad, *popt)
                # tighten others (amp, sigma, etc)
                bounds[0][:-1] = popt[:-1] * 0.5
                bounds[1][:-1] = popt[:-1] * 2

                popt, _ = curve_fit(
                    fitter, _r[~mask], _i[~mask], p0=popt, bounds=bounds
                )

        return _r, _i, fitter, popt, fwhm_r(popt, fun=fitfunc)

    return dists[~mask], cut[~mask]

    # # --- 1. mask out only the central region for quick initial fitting
    # _small_mask = (_r < max(rmax/4, 5))
    # # assuming the user gave ~ 2x FWHM, and we want ~ 0.5xFWHM
    # _popt, _ = curve_fit(fitter, _r[_small_mask], _i[_small_mask], p0=p0)
    # # --- 2. Use that as the initial input for the full fitting
