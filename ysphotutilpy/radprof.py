""" Radial profiles
"""
import numpy as np
from astropy.nddata import CCDData, Cutout2D
from scipy.ndimage import center_of_mass
from photutils.aperture import CircularAperture, CircularAnnulus
from .background import sky_fit
from .center import circular_bbox_cut

__all__ = [
    "moffat_r", "gauss_r", "bivt_r", "radprof_pix",
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
    return amp*np.exp(-0.5*(r/sig)**2) + const


def moffat_r(r, amp=1, core=1, power=2.5, const=0):
    return amp*(1 + (r/core)**2)**(-power) + const


def bivt_r(r, amp=1, num2=1, sig=1, const=0):
    """ Bivariate t distribution (generalized Moffat)
    num2 : float
        The degrees of freedom (nu) minus 2.
    """
    return amp*(1 + (r**2/(num2*sig**2)))**(-(num2+4)/2) + const


def fwhm_r(popt, fun):
    if fun == "gauss":
        return 2*np.sqrt(2*np.log(2))*popt[1]
    elif fun == "moffat":
        return 2*popt[1]*np.sqrt(2**(1/popt[2]) - 1)
    elif fun == "bivt":
        # 2*sig*np.sqrt((nu-2)*(2**(2/(nu+2)) - 1))
        return 2*popt[2]*np.sqrt(popt[1]*(2**(2/(popt[1]+4)) - 1))
    else:
        raise ValueError("Unknown function: {}".format(fun))


def radprof_pix(img, pos, mask=None, rmax=10, sort_dist=False, fitfunc=None,
                refit=1):
    """Get radial profile (pixel values) of an object from n-D image.

    Parameters
    ----------
    img : CCDData or ndarray
        The image to be profiled.
    pos : array_like
        The xy coordinates of the center of the object (0-indexing).
    rmax : int, optional
        The maximum radius to be profiled.
    refit : float, None, optional
        If not None, refit the profile for pixels within max(refit*FWHM, 3). It this exceeds rmax, refit is ignored.

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
        raise TypeError(f'img must be a CCDData or ndarray (now {type(img) = })')

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
            p0 = [_i[_r == _r.min()][0] - _imin, max(1, rmax/6), _imin]
            bounds = np.array([[0, 0, _imin-1], [np.inf, rmax/2, _imax+1]])
            # assuming the user gave ~ (2-3)x FWHM, and we want sigma ~ 0.4xFWHM
        elif fitfunc == "moffat":
            fitter = moffat_r
            p0 = [_i[_r == _r.min()][0] - _imin, 1, 2.5, _imin]
            bounds = np.array([[0, 0.1, 1., _imin-1], [np.inf, rmax, np.inf, _imax+1]])
        elif fitfunc == "bivt":
            fitter = bivt_r
            p0 = [_i[_r == _r.min()][0] - _imin, 1, max(1, rmax/6), _imin]
            bounds = np.array([[0, -2, 1.e-10, _imin-1], [np.inf, 15, 2*rmax, _imax+1]])
            # assuming the user gave ~ (2-3)x FWHM, and we want sigma ~ 0.4xFWHM
        else:
            raise ValueError("Unknown function: {}".format(fitfunc))

        popt, _ = curve_fit(fitter, _r, _i, p0=p0, bounds=bounds)
        if refit is not None:
            fitrad = max(refit*fwhm_r(popt, fun=fitfunc), 3)
            if fitrad < rmax:
                mask = (_r > fitrad)
                # upper bound of sky as the profile at fitrad
                bounds[1][-1] = fitter(fitrad, *popt)
                # tighten others (amp, sigma, etc)
                bounds[0][:-1] = popt[:-1]*0.5
                bounds[1][:-1] = popt[:-1]*2

                popt, _ = curve_fit(fitter, _r[~mask], _i[~mask], p0=popt, bounds=bounds)

        return _r, _i, fitter, popt, fwhm_r(popt, fun=fitfunc)

    return dists[~mask], cut[~mask]

    # # --- 1. mask out only the central region for quick initial fitting
    # _small_mask = (_r < max(rmax/4, 5))
    # # assuming the user gave ~ 2x FWHM, and we want ~ 0.5xFWHM
    # _popt, _ = curve_fit(fitter, _r[_small_mask], _i[_small_mask], p0=p0)
    # # --- 2. Use that as the initial input for the full fitting
