from warnings import warn

import numpy as np
from astropy import units as u
from astropy.nddata import CCDData
from astropy.table import QTable
from photutils import aperture_photometry

from .background import sky_fit

__all__ = ["apphot_annulus"]


# TODO: Put centroiding into this apphot_annulus ?
# TODO: use variance instead of error (see photutils 0.7)
# TODO: one_aperture_per_row : bool, optional.
# `photutils.aperture_photometry` produces 1-row result if multiple radii aperture is given with column
# names starting from ``aperture_sum_0`` and ``aperture_sum_err_0``.
def apphot_annulus(ccd, aperture, annulus=None, t_exposure=None, exposure_key="EXPTIME", error=None,
                   mask=None, sky_keys={}, aparea_exact=False, verbose=False, pandas=False, **kwargs):
    ''' Do aperture photometry using annulus.

    Parameters
    ----------
    ccd : CCDData
        The data to be photometried. Preferably in ADU.

    aperture, annulus : aperture and annulus object or list of such.
        The aperture and annulus to be used for aperture photometry. For multi-position aperture, just
        use, e.g., ``CircularAperture(positions, r=10)``. For multiple radii, use, e.g.,
        ``[CircularAperture(positions, r=r) for r in radii]``.

    exposure_key : str
        The key for exposure time. Together with ``t_exposure_unit``, the function will normalize the
        signal to exposure time. If ``t_exposure`` is not None, this will be ignored.

    error : array-like or Quantity, optional
        See ``photutils.aperture_photometry`` documentation. The pixel-wise error map to be propagated
        to magnitued error.

    sky_keys : dict
        kwargs of ``sky_fit``. Mostly one doesn't change the default setting, so I intentionally made
        it to be dict rather than usual kwargs, etc.

    aparea_exact : bool, optional
        Whether to calculate the aperture area (``'aparea'`` column) exactly or not. If ``True``, the
        area outside the image  **and** those specified by mask are not counted. Default is ``False``.

    pandas : bool, optional.
        Whether to convert to ``pandas.DataFrame``.

    **kwargs :
        kwargs for ``photutils.aperture_photometry``.

    Returns
    -------
    phot_f: astropy.table.Table
        The photometry result.
    '''
    _ccd = ccd.copy()

    if isinstance(_ccd, CCDData):
        _arr = _ccd.data
        _mask = _ccd.mask
        if t_exposure is None:
            try:
                t_exposure = _ccd.header[exposure_key]
            except (KeyError, IndexError):
                t_exposure = 1
                warn(f"The exposure time info not given and not found from the header({exposure_key})."
                     + " Setting it to 1 sec.")
    else:  # ndarray
        _arr = np.array(_ccd)
        _mask = None
        if t_exposure is None:
            t_exposure = 1
            warn("The exposure time info not given. Setting it to 1 sec.")

    if _mask is None:
        _mask = np.zeros_like(_arr).astype(bool)

    if mask is not None:
        _mask |= mask

    if error is not None:
        if verbose:
            print("Ignore any uncertainty extension in the original CCD and use provided error.")
        err = error.copy()
        if isinstance(err, CCDData):
            err = err.data
    else:
        try:
            err = _ccd.uncertainty.array
        except AttributeError:
            if verbose:
                warn("Couldn't find Uncertainty extension in ccd. Will not calculate errors.")
            err = np.zeros_like(_arr)

    if aparea_exact:
        _ones = np.ones_like(_arr)
        _area = aperture_photometry(_ones, aperture, mask=_mask, **kwargs)
        n_ap = []
        for c in _area.colnames:
            if c.startswith("aperture_sum"):
                n_ap.append(_area[c][0])
        n_ap = np.array(n_ap)
    else:
        try:
            n_ap = aperture.area
        except TypeError:  # prior to photutils 0.7
            n_ap = aperture.area()
        except AttributeError:  # if array of aperture given
            try:
                n_ap = np.array([ap.area for ap in aperture])
            except TypeError:  # prior to photutils 0.7
                n_ap = np.array([ap.area() for ap in aperture])

    _phot = aperture_photometry(_arr, aperture, mask=_mask, error=err, **kwargs)
    # If we use ``_ccd``, photutils deal with the unit, and the lines below will give a lot of headache
    # for units. It's not easy since aperture can be pixel units or angular units (Sky apertures).
    # ysBach 2018-07-26

    if annulus is not None:
        skys = sky_fit(_arr, annulus, mask=_mask, **sky_keys)
        for c in skys.colnames:
            _phot[c] = [skys[c][0]]
    else:
        _phot['msky'] = 0
        _phot['nsky'] = 1
        _phot['nrej'] = 0
        _phot['ssky'] = 0

    if isinstance(aperture, (list, tuple, np.ndarray)):
        # If multiple apertures at each position
        # Convert aperture_sum_xx columns into 1-column...
        n = len(aperture)
        apsums = []
        aperrs = []
        phot = QTable(meta=_phot.meta)

        for i, c in enumerate(_phot.colnames):
            if not c.startswith("aperture"):  # all other columns
                phot[c] = [_phot[c][0]]*n
            elif c.startswith("aperture_sum_err"):  # aperture_sum_err_xx
                aperrs.append(_phot[c][0])
            else:  # aperture_sum_xx
                apsums.append(_phot[c][0])

        phot["aperture_sum"] = apsums
        if aperrs:
            phot["aperture_sum_err"] = aperrs
        # I guess we should not have this..? :
        # else:
        #     phot = _phot

    else:
        phot = _phot

    phot['aparea'] = n_ap
    phot["source_sum"] = phot["aperture_sum"] - n_ap * phot["msky"]

    # see, e.g., http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?radprof.hlp
    # Poisson + RDnoise (Poisson includes signal + sky + dark) :
    var_errmap = phot["aperture_sum_err"]**2
    # Sum of n_ap Gaussians (kind of random walk):
    var_skyrand = n_ap * phot["ssky"]**2
    # The CLT error (although not correct, let me denote it as "systematic" error for simplicity) of
    # the mean estimation is ssky/sqrt(nsky), and that is propagated for n_ap pixels, so we have std =
    # n_ap*ssky/sqrt(nsky), so variance is:
    var_sky = (n_ap * phot['ssky'])**2 / phot['nsky']

    phot["source_sum_err"] = np.sqrt(var_errmap + var_skyrand + var_sky)

    phot["mag"] = -2.5*np.log10(phot['source_sum']/t_exposure)
    phot["merr"] = (2.5/np.log(10)*phot["source_sum_err"]/phot['source_sum'])

    if pandas:
        return phot.to_pandas()
    else:
        return phot


#TODO: make this...
def apphot_ellip_sep(ccd, x, y, a, a_in, a_out, bpa=1, theta=0, t_exposure=None,
                     exposure_key="EXPTIME", error=None, mask=None, sky_keys={}, aparea_exact=False,
                     t_exposure_unit=u.s, verbose=False, pandas=False, **kwargs):
    ''' Similar to apphot_annulus but use sep to speedup.
    bpa : float
        b per a (ellipticity)
    '''
    try:
        import sep
    except ImportError:
        raise ImportError("sep is required for apphot_annulus_sep")

    _ccd = ccd.copy()

    if isinstance(_ccd, CCDData):
        _arr = _ccd.data
        _mask = _ccd.mask
        if t_exposure is None:
            try:
                t_exposure = _ccd.header[exposure_key]
            except (KeyError, IndexError):
                t_exposure = 1
                warn("The exposure time info not given and not found from the"
                     + f"header({exposure_key}). Setting it to 1 sec.")
    else:  # ndarray
        _arr = np.array(_ccd)
        _mask = None
        if t_exposure is None:
            t_exposure = 1
            warn("The exposure time info not given. Setting it to 1 sec.")

    if _mask is None:
        _mask = np.zeros_like(_arr).astype(bool)

    if mask is not None:
        _mask |= mask

    if error is not None:
        if verbose:
            print("Ignore any uncertainty extension in the original CCD and use provided error.")
        err = error.copy()
        if isinstance(err, CCDData):
            err = err.data
    else:
        try:
            err = _ccd.uncertainty.array
        except AttributeError:
            if verbose:
                warn("Couldn't find Uncertainty extension in ccd. "
                     + "Will not calculate errors.")
            err = np.zeros_like(_arr)

    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    multipos = False

    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    elif x.size > 1:
        multipos = True

    a = np.atleast_1d(a)
    bpa = np.atleast_1d(bpa)
    theta = np.atleast_1d(theta)
    if (a.size > 1) + (bpa.size > 1) + (theta.size > 1) > 1:
        raise ValueError("Only one of a, bpa, theta can have size > 1.")

    num_apertures = max(a.size, bpa.size, theta.size)
    a = np.repeat(a, num_apertures)
    bpa = np.repeat(bpa, num_apertures)
    theta = np.repeat(theta, num_apertures)
    b = a*bpa

    a_in = np.atleast_1d(a_in)
    a_out = np.atleast_1d(a_out)
    if a_in.size > 1 or a_out.size > 1 or bpa.size > 1:
        raise ValueError("multiple annuli not allowed yet.")

    pass
