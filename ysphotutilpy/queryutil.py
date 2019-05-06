import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u

__all__ = ["group_stars", "get_xy", "xyinFOV"]


def group_stars(table, crit_separation, xcol="x", ycol="y"):
    ''' Group stars using DAOGROUP algorithm and return row indices.
    Parameters
    ----------
    table: astropy.table.Table
        The queried result table.
    crit_separation: float or int
        Distance, in units of pixels, such that any two stars separated by
        less than this distance will be placed in the same group.
    xcol, ycol: str, optional
        The column names for x and y positions. This is necessary since
        ``photutils.DAOGroup`` accepts a table which has x y positions
        designated as ``"x_0"`` and ``"y_0"``.

    Notes
    -----
    Assuming the psf fwhm to be known, ``crit_separation`` may be set to
    ``k * fwhm``, for some positive real k.

    See Also
    --------
    photutils.DAOStarFinder

    References
    ----------
    [1] Stetson, Astronomical Society of the Pacific, Publications,
        (ISSN 0004-6280), vol. 99, March 1987, p. 191-222.
        Available at: http://adsabs.harvard.edu/abs/1987PASP...99..191S

    Return
    ------
    grouped_rows: list
        The indices of the rows which are "grouped" stars. You may remove
        such rows using ``table.remove_rows(grouped_rows)``.
    '''
    from photutils.psf.groupstars import DAOGroup
    tab = table.copy()

    tab[xcol].name = "x_0"
    tab[ycol].name = "y_0"
    gtab = DAOGroup(crit_separation=crit_separation)(tab).group_by("group_id")
    gid, gnum = np.unique(gtab["group_id"], return_counts=True)
    gmask = gid[gnum != 1]  # group id with > 1 stars
    grouped_rows = []
    for i, gid in enumerate(gtab["group_id"]):
        if gid in gmask:
            grouped_rows.append(i)
    return grouped_rows


def get_xy(header, ra, dec, unit=u.deg, origin=0, mode='all'):
    ''' Get image XY from the header WCS
    Parameters
    ----------
    header: astropy.io.fits.Header or pandas.DataFrame
        The header to extract WCS information.
    ra, dec: float or Quantity or array-like of such
        The coordinates to get XY position. If Quantity, ``unit`` will likely
        be ignored.
    unit: astropy Quantity
        Unit of the ``ra`` and ``dec`` given.
    origin: int, optional
       Whether to return 0 or 1-based pixel coordinates.
    mode: 'all' or 'wcs', optional
        Whether to do the transformation including distortions (``'all'``) or
        only including only the core WCS transformation (``'wcs'``).
    '''
    w = WCS(header)
    coo = SkyCoord(ra, dec, unit=unit)
    xy = SkyCoord.to_pixel(coo, wcs=w, origin=origin, mode=mode)
    return xy


def xyinFOV(header, table, ra_key='ra', dec_key='dec', bezel=0, origin=0,
            mode='all'):
    ''' Convert RA/DEC to pixel with rejection at bezels
    Parameters
    ----------
    header: astropy.io.fits.Header or pandas.DataFrame
        The header to extract WCS information.
    table: astropy.table.Table
        The queried result table.
    ra_key, dec_key: str, optional
        The column names containing RA/DEC.
    bezel: int or float, optional
        The bezel size to exclude stars at the image edges. If you want to
        keep some stars outside the edges, put negative values (e.g., ``-5``).
    origin: int, optional
       Whether to return 0 or 1-based pixel coordinates.
    mode: 'all' or 'wcs', optional
        Whether to do the transformation including distortions (``'all'``) or
        only including only the core WCS transformation (``'wcs'``).
    '''
    _tab = table.copy()
    if isinstance(table, pd.DataFrame):
        _tab = Table.from_pandas(table)
    elif not isinstance(table, Table):
        raise TypeError("table must be either astropy Table or pandas DataFrame.")

    w = WCS(header)
    coo = SkyCoord(_tab[ra_key], _tab[dec_key])
    x, y = coo.to_pixel(wcs=w, origin=0, mode=mode)

    nx, ny = header['naxis1'], header['naxis2']
    mask = ((x < (0 + bezel))
            | (x > (nx - bezel))
            | (y < (0 + bezel))
            | (y > (ny - bezel)))
    x = x[~mask]
    y = y[~mask]
    _tab.remove_rows(mask)

    _tab["x"] = x
    _tab["y"] = y

    return _tab


# def sdss2BV(g, r, gerr=None, rerr=None):
#     '''
#     Pan-STARRS DR1 (PS1) uses AB mag.
#     https://www.sdss.org/dr12/algorithms/fluxcal/#SDSStoAB
#     Jester et al. (2005) and Lupton (2005):
#     https://www.sdss.org/dr12/algorithms/sdssubvritransform/
#     Here I used Lupton. Application to PS1, it seems like Jester - Lupton VS
#     Lupton V mag is scattered around -0.013 +- 0.003 (minmax = -0.025, -0.005)
#     --> Lupton conversion is fainter.
#     V = g - 0.5784*(g - r) - 0.0038;  sigma = 0.0054
#     '''
#     if gerr is None:
#         gerr = np.zeros_like(g)

#     if rerr is None:
#         rerr = np.zeros_like(r)

#     V = g - 0.5784 * (g - r) - 0.0038
#     dV = np.sqrt((1.5784 * gerr)**2 + (0.5784 * rerr)**2 + 0.0052**2)
#     return V, dV
