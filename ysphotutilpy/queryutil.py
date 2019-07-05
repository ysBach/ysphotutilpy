import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astropy.wcs import WCS
from astroquery.jplhorizons import Horizons
from astroquery.vizier import Vizier

__all__ = ["HorizonsDiscreteEpochsQuery",
           "panstarrs_query", "group_stars", "get_xy", "xyinFOV"]


class HorizonsDiscreteEpochsQuery:
    def __init__(self, targetname, location, epochs, id_type="smallbody"):
        '''
        Parameters
        ----------
        id : str
            Name, number, or designation of the object to be queried.
        location : str or dict
            Observer's location for ephemerides queries or center body
            name for orbital element or vector queries. Uses the same
            codes as JPL Horizons. If no location is provided, Earth's
            center is used for ephemerides queries and the Sun's center
            for elements and vectors queries. Arbitrary topocentic
            coordinates for ephemerides queries can be provided in the
            format of a dictionary. The dictionary has to be of the form
            {``'lon'``: longitude in deg (East positive, West negative),
            ``'lat'``: latitude in deg (North positive, South negative),
            ``'elevation'``: elevation in km above the reference
            ellipsoid, [``'body'``: Horizons body ID of the central
            body; optional; if this value is not provided it is assumed
            that this location is on Earth]}.
        epochs : scalar, list-like, or dictionary
            Either a list of epochs in JD or MJD format or a dictionary
            defining a range of times and dates; the range dictionary
            has to be of the form {``'start'``:'YYYY-MM-DD [HH:MM:SS]',
            ``'stop'``:'YYYY-MM-DD [HH:MM:SS]',
            ``'step'``:'n[y|d|m|s]'}. If no epochs are provided, the
            current time is used.
        id_type : str, optional
            Identifier type, options:
            ``'smallbody'``, ``'majorbody'`` (planets but also anything
            that is not a small body), ``'designation'``, ``'name'``,
            ``'asteroid_name'``, ``'comet_name'``, ``'id'`` (Horizons id
            number), or ``'smallbody'`` (find the closest match under
            any id_type), default: ``'smallbody'``
        '''
        self.targetname = str(targetname)
        self.location = location
        self.epochs = np.asarray(epochs)
        self.id_type = id_type
        self.query_table = None
        self.uri = []

    def __str__(self):
        _str = "Query {:s} at location {} for given discrete epochs."
        return _str.format(self.targetname, self.location)

    def query(self, depoch=100, *args, **kwargs):
        '''
        Parameters
        ----------
        depoch : int, optional
            The number of discrete epochs to be chopped.

        args, kwargs : optional.
            Passed to ``.ephemerides()`` of ``Horizons``.
        '''
        Nepoch = np.shape(self.epochs)[0]
        Nquery = (Nepoch - 1) // depoch + 1
        tabs = []

        print(f'Query: {self.targetname} '
              + f'at {self.location} for {Nepoch} epochs''')

        if Nquery > 1:
            print(f"Query chopped into {Nquery} chunks: Doing ",
                  end=' ')

        for i in range(Nquery):
            print(f"{i+1}...", end=' ')
            i_0 = i*depoch
            i_1 = (i + 1)*depoch
            epochs_i = self.epochs[i_0:i_1]

            obj = Horizons(id=self.targetname,      #
                           location=self.location,  #
                           epochs=epochs_i,         #
                           id_type=self.id_type)
            eph = obj.ephemerides(*args, **kwargs)

            tabs.append(eph)
            self.uri.append(obj.uri)

        if len(tabs) == 1:
            self.query_table = tabs[0]

        elif len(tabs) > 1:
            self.query_table = vstack(tabs)

        print("Query done.")

        return self.query_table


def panstarrs_query(ra_deg, dec_deg, radius=None, inner_radius=None,
                    width=None, height=None, columns=None, column_filters={},
                    maxsources=10000):
    """ Query PanSTARRS @ VizieR using astroquery.vizier
    Got ideas from Michael Mommert:
    https://michaelmommert.wordpress.com/2017/02/13/accessing-the-gaia-and-pan-starrs-catalogs-using-python/

    Parameters
    ----------
    ra_deg, dec_deg, rad_deg : float
        The central RA, DEC and the cone search radius in degrees unit.

    radius : convertible to `~astropy.coordinates.Angle`
        The radius of the circular region to query.

    inner_radius : convertible to `~astropy.coordinates.Angle`
        When set in addition to ``radius``, the queried region becomes
        annular, with outer radius ``radius`` and inner radius
        ``inner_radius``.

    width : convertible to `~astropy.coordinates.Angle`
        The width of the square region to query.

    height : convertible to `~astropy.coordinates.Angle`
        When set in addition to ``width``, the queried region becomes
        rectangular, with the specified ``width`` and ``height``.

    columns : list of str, str in ['*', 'all', 'full'], None, optional
        The columns to be retrieved. Any str of ``['*', 'all', 'full']``
        will return full list of columns.

    column_filters : dict, optional
        The column filters for astroquery.vizier.
        Example can be ``{"gmag":"13.0..20.0", "e_gmag":"<0.10"}``.

    Return
    ------
    queried : astropy.table object
        The queried result.

    Note
    ----
    All columns: http://vizier.u-strasbg.fr/viz-bin/VizieR?-source=II/349
    """
    if columns is None:
        columns = ['objID', 'RAJ2000', 'DEJ2000', 'e_RAJ2000', 'e_DEJ2000',
                   'Ng', 'gmag', 'e_gmag', 'gmagStd',
                   'Nr', 'rmag', 'e_rmag', 'rmagStd',
                   'Ni', 'imag', 'e_imag', 'imagStd',
                   'Nz', 'zmag', 'e_zmag', 'zmagStd',
                   'Ny', 'ymag', 'e_ymag', 'ymagStd',
                   ]
    elif isinstance(columns, str):
        if columns in ['*', 'all', 'full']:
            columns = "all"
        else:
            raise ValueError("If columns is str, it must be one of "
                             + "['*', 'all', 'full']")

    vquery = Vizier(columns=columns,
                    column_filters=column_filters,
                    row_limit=-1)

    field = SkyCoord(ra=ra_deg, dec=dec_deg,
                     unit=(u.deg, u.deg),
                     frame='icrs')

    queried = vquery.query_region(field,
                                  radius=radius,
                                  inner_radius=inner_radius,
                                  width=width,
                                  height=height,
                                  catalog="II/349/ps1")[0]
    return queried


def group_stars(table, crit_separation, xcol="x", ycol="y", index_only=True):
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
    index : bool, optional
        Whether to return only the index of the grouped rows (group
        information is lost) or the full grouped table (after group_by).

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
    gtab: Table
        Returned when ``index_only=False``.
        The table underwent ``.group_by("group_id")``.

    grouped_rows: list
        Returned when ``index_only=True``.
        The indices of the rows which are "grouped" stars. You may remove
        such rows using ``table.remove_rows(grouped_rows)``.
    '''
    from photutils.psf.groupstars import DAOGroup
    tab = table.copy()

    tab[xcol].name = "x_0"
    tab[ycol].name = "y_0"
    gtab = DAOGroup(crit_separation=crit_separation)(tab).group_by("group_id")
    if not index_only:
        return gtab
    else:
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
        raise TypeError(
            "table must be either astropy Table or pandas DataFrame.")

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


"""
def sdss2BV(g, r, gerr=None, rerr=None):
    '''
    Pan-STARRS DR1 (PS1) uses AB mag.
    https://www.sdss.org/dr12/algorithms/fluxcal/#SDSStoAB
    Jester et al. (2005) and Lupton (2005):
    https://www.sdss.org/dr12/algorithms/sdssubvritransform/
    Here I used Lupton. Application to PS1, it seems like Jester - Lupton VS
    Lupton V mag is scattered around -0.013 +- 0.003 (minmax = -0.025, -0.005)
    --> Lupton conversion is fainter.
    V = g - 0.5784*(g - r) - 0.0038;  sigma = 0.0054
    '''
    if gerr is None:
        gerr = np.zeros_like(g)

    if rerr is None:
        rerr = np.zeros_like(r)

    V = g - 0.5784 * (g - r) - 0.0038
    dV = np.sqrt((1.5784 * gerr)**2 + (0.5784 * rerr)**2 + 0.0052**2)
    return V, dV
"""
