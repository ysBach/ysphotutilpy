import numpy as np
from astropy.units import Quantity, UnitsError
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling import Fittable2DModel, Parameter
from astropy.nddata import Cutout2D
from astropy.table import Table, vstack
from photutils.psf.groupstars import DAOGroup

__all__ = ["dao_nstar_clamp", "dao_weight_map", "dao_nstar",
           "daophot_concat", "IntegratedGaussianPRF", "IntegratedGaussian2D"]


def dao_nstar_clamp(p_old, p_new_raw, p_clamp):
    ''' The "clamp" for NSTAR routine
    Notes
    -----
    StetsonPB 1987, PASP, 99, 191, p.208
    '''
    dp_raw = p_new_raw - p_old
    p_new = p_old + dp_raw / (1 + np.abs(dp_raw) / p_clamp)
    return p_new


def dao_weight_map(data, position, r_fit):
    ''' The weight for centering routine
    Notes
    -----
    StetsonPB 1987, PASP, 99, 191, p.207
    https://iraf.net/irafhelp.php?val=daopars&help=Help+Page
    https://iraf.readthedocs.io/en/latest/tasks/noao/digiphot/daophot/daopars.html
    '''
    x0, y0 = position
    is_cut = False
    if np.any(np.array(data.shape) > (2 * r_fit + 1)):  # To save CPU
        is_cut = True
        cut = Cutout2D(data=data, position=(x0, y0),
                       size=(2 * r_fit + 1), mode='partial')
        data = cut.data
        x0, y0 = cut.to_cutout_position((x0, y0))

    nx, ny = data.shape[1], data.shape[0]
    yy_data, xx_data = np.mgrid[:ny, :nx]

    # add 1.e-6 to avoid zero division
    distance_map = np.sqrt((xx_data - x0)**2 + (yy_data - y0)**2) + 1.e-6
    dist = np.ma.array(data=distance_map, mask=(distance_map > r_fit))
    rsq = dist**2 / r_fit**2
    weight_map = 5.0 / (5.0 + rsq / (1.0 - rsq))
    return weight_map


def dao_nstar(data, psf, position=None, r_fit=2, flux_init=1, sky=0, err=None,
              fitter=LevMarLSQFitter(), full=True):
    '''
    psf: photutils.psf.FittableImageModel
    '''
    if position is None:
        position = ((data.shape[1] - 1) / 2, (data.shape[0] - 1) / 2)

    if err is None:
        err = np.zeros_like(data)

    psf_init = psf.copy()
    psf_init.flux = flux_init

    fbox = 2 * r_fit + 1  # fitting box size
    fcut = Cutout2D(data, position=position, size=fbox,
                    mode='partial')  # "fitting" cut
    fcut_err = Cutout2D(err, position=position, size=fbox, mode='partial').data
    fcut_skysub = fcut.data - sky  # Must be sky subtracted before PSF fitting
    pos_fcut_init = fcut.to_cutout_position(position)  # Order of x, y
    psf_init.x_0, psf_init.y_0 = fcut.center_cutout

    dao_weight = dao_weight_map(fcut_skysub, pos_fcut_init, r_fit)
    # astropy gets ``weight`` = 1 / sigma.. strange..
    astropy_weight = np.sqrt(dao_weight.data) / fcut_err
    astropy_weight[dao_weight.mask] = 0

    yy_fit, xx_fit = np.mgrid[:fcut_skysub.shape[1], :fcut_skysub.shape[0]]
    fit = fitter(psf_init, xx_fit, yy_fit, fcut_skysub, weights=dao_weight)
    pos_fit = fcut.to_original_position((fit.x_0, fit.y_0))
    fit.x_0, fit.y_0 = pos_fit

    if full:
        return (fit, pos_fit, fitter,
                astropy_weight, fcut, fcut_skysub, fcut_err)

    else:
        return fit, pos_fit, fitter


def dao_substar(data, position, fitted_psf, size):
    pass


def daophot_concat(filelist, crit_separation, xcol="x", ycol="y",
                   table_reader=Table.read, reader_kwargs={}):
    ''' Concatenates the DAOPHOT-like results
    filelist : list of path-like
        The list of file paths to be concatenated.
    '''

    tablist = []
    for fpath in filelist:
        tab = table_reader(fpath, **reader_kwargs)
        tablist.append(tab)
    tabs = vstack(tablist)
    if "group_id" in tabs.colnames:
        tabs.remove_column("group_id")
    tabs["id"] = np.arange(len(tabs)) + 1

    tabs[xcol].name = "x_0"
    tabs[ycol].name = "y_0"
    tabs_g = DAOGroup(crit_separation=crit_separation)(tabs)
    tabs_g["x_0"].name = xcol
    tabs_g["y_0"].name = ycol
    return tabs_g


class IntegratedGaussianPRF(Fittable2DModel):
    r"""
    Circular Gaussian model integrated over pixels.

    Because it is integrated, this model is considered a PRF, *not* a PSF (see
    :ref:`psf-terminology` for more about the terminology used here.)

    This model is a Gaussian *integrated* over an area of ``1`` (in units of
    the model input coordinates, e.g., 1 pixel). This is in contrast to the
    apparently similar `astropy.modeling.functional_models.Gaussian2D`, which
    is the value of a 2D Gaussian *at* the input coordinates, with no
    integration. So this model is equivalent to assuming the PSF is Gaussian at
    a *sub-pixel* level.

    Parameters
    ----------
    sigma : float
        Width of the Gaussian PSF.
    flux : float, optional
        Total integrated flux over the entire PSF
    x_0 : float, optional
        Position of the peak in x direction.
    y_0 : float, optional
        Position of the peak in y direction.

    Notes
    -----
    This model is evaluated according to the following formula:

        .. math::

            f(x, y) =
                \frac{F}{4}
                \left[
                {\rm erf} \left(\frac{x - x_0 + 0.5}
                {\sqrt{2} \sigma} \right) -
                {\rm erf} \left(\frac{x - x_0 - 0.5}
                {\sqrt{2} \sigma} \right)
                \right]
                \left[
                {\rm erf} \left(\frac{y - y_0 + 0.5}
                {\sqrt{2} \sigma} \right) -
                {\rm erf} \left(\frac{y - y_0 - 0.5}
                {\sqrt{2} \sigma} \right)
                \right]

    where ``erf`` denotes the error function and ``F`` the total integrated
    flux.

    The only difference from photutils.psf.IntegratedGaussianPRF is that the
    bounding box is 7-sigma, following DAOPHOT ver 4 mathsubs.f/DAOERF
    function.
    """

    flux = Parameter(default=1)
    x_0 = Parameter(default=0)
    y_0 = Parameter(default=0)
    sigma = Parameter(default=1, fixed=True)

    _erf = None

    @property
    def bounding_box(self):
        halfwidth = 7 * self.sigma
        return ((int(self.y_0 - halfwidth), int(self.y_0 + halfwidth)),
                (int(self.x_0 - halfwidth), int(self.x_0 + halfwidth)))

    def __init__(self, sigma=sigma.default,
                 x_0=x_0.default, y_0=y_0.default, flux=flux.default,
                 **kwargs):
        if self._erf is None:
            from scipy.special import erf
            self.__class__._erf = erf

        super().__init__(n_models=1, sigma=sigma, x_0=x_0, y_0=y_0, flux=flux, **kwargs)

    def evaluate(self, x, y, flux, x_0, y_0, sigma):
        """Model function Gaussian PSF model."""
        sqrt2sig = np.sqrt(2) * sigma
        return (flux / 4
                * ((self._erf((x - x_0 + 0.5) / sqrt2sig)
                    - self._erf((x - x_0 - 0.5) / sqrt2sig))
                   * (self._erf((y - y_0 + 0.5) / sqrt2sig)
                      - self._erf((y - y_0 - 0.5) / sqrt2sig))))



TWOPI = 2 * np.pi
FLOAT_EPSILON = float(np.finfo(np.float32).tiny)

# Note that we define this here rather than using the value defined in
# astropy.stats to avoid importing astropy.stats every time astropy.modeling
# is loaded.
GAUSSIAN_SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))
from scipy.special import erf

class IntegratedGaussian2D(Fittable2DModel):
    r"""
    Integrated Two dimensional Gaussian model. Only circular Gaussian has analytic form (using erf) for this.
    Parameters
    ----------
    amplitude : float or `~astropy.units.Quantity`.
        Amplitude (peak value) of the Gaussian.
    x_mean : float or `~astropy.units.Quantity`.
        Mean of the Gaussian in x.
    y_mean : float or `~astropy.units.Quantity`.
        Mean of the Gaussian in y.
    x_stddev : float or `~astropy.units.Quantity` or None.
        Standard deviation of the Gaussian in x before rotating by theta. Must
        be None if a covariance matrix (``cov_matrix``) is provided. If no
        ``cov_matrix`` is given, ``None`` means the default value (1).
    y_stddev : float or `~astropy.units.Quantity` or None.
        Standard deviation of the Gaussian in y before rotating by theta. Must
        be None if a covariance matrix (``cov_matrix``) is provided. If no
        ``cov_matrix`` is given, ``None`` means the default value (1).
    theta : float or `~astropy.units.Quantity`, optional.
        The rotation angle as an angular quantity
        (`~astropy.units.Quantity` or `~astropy.coordinates.Angle`)
        or a value in radians (as a float). The rotation angle
        increases counterclockwise. Must be `None` if a covariance matrix
        (``cov_matrix``) is provided. If no ``cov_matrix`` is given,
        `None` means the default value (0).
    cov_matrix : ndarray, optional
        A 2x2 covariance matrix. If specified, overrides the ``x_stddev``,
        ``y_stddev``, and ``theta`` defaults.
    Notes
    -----
    Either all or none of input ``x, y``, ``[x,y]_mean`` and ``[x,y]_stddev``
    must be provided consistently with compatible units or as unitless numbers.
    Model formula:
        .. math::
            f(x, y) = A e^{-a\left(x - x_{0}\right)^{2}  -b\left(x - x_{0}\right)
            \left(y - y_{0}\right)  -c\left(y - y_{0}\right)^{2}}
    Using the following definitions:
        .. math::
            a = \left(\frac{\cos^{2}{\left (\theta \right )}}{2 \sigma_{x}^{2}} +
            \frac{\sin^{2}{\left (\theta \right )}}{2 \sigma_{y}^{2}}\right)
            b = \left(\frac{\sin{\left (2 \theta \right )}}{2 \sigma_{x}^{2}} -
            \frac{\sin{\left (2 \theta \right )}}{2 \sigma_{y}^{2}}\right)
            c = \left(\frac{\sin^{2}{\left (\theta \right )}}{2 \sigma_{x}^{2}} +
            \frac{\cos^{2}{\left (\theta \right )}}{2 \sigma_{y}^{2}}\right)
    If using a ``cov_matrix``, the model is of the form:
        .. math::
            f(x, y) = A e^{-0.5 \left(
                    \vec{x} - \vec{x}_{0}\right)^{T} \Sigma^{-1} \left(\vec{x} - \vec{x}_{0}
                \right)}
    where :math:`\vec{x} = [x, y]`, :math:`\vec{x}_{0} = [x_{0}, y_{0}]`,
    and :math:`\Sigma` is the covariance matrix:
        .. math::
            \Sigma = \left(\begin{array}{ccc}
            \sigma_x^2               & \rho \sigma_x \sigma_y \\
            \rho \sigma_x \sigma_y   & \sigma_y^2
            \end{array}\right)
    :math:`\rho` is the correlation between ``x`` and ``y``, which should
    be between -1 and +1.  Positive correlation corresponds to a
    ``theta`` in the range 0 to 90 degrees.  Negative correlation
    corresponds to a ``theta`` in the range of 0 to -90 degrees.
    See [1]_ for more details about the 2D Gaussian function.
    See Also
    --------
    Gaussian1D, Box2D, Moffat2D
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Gaussian_function
    """

    flux = Parameter(default=1, description="Total flux of the integrated Gaussian")
    x_mean = Parameter(
        default=0, description="Peak position (along x axis) of Gaussian"
    )
    y_mean = Parameter(
        default=0, description="Peak position (along y axis) of Gaussian"
    )
    stddev = Parameter(
        default=1, description="Standard deviation of the Gaussian"
    )

    def __init__(
        self,
        flux=flux.default,
        x_mean=x_mean.default,
        y_mean=y_mean.default,
        stddev=None,
        **kwargs,
    ):
        if stddev is None:
            stddev = self.__class__.stddev.default

        # Ensure stddev makes sense if its bounds are not explicitly set.
        # stddev must be non-zero and positive.
        # TODO: Investigate why setting this in Parameter above causes
        #       convolution tests to hang.
        kwargs.setdefault("bounds", {})
        kwargs["bounds"].setdefault("stddev", (FLOAT_EPSILON, None))

        super().__init__(
            flux=flux,
            x_mean=x_mean,
            y_mean=y_mean,
            stddev=stddev,
            **kwargs,
        )

    @property
    def fwhm(self):
        """Gaussian full width at half maximum."""
        return self.stddev * GAUSSIAN_SIGMA_TO_FWHM

    def bounding_box(self, factor=5.5):
        """
        Tuple defining the default ``bounding_box`` limits in each dimension,
        ``((y_low, y_high), (x_low, x_high))``.
        The default offset from the mean is 5.5-sigma, corresponding
        to a relative error < 1e-7. The limits are adjusted for rotation.
        Parameters
        ----------
        factor : float, optional
            The multiple of `x_stddev` and `y_stddev` used to define the limits.
            The default is 5.5.
        Examples
        --------
        >>> from astropy.modeling.models import Gaussian2D
        >>> model = Gaussian2D(x_mean=0, y_mean=0, x_stddev=1, y_stddev=2)
        >>> model.bounding_box
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-5.5, upper=5.5)
                y: Interval(lower=-11.0, upper=11.0)
            }
            model=Gaussian2D(inputs=('x', 'y'))
            order='C'
        )
        This range can be set directly (see: `Model.bounding_box
        <astropy.modeling.Model.bounding_box>`) or by using a different factor
        like:
        >>> model.bounding_box = model.bounding_box(factor=2)
        >>> model.bounding_box
        ModelBoundingBox(
            intervals={
                x: Interval(lower=-2.0, upper=2.0)
                y: Interval(lower=-4.0, upper=4.0)
            }
            model=Gaussian2D(inputs=('x', 'y'))
            order='C'
        )
        """
        a = factor * self.stddev

        return (
            (self.y_mean - a, self.y_mean + a),
            (self.x_mean - a, self.x_mean + a),
        )

    @staticmethod
    def evaluate(x, y, flux, x_mean, y_mean, stddev):
        """Two dimensional Gaussian function."""
        sqrt2sig = np.sqrt(2) * stddev
        return (flux / 4
                * ((erf((x - x_mean + 0.5) / sqrt2sig)
                    - erf((x - x_mean - 0.5) / sqrt2sig))
                   * (erf((y - y_mean + 0.5) / sqrt2sig)
                      - erf((y - y_mean - 0.5) / sqrt2sig))))

    @staticmethod
    def fit_deriv(x, y, flux, x_mean, y_mean, stddev):
        """Two dimensional Gaussian function derivative with respect to parameters."""
        sqrt2sig = np.sqrt(2) * stddev
        twosigsq = 2 * stddev**2
        flux_2sqrtpi = flux / (2 * np.sqrt(np.pi))
        x1 = x - x_mean - 0.5
        x2 = x - x_mean + 0.5
        y1 = y - y_mean - 0.5
        y2 = y - y_mean + 0.5
        e_x1_2s_sq = np.exp(-x1**2 / twosigsq)
        e_x2_2s_sq = np.exp(-x2**2 / twosigsq)
        e_y1_2s_sq = np.exp(-y1**2 / twosigsq)
        e_y2_2s_sq = np.exp(-y2**2 / twosigsq)
        erfxx = erf(x2 / sqrt2sig) - erf(x1 / sqrt2sig)
        erfyy = erf(y2 / sqrt2sig) - erf(y1 / sqrt2sig)

        dg_df = (erfxx * erfyy)/4
        dg_dx_mean = flux_2sqrtpi*(e_x1_2s_sq - e_x2_2s_sq) * erfyy
        dg_dy_mean = flux_2sqrtpi*(e_y1_2s_sq - e_y2_2s_sq) * erfxx
        dg_dstddev = (
            flux/(twosigsq*np.sqrt(2*np.pi))
            *((x1*e_x1_2s_sq - x2*e_x2_2s_sq)*erfyy
              + (y1*e_y1_2s_sq - y2*e_y2_2s_sq)*erfxx)
        )

        return [dg_df, dg_dx_mean, dg_dy_mean, dg_dstddev]

    @property
    def input_units(self):
        if self.x_mean.unit is None and self.y_mean.unit is None:
            return None
        return {self.inputs[0]: self.x_mean.unit, self.inputs[1]: self.y_mean.unit}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        # Note that here we need to make sure that x and y are in the same
        # units otherwise this can lead to issues since rotation is not well
        # defined.
        if inputs_unit[self.inputs[0]] != inputs_unit[self.inputs[1]]:
            raise UnitsError("Units of 'x' and 'y' inputs should match")
        return {
            "x_mean": inputs_unit[self.inputs[0]],
            "y_mean": inputs_unit[self.inputs[0]],
            "stddev": inputs_unit[self.inputs[0]],
            "flux": outputs_unit[self.outputs[0]],
        }
