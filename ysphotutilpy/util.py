"""
A collection of temporary utilities, and likely be removed if similar
functionality can be achieved by pre-existing packages.
"""
import numpy as np
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.functional_models import Gaussian2D

__all__ = ["magsum", "sqsum", "err_prop",
           "_linear_unit_converter", "convert_pct", "convert_deg",
           "bezel_mask", "Gaussian2D_correct",
           "fit_astropy_model", "fit_Gaussian2D", "gaussian_kernel"]


def magsum(*args):
    """ Calculates the "sum" of magnitudes.
    Note
    ----
    To calculate magnitude error, use ``err_prop(*merrs)``.
    """
    return -2.5 * np.log10(np.sum(10**(-0.4*np.array(args))))


def sqsum(*args):
    _sum = 0
    for a in args:
        _sum += a**2
    return _sum


def err_prop(*errs):
    return np.sqrt(sqsum(*errs))


def _linear_unit_converter(*args, factor=1, already=False, convert2unit=False):
    ''' Converts units among non-physical units (%, deg-radian, etc).
    Parameters
    ----------
    factor : float, optional.
        The factor to convert natural unit (dimensionless) to desired unit.
        ``factor=100`` will be **multiplied** to convert to %, and will be
        **divided** to convert the value to natural unit.

    already : bool, optional.
        Whether the input args are already in the desired unit specified by
        ``factor``.

    convert2unit : bool, optional.
        Whether to convert the input args to the unit specified by ``factor``.


    Example
    -------
    This is a converter for per-cent (%) VS natural unit (absolute value <=
    1)::

        _linear_unit_converter(*args, factor=100, already=already,
                               convert2unit=convert2unit)

    and this is a converter for degrees VS radian::

        _linear_unit_converter(*args, factor=180/np.pi, already=already,
                               convert2unit=convert2unit)

    '''
    if not already and convert2unit:
        factor = factor
    elif already and not convert2unit:
        factor = 1/factor
    else:  # i.e., both True or both False
        factor = 1
    return [a*factor for a in args]


def convert_pct(*args, already=False, convert2unit=False):
    return _linear_unit_converter(*args, factor=100, already=already,
                                  convert2unit=convert2unit)


def convert_deg(*args, already=False, convert2unit=False):
    return _linear_unit_converter(*args, factor=180/np.pi, already=already,
                                  convert2unit=convert2unit)


def bezel_mask(
        xvals,
        yvals,
        nx,
        ny,
        bezel=(0, 0),
        bezel_x=None,
        bezel_y=None
):
    '''
    Parameters
    ----------
    xvals, yvals : array-like
        The x and y position values.

    nx, ny : int or float
        The number of x and y pixels (``NAXIS2`` and ``NAXIS1``, respectively
        from header).

    bezel : int, float, array-like, optional
        The bezel size. If array-like, it should be ``(lower, upper)``. If only
        this is given and ``bezel_x`` and/or ``bezel_y`` is/are ``None``,
        it/both will be replaced by ``bezel``. If you want to keep some stars
        outside the edges, put negative values (e.g., ``-5``).

    bezel_x, bezel_y : int, float, 2-array-like, optional
        The bezel (border width) for x and y axes. If array-like, it should be
        ``(lower, upper)``. Mathematically put, only objects with center
        ``(bezel_x[0] + 0.5 < center_x) & (center_x < nx - bezel_x[1] - 0.5)``
        (similar for y) will be selected. If you want to keep some stars
        outside the edges, put negative values (e.g., ``-5``).
    '''
    bezel = np.array(bezel)
    if len(bezel) == 1:
        bezel = np.repeat(bezel, 2)

    if bezel_x is None:
        bezel_x = bezel.copy()
    else:
        bezel_x = np.atleast_1d(bezel_x)
        if len(bezel_x) == 1:
            bezel_x = np.repeat(bezel_x, 2)

    if bezel_y is None:
        bezel_y = bezel.copy()
    else:
        bezel_y = np.atleast_1d(bezel_y)
        if len(bezel_y) == 1:
            bezel_y = np.repeat(bezel_y, 2)

    mask = ((xvals < bezel_x[0] + 0.5)
            | (yvals < bezel_y[0] + 0.5)
            | (xvals > (nx - bezel_x[1]) - 0.5)
            | (yvals > (ny - bezel_y[1]) - 0.5)
            )
    return mask


def normalize(num, lower=0, upper=360, b=False):
    """Normalize number to range [lower, upper) or [lower, upper].
    From phn: https://github.com/phn/angles

    Parameters
    ----------
    num : float
        The number to be normalized.

    lower : int
        Lower limit of range. Default is 0.

    upper : int
        Upper limit of range. Default is 360.

    b : bool
        Type of normalization. Default is False. See notes. When b=True, the
        range must be symmetric about 0. When b=False, the range must be
        symmetric about 0 or ``lower`` must be equal to 0.

    Returns
    -------
    n : float
        A number in the range [lower, upper) or [lower, upper].

    Raises
    ------
    ValueError
      If lower >= upper.

    Notes
    -----
    If the keyword `b == False`, then the normalization is done in the
    following way. Consider the numbers to be arranged in a circle, with the
    lower and upper ends sitting on top of each other. Moving past one limit,
    takes the number into the beginning of the other end. For example, if range
    is [0 - 360), then 361 becomes 1 and 360 becomes 0. Negative numbers move
    from higher to lower numbers. So, -1 normalized to [0 - 360) becomes 359.
    When b=False range must be symmetric about 0 or lower=0. If the keyword `b
    == True`, then the given number is considered to "bounce" between the two
    limits. So, -91 normalized to [-90, 90], becomes -89, instead of 89. In
    this case the range is [lower, upper]. This code is based on the function
    `fmt_delta` of `TPM`. When b=True range must be symmetric about 0.

    Examples
    --------
    >>> normalize(-270,-180,180)
    90.0
    >>> import math
    >>> math.degrees(normalize(-2*math.pi,-math.pi,math.pi))
    0.0
    >>> normalize(-180, -180, 180)
    -180.0
    >>> normalize(180, -180, 180)
    -180.0
    >>> normalize(180, -180, 180, b=True)
    180.0
    >>> normalize(181,-180,180)
    -179.0
    >>> normalize(181, -180, 180, b=True)
    179.0
    >>> normalize(-180,0,360)
    180.0
    >>> normalize(36,0,24)
    12.0
    >>> normalize(368.5,-180,180)
    8.5
    >>> normalize(-100, -90, 90)
    80.0
    >>> normalize(-100, -90, 90, b=True)
    -80.0
    >>> normalize(100, -90, 90, b=True)
    80.0
    >>> normalize(181, -90, 90, b=True)
    -1.0
    >>> normalize(270, -90, 90, b=True)
    -90.0
    >>> normalize(271, -90, 90, b=True)
    -89.0
    """
    if lower >= upper:
        ValueError("lower must be lesser than upper")
    if not b:
        if not ((lower + upper == 0) or (lower == 0)):
            raise ValueError(
                'When b=False lower=0 or range must be symmetric about 0.')
    else:
        if not (lower + upper == 0):
            raise ValueError('When b=True range must be symmetric about 0.')

    from math import ceil, floor

    # abs(num + upper) and abs(num - lower) are needed, instead of abs(num),
    # since the lower and upper limits need not be 0. We need to add half size
    # of the range, so that the final result is lower + <value> or upper -
    # <value>, respectively.
    res = num
    if not b:
        res = num
        if num > upper or num == lower:
            num = lower + abs(num + upper) % (abs(lower) + abs(upper))
        if num < lower or num == upper:
            num = upper - abs(num - lower) % (abs(lower) + abs(upper))

        res = lower if num == upper else num
    else:
        total_length = abs(lower) + abs(upper)
        if num < -total_length:
            num += ceil(num / (-2 * total_length)) * 2 * total_length
        if num > total_length:
            num -= floor(num / (2 * total_length)) * 2 * total_length
        if num > upper:
            num = total_length - num
        if num < lower:
            num = -total_length - num

        res = num

    res *= 1.0  # Make all numbers float, to be consistent

    return res


def Gaussian2D_correct(model, theta_lower=-np.pi/2, theta_upper=np.pi/2):
    ''' Sets x = semimajor axis and theta to be in [-pi/2, pi/2] range.
    Example
    -------
    >>> from astropy.modeling.functional_models import Gaussian2D
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> from yspy.util import astropy_util as au
    >>> gridsize = np.zeros((40, 60))
    >>> common = dict(x_mean=20, y_mean=20, x_stddev=5)
    >>> y, x = np.mgrid[:gridsize.shape[0], :gridsize.shape[1]]
    >>> theta_arr = [-12345.678, -100, -1, -0.1, 0, 0.1, 1, 100, 12345.678]
    >>> for sig_y in [-1, -0.1, 0.1, 1, 10]:
    >>>     for theta in theta_arr:
    >>>         g = Gaussian2D(**common, theta=theta)
    >>>         g_c = Gaussian2D_correct(g)
    >>>         f, ax = plt.subplots(3)
    >>>         ax[0].imshow(g(x, y), vmax=1, vmin=1.e-12)
    >>>         ax[1].imshow(g_c(x, y), vmax=1, vmin=1.e-12)
    >>>         ax[2].imshow(g(x, y) - g_c(x, y), vmin=1.e-20, vmax=1.e-12)
    >>>         np.testing.assert_almost_equal(g(x, y) - g_c(x, y), gridsize)
    >>>         plt.pause(0.1)
    You may see some patterns in the residual image, they are < 10**(-13).
    '''
    # I didn't use ``Gaussian2D`` directly, because GaussianConst2D from
    # photutils may also be used.
    new_model = model.copy()
    sig_x = np.abs(model.x_stddev.value)
    sig_y = np.abs(model.y_stddev.value)
    theta = model.theta.value

    if sig_x > sig_y:
        theta_norm = normalize(theta, theta_lower, theta_upper)
        new_model.x_stddev.value = sig_x
        new_model.y_stddev.value = sig_y
        new_model.theta.value = theta_norm

    else:
        theta_norm = normalize(theta + np.pi/2, theta_lower, theta_upper)
        new_model.x_stddev.value = sig_y
        new_model.y_stddev.value = sig_x
        new_model.theta.value = theta_norm

    return new_model


def fit_astropy_model(data, model_init, sigma=None, fitter=LevMarLSQFitter(), **kwargs):
    """
    Parameters
    ----------
    data : ndarray
        The data to fit the model.

    sigma : ndarray, None, optional
        The Gaussian error-bar of each pixel. In ``astropy``, we must give
        ``weights=1/sigma``, which can be confusing.

    fitter : astropy fitter
        The fitter that will be used to fit.

    kwargs :
        The keyword arguments for the ``fitter.__call__``.

    Returns
    -------
    fitted : model
        The fitted model.

    fitter : fitter
        The fitter (maybe informative, e.g., ``fitter.fit_info``).
    """
    yy, xx = np.mgrid[:data.shape[0], :data.shape[1]]
    if sigma is not None:
        weights = 1/sigma
    else:
        weights = None
    fitted = fitter(model_init, xx, yy, data, weights=weights, **kwargs)
    return fitted, fitter


def fit_Gaussian2D(data, model_init, correct=True, sigma=None,
                   fitter=LevMarLSQFitter(), **kwargs):
    """ Identical to fit_astropy_model but for Gaussian2D correct.

    Notes
    -----
    photutils.centroids.GaussianConst2D is also usable.
    """
    fitted, fitter = fit_astropy_model(
        data=data, model_init=model_init, sigma=sigma, fitter=fitter, **kwargs)
    if correct:
        fitted = Gaussian2D_correct(fitted)
    return fitted, fitter


'''
from warnings import warn
from astropy.modeling import Fittable2DModel, Parameter
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.modeling.models import CONSTRAINTS_DOC, Const2D, Moffat2D
from astropy.utils.exceptions import AstropyUserWarning

# TODO: Elliptical moffat...
# https://iraf.net/irafhelp.php?val=daopars&help=Help+Page
class MoffatConst2D(Fittable2DModel):
    """
    A model for a 2D Moffat plus a constant.

    Parameters
    ----------
    constant : float
        Value of the constant.
    amplitude : float
        Amplitude of the model.
    x_0 : float
        x position of the maximum of the Moffat model.
    y_0 : float
        y position of the maximum of the Moffat model.
    gamma : float
        Core width of the Moffat model.
    alpha : float
        Power index of the Moffat model.
    """

    constant = Parameter(default=1)
    amplitude = Parameter(default=1)
    x_0 = Parameter(default=0)
    y_0 = Parameter(default=0)
    gamma = Parameter(default=1)
    alpha = Parameter(default=1)

    @staticmethod
    def evaluate(x, y, constant, amplitude, x_0, y_0, gamma, alpha):
        """Two dimensional Gaussian plus constant function."""

        model = Const2D(constant)(x, y) + Moffat2D(amplitude, x_0, y_0,
                                                   gamma, alpha)(x, y)
        return model


MoffatConst2D.__doc__ += CONSTRAINTS_DOC


def fit_2dmoffat(data, error=None, mask=None):
    """
    Fit a 2D Moffat plus a constant to a 2D image.

    Invalid values (e.g. NaNs or infs) in the ``data`` or ``error``
    arrays are automatically masked.  The mask for invalid values
    represents the combination of the invalid-value masks for the
    ``data`` and ``error`` arrays.

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
    result : A `MoffatConst2D` model instance.
        The best-fitting Moffat 2D model.
    """

    from ..morphology import data_properties  # prevent circular imports

    data = np.ma.asanyarray(data)

    if mask is not None and mask is not np.ma.nomask:
        mask = np.asanyarray(mask)
        if data.shape != mask.shape:
            raise ValueError('data and mask must have the same shape.')
        data.mask |= mask

    if np.any(~np.isfinite(data)):
        data = np.ma.masked_invalid(data)
        warn('Input data contains input values (e.g. NaNs or infs), '
             'which were automatically masked.', AstropyUserWarning)

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
                         'fit a 2D Moffat plus a constant.')

    # assign zero weight to masked pixels
    if data.mask is not np.ma.nomask:
        weights[data.mask] = 0.

    mask = data.mask
    data.fill_value = 0.0
    data = data.filled()

    # Subtract the minimum of the data as a crude background estimate.
    # This will also make the data values positive, preventing issues with
    # the moment estimation in data_properties (moments from negative data
    # values can yield undefined Gaussian parameters, e.g. x/y_stddev).
    props = data_properties(data - np.min(data), mask=mask)

    init_const = 0.  # subtracted data minimum above
    init_amplitude = np.ptp(data)
    g_init = GaussianConst2D(constant=init_const, amplitude=init_amplitude,
                             x_mean=props.xcentroid.value,
                             y_mean=props.ycentroid.value,
                             x_stddev=props.semimajor_axis_sigma.value,
                             y_stddev=props.semiminor_axis_sigma.value,
                             theta=props.orientation.value)
    fitter = LevMarLSQFitter()
    y, x = np.indices(data.shape)
    gfit = fitter(g_init, x, y, data, weights=weights)

    return gfit
'''


def gaussian_kernel(fwhm=None, sigma=None, theta=0, nsigma=5, normalize_area=False):
    """ Generates a 2D array of Gaussian, usable a kernel.

    Parameters
    ----------
    fwhm, sigma : float, array-like, optional.
        The full-width at half-maximum (FWHM) and standard deviation (sigma) of
        the Gaussian. One and only one of these should be given. If array-like,
        it should be ``(sigma_or_fwhm_x, sigma_or_fwhm_y)``. If only one value
        is given, it will be used for both x and y.

    theta : float, optional.
        The rotation angle of the Gaussian in degrees.

    nsigma : int or float, optional.
        The number of sigma to be used for the kernel size. The kernel size
        will be ``ceil(nsigma*sigma)`` along each direction. If variable size
        is desired, use size-2 array (e.g., ``nsigma=(5, 10)`` to make 5-sigma
        along the x- and 10-sigma along the y-direction).

    normalize_area : bool, optional.
        Whether to normalize the area of the Gaussian to 1. If ``False``, the
        amplitude will be 1. If ``True``, the amplitude will be
        1/(2*pi*sigma_x*sigma_y).

    Returns
    -------
    gauss : array-like
        The 2D Gaussian kernel of shape ``(ceil(nsigma[0]*sigma[0]),
        ceil(nsigma[1]*sigma[1])``.
    """
    if ((fwhm is None) + (sigma is None)) != 1:
        raise ValueError("One and only one of `fwhm` and `sigma` should be given.")
    elif fwhm is not None:
        sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    sigma = np.atleast_1d(sigma)
    if len(sigma) == 1:
        sigma = np.repeat(sigma, 2)
    _amp = 1/(2*np.pi*sigma[0]*sigma[1]) if normalize_area else 1
    gauss = Gaussian2D(
        amplitude=_amp, x_mean=0, y_mean=0,
        x_stddev=sigma[0], y_stddev=sigma[1], theta=theta
    )
    shape = np.ceil(nsigma*sigma[::-1]).astype(int)

    return gauss(*np.ogrid[-shape[0]/2:shape[0]/2+1, -shape[1]/2:shape[1]/2+1])
