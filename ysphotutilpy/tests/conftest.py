"""
Shared pytest fixtures for ysphotutilpy tests.

All test data uses analytically-derived values for exact verification.

Note: Expected warnings (divide by zero, invalid value, etc.) are filtered
in pyproject.toml under [tool.pytest.ini_options].
"""

import numpy as np
import pytest
from astropy.nddata import CCDData
from astropy.modeling.functional_models import Gaussian2D


# =============================================================================
# Constants for reproducibility
# =============================================================================
RANDOM_SEED = 42
SHAPE_SMALL = (50, 50)
SHAPE_MEDIUM = (100, 100)
SHAPE_LARGE = (200, 200)


# =============================================================================
# Uniform Arrays - exact values for area/sum calculations
# =============================================================================
@pytest.fixture
def uniform_50x50():
    """50x50 array filled with value 10.0"""
    return np.full(SHAPE_SMALL, 10.0, dtype=np.float64)


@pytest.fixture
def uniform_100x100():
    """100x100 array filled with value 10.0"""
    return np.full(SHAPE_MEDIUM, 10.0, dtype=np.float64)


@pytest.fixture
def uniform_200x200():
    """200x200 array filled with value 100.0"""
    return np.full(SHAPE_LARGE, 100.0, dtype=np.float64)


# =============================================================================
# Gaussian 2D Sources - known amplitude, position, sigma
# =============================================================================
def make_gaussian_2d(shape, x_mean, y_mean, amplitude, x_stddev, y_stddev=None,
                     theta=0, background=0):
    """
    Create a 2D Gaussian array with known parameters.

    Parameters
    ----------
    shape : tuple
        (ny, nx) shape of the output array
    x_mean, y_mean : float
        Center position (0-indexed, x=column, y=row)
    amplitude : float
        Peak amplitude above background
    x_stddev, y_stddev : float
        Standard deviations along x and y axes
    theta : float
        Rotation angle in radians
    background : float
        Constant background level

    Returns
    -------
    data : ndarray
        2D array with Gaussian + background
    """
    if y_stddev is None:
        y_stddev = x_stddev

    yy, xx = np.mgrid[:shape[0], :shape[1]]
    gauss = Gaussian2D(
        amplitude=amplitude,
        x_mean=x_mean,
        y_mean=y_mean,
        x_stddev=x_stddev,
        y_stddev=y_stddev,
        theta=theta
    )
    return gauss(xx, yy) + background


@pytest.fixture
def gaussian_source_centered():
    """
    Gaussian source at center (50, 50) of 100x100 array.

    Parameters:
        amplitude = 1000
        x_mean = y_mean = 50.0
        sigma = 3.0
        background = 100
    """
    return make_gaussian_2d(
        shape=SHAPE_MEDIUM,
        x_mean=50.0,
        y_mean=50.0,
        amplitude=1000.0,
        x_stddev=3.0,
        background=100.0
    )


@pytest.fixture
def gaussian_source_offset():
    """
    Gaussian source at (50.3, 50.7) for centroiding tests.

    Parameters:
        amplitude = 1000
        x_mean = 50.3
        y_mean = 50.7
        sigma = 3.0
        background = 100
    """
    return make_gaussian_2d(
        shape=SHAPE_MEDIUM,
        x_mean=50.3,
        y_mean=50.7,
        amplitude=1000.0,
        x_stddev=3.0,
        background=100.0
    )


@pytest.fixture
def gaussian_params_centered():
    """Parameters dict for gaussian_source_centered"""
    return {
        'amplitude': 1000.0,
        'x_mean': 50.0,
        'y_mean': 50.0,
        'x_stddev': 3.0,
        'y_stddev': 3.0,
        'background': 100.0
    }


@pytest.fixture
def gaussian_params_offset():
    """Parameters dict for gaussian_source_offset"""
    return {
        'amplitude': 1000.0,
        'x_mean': 50.3,
        'y_mean': 50.7,
        'x_stddev': 3.0,
        'y_stddev': 3.0,
        'background': 100.0
    }


# =============================================================================
# CCDData objects with headers
# =============================================================================
@pytest.fixture
def ccd_uniform():
    """
    CCDData with uniform value 100, standard header.

    Header:
        GAIN = 2.0 e/ADU
        RDNOISE = 5.0 e
        EXPTIME = 60.0 s
    """
    data = np.full(SHAPE_MEDIUM, 100.0, dtype=np.float64)
    header = {
        'GAIN': 2.0,
        'RDNOISE': 5.0,
        'EXPTIME': 60.0,
    }
    return CCDData(data=data, unit='adu', header=header)


@pytest.fixture
def ccd_with_source(gaussian_source_centered):
    """
    CCDData with Gaussian source, standard header.

    Source at (50, 50), amplitude=1000, sigma=3, background=100
    """
    header = {
        'GAIN': 2.0,
        'RDNOISE': 5.0,
        'EXPTIME': 60.0,
    }
    return CCDData(data=gaussian_source_centered, unit='adu', header=header)


@pytest.fixture
def ccd_with_source_offset(gaussian_source_offset):
    """
    CCDData with Gaussian source at offset position (50.3, 50.7).
    """
    header = {
        'GAIN': 2.0,
        'RDNOISE': 5.0,
        'EXPTIME': 60.0,
    }
    return CCDData(data=gaussian_source_offset, unit='adu', header=header)


# =============================================================================
# Arrays with noise for sky estimation tests
# =============================================================================
@pytest.fixture
def uniform_with_noise():
    """
    Uniform array (mean=100) with Gaussian noise (std=10).

    Uses fixed seed for reproducibility.
    """
    np.random.seed(RANDOM_SEED)
    return np.random.normal(loc=100.0, scale=10.0, size=SHAPE_MEDIUM)


@pytest.fixture
def sky_array_with_outliers():
    """
    Uniform sky (100) with a few outlier pixels for sigma-clipping tests.
    """
    data = np.full(SHAPE_SMALL, 100.0, dtype=np.float64)
    # Add outliers at known positions
    data[10, 10] = 500.0  # high outlier
    data[20, 20] = 500.0  # high outlier
    data[30, 30] = -100.0  # low outlier (if sky can be negative)
    return data


# =============================================================================
# Polarimetry test data
# =============================================================================
@pytest.fixture
def polarimetry_unpolarized():
    """
    Unpolarized source: all o-ray and e-ray intensities equal.

    o_xxx = e_xxx = 1000 for all HWP angles
    Expected: q = 0, u = 0
    """
    return {
        'o_000': 1000.0, 'e_000': 1000.0,
        'o_450': 1000.0, 'e_450': 1000.0,
        'o_225': 1000.0, 'e_225': 1000.0,
        'o_675': 1000.0, 'e_675': 1000.0,
    }


@pytest.fixture
def polarimetry_q_only():
    """
    Linearly polarized source with q != 0, u = 0.

    Setup: e_000/o_000 > 1, e_450/o_450 < 1 (inverse), others = 1
    r_q = sqrt((e_000/o_000) / (e_450/o_450)) = sqrt((2/1)/(0.5/1)) = sqrt(4) = 2
    q = (r_q - 1) / (r_q + 1) = (2 - 1) / (2 + 1) = 1/3

    For u=0: e_225/o_225 = e_675/o_675
    """
    return {
        'o_000': 1000.0, 'e_000': 2000.0,  # ratio = 2
        'o_450': 1000.0, 'e_450': 500.0,   # ratio = 0.5
        'o_225': 1000.0, 'e_225': 1000.0,  # ratio = 1
        'o_675': 1000.0, 'e_675': 1000.0,  # ratio = 1
    }


@pytest.fixture
def polarimetry_qu_known():
    """
    Polarized source with known q and u values.

    For q: r_q = sqrt((e_000/o_000)/(e_450/o_450)) = sqrt(4) = 2
           q = (2-1)/(2+1) = 1/3

    For u: r_u = sqrt((e_225/o_225)/(e_675/o_675)) = sqrt(4) = 2
           u = (2-1)/(2+1) = 1/3
    """
    return {
        'o_000': 1000.0, 'e_000': 2000.0,
        'o_450': 1000.0, 'e_450': 500.0,
        'o_225': 1000.0, 'e_225': 2000.0,
        'o_675': 1000.0, 'e_675': 500.0,
    }


# =============================================================================
# Test arrays for specific modules
# =============================================================================
@pytest.fixture
def simple_array_for_std():
    """Array [1, 2, 3, 4, 5] for standard deviation tests."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture
def array_with_nan():
    """Array with NaN values for nanstd tests."""
    return np.array([1.0, 2.0, np.nan, 4.0, 5.0])


# =============================================================================
# Helper to skip if sep not installed
# =============================================================================
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_sep: mark test as requiring sep package"
    )


@pytest.fixture
def skip_if_no_sep():
    """Skip test if sep is not installed."""
    pytest.importorskip("sep")
