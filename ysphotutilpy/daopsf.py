import numpy as np
from astropy.nddata import Cutout2D
from astropy.modeling.fitting import LevMarLSQFitter

__all__ = ["dao_nstar_clamp", "dao_weight_map", "dao_nstar"]


def dao_nstar_clamp(p_old, p_new_raw, p_clamp):
    ''' The "clamp" for NSTAR routine
    Note
    ----
    StetsonPB 1987, PASP, 99, 191, p.208
    '''
    dp_raw = p_new_raw - p_old
    p_new = p_old + dp_raw / (1 + np.abs(dp_raw) / p_clamp)
    return p_new


def dao_weight_map(data, position, r_fit):
    ''' The weight for centering routine
    Note
    ----
    StetsonPB 1987, PASP, 99, 191, p.207
    '''
    x0, y0 = position
    is_cut = False
    if np.any(np.array(data.shape) > (2 * r_fit + 1)):  # To save CPU
        is_cut = True
        cut = Cutout2D(data=data, position=(x0, y0), size=(2 * r_fit + 1))
        data = cut.data
        x0, y0 = cut.to_cutout_position((x0, y0))

    nx, ny = data.shape[1], data.shape[0]
    xx_data, yy_data = np.mgrid[:ny, :nx]

    # add 1.e-6 to avoid zero division
    distance_map = np.sqrt((xx_data - x0)**2 + (yy_data - y0)**2) + 1.e-6
    dist = np.ma.array(data=distance_map, mask=(distance_map > r_fit))
    Rr_term = (r_fit / dist)**2 - 1
    weight_map = 5 / (5 + 1 / Rr_term)
    return weight_map


def dao_nstar(data, position, psf, r_fit, flux_init=1, sky=0, err=None,
              fitter=LevMarLSQFitter(), full=True):
    '''
    psf: photutils.psf.FittableImageModel
    '''
    if err is None:
        err = np.zeros_like(data)

    fbox = 2 * r_fit + 1  # fitting box size
    fcut = Cutout2D(data, position=position, size=fbox)  # "fitting" cut
    fcut_err = Cutout2D(err, position=position, size=fbox).data
    fcut_skysub = fcut.data - sky  # Must be sky subtracted before PSF fitting
    pos_fcut_init = fcut.to_cutout_position(position)  # Order of x, y

    dao_weight = dao_weight_map(fcut_skysub, pos_fcut_init, r_fit)
    # astropy gets ``weight`` = 1 / sigma.. strange..
    astropy_weight = np.sqrt(dao_weight.data) / fcut_err
    astropy_weight[dao_weight.mask] = 0

    xx_fit, yy_fit = np.mgrid[:fcut_skysub.shape[1], :fcut_skysub.shape[0]]
    psf.flux = flux_init
    psf.x_0, psf.y_0 = fcut.center_cutout
    fit = fitter(psf, xx_fit, yy_fit, fcut_skysub, weights=dao_weight)
    pos_fit = fcut.to_original_position((fit.x_0, fit.y_0))

    if full:
        return fit, pos_fit, fitter, astropy_weight, fcut_skysub, fcut_err

    else:
        return fit, pos_fit, fitter
