from warnings import warn

import numpy as np
import bottleneck as bn
from astropy.nddata import CCDData
from astropy.stats import sigma_clip
from astropy.table import Table

__all__ = ['quick_sky_circ', 'sky_fit', "annul2values"]


def quick_sky_circ(ccd, pos, r_in=10, r_out=20):
    """ Estimate sky with crude presets
    """
    from photutils.aperture import CircularAnnulus
    annulus = CircularAnnulus(pos, r_in=r_in, r_out=r_out)
    return sky_fit(ccd, annulus)


def sky_fit(
        ccd,
        annulus=None,
        mask=None,
        method='mode',
        mode_option='sex',
        std_ddof=1,
        to_table=True,
        return_skyarr=False,
        **kwargs
):
    """ Estimate the sky value from image and annulus.

    Parameters
    ----------
    ccd: `~astropy.nddata.CCDData`, HDU, ndarray
        The image data to extract sky at given annulus.

    annulus: annulus object, optional.
        The annulus which will be used to estimate sky values.
        If `None` (default), the whole image will be used.

    method : {"mean", "median", "mode"}, optional.
        The method to estimate sky value. You can give options to "mode" case;
        see `mode_option`. "mode" is analogous to Mode Estimator Background
        of photutils.

    mode_option : {"sex", "IRAF", "MMM"}, optional.
        Three options::

          * sex  == (med_factor, mean_factor) = (2.5, 1.5)
          * IRAF == (med_factor, mean_factor) = (3, 2)
          * MMM  == (med_factor, mean_factor) = (3, 2)

        where ``msky = (med_factor * med) - (mean_factor * mean)``.

    std_ddof : int, optional.
        The "delta-degrees of freedom" for sky standard deviation calculation.

    to_table : bool, optional.
        If True, the output will be a `~astropy.table.Table` object. Otherwise,
        return a `dict`.

    return_skyarr : bool, optional.
        If True, the 1-d array (or list of such if multiple annuli) of sky
        values will be returned.

    kwargs : dict, optional.
        The keyword arguments for sigma-clipping.

    Returns
    -------
    skytable: `~astropy.table.Table` or dict
        The table or dict of the followings.

        msky : float
            The estimated sky value within the all_sky data, after sigma clipping.

        ssky : float
            The sample standard deviation of sky value within the all_sky data,
            after sigma clipping.

        nsky : int
            The number of pixels which were used for sky estimation after the
            sigma clipping.

        nrej : int
            The number of pixels which are rejected after sigma clipping.

    skys : ndarray or list of ndarray
        The 1-d array (or list of such if multiple annuli) of sky values.

    """
    def _sstd(arr, ddof=0, axis=None):
        return np.sqrt(arr.size/(arr.size - ddof))*bn.nanstd(arr, axis=axis)

    skydicts = []
    if annulus is None:
        try:  # CCDData or HDU
            skys = [ccd.data.ravel()]
        except AttributeError:  # ndarray
            skys = [ccd.ravel()]
    else:
        skys = annul2values(ccd, annulus, mask=mask)

    for i, sky in enumerate(skys):
        skydict = {}
        sky_clip = sigma_clip(sky, masked=False, stdfunc=_sstd, **kwargs)
        std = np.std(sky_clip, ddof=std_ddof)
        nrej = sky.size - sky_clip.size
        nsky = sky.size - nrej
        if nrej < 0:
            raise ValueError('nrej < 0: check the code')

        if nrej > nsky:  # rejected > survived
            warn('More than half of the pixels rejected.')

        if method.lower() == 'mode':
            mean = np.mean(sky_clip)
            med = np.median(sky_clip)
            if mode_option.lower() == 'sex':
                skydict["msky"] = med if (mean - med)/std > 0.3 else (2.5*med) - (1.5*mean)
            elif mode_option.lower() == 'iraf':
                skydict["msky"] = mean if (mean < med) else 3*med - 2*mean
            elif mode_option.lower() == 'mmm':
                skydict["msky"] = 3*med - 2*mean
            else:
                raise ValueError('mode_option not understood')
        elif method.lower() == 'mean':
            skydict["msky"] = np.mean(sky_clip)
        elif method.lower() == 'median':
            skydict["msky"] = np.median(sky_clip)

        skydict['ssky'] = std
        skydict['nsky'] = nsky
        skydict['nrej'] = nrej
        skydicts.append(skydict)
    if to_table:
        if return_skyarr:
            return Table(skydicts), skys
        return Table(skydicts)
    return (skydict, skys) if return_skyarr else skydict


def annul2values(
        ccd,
        annulus,
        mask=None
):
    ''' Extracts the pixel values from the image with annuli

    Parameters
    ----------
    ccd : CCDData, ndarray
        The image which the annuli in ``annulus`` are to be applied.
    annulus : `~photutils.Aperture` object
        The annuli to be used to extract the pixel values.
    # fill_value: float or nan
    #     The pixels which are masked by ``ccd.mask`` will be replaced with
    #     this value.

    Returns
    -------
    values: list of ndarray
        The list of pixel values. Length is the same as the number of annuli in
        ``annulus``.
    '''
    if isinstance(ccd, CCDData):
        _ccd = ccd.copy()
        _arr = _ccd.data
        _mask = _ccd.mask
    else:  # ndarrayk
        _arr = np.array(ccd)
        _mask = None

    if _mask is None:
        _mask = np.zeros_like(_arr).astype(bool)

    if mask is not None:
        _mask |= mask

    an_masks = annulus.to_mask(method='center')
    try:
        if annulus.isscalar:  # as of photutils 0.7
            an_masks = [an_masks]
    except AttributeError:
        pass

    # FIXME: use the new an_mask.get_values() introduced in 2021 Feb
    # values = []
    # for i, an_mask in enumerate(an_masks):
    #     # result identical to an.data itself, but just for safety...
    #     in_an = (an_mask.data == 1).astype(float)  # float for NaN below
    #     # replace to NaN for in_an=0, because sometimes pixel itself is 0...
    #     in_an[in_an == 0] = np.nan
    #     skys_i = an_mask.multiply(_arr, fill_value=np.nan) * in_an
    #     ccdmask_i = an_mask.multiply(_mask, fill_value=False)
    #     mask_i = (np.isnan(skys_i) + ccdmask_i).astype(bool)
    #     # skys_i = an.multiply(_arr, fill_value=np.nan)
    #     # sky_xy = np.nonzero(an.data)
    #     # sky_all = mask_im[sky_xy]
    #     # sky_values = sky_all[~np.isnan(sky_all)]
    #     # values.append(sky_values)
    #     values.append(np.array(skys_i[~mask_i].ravel(), dtype=_arr.dtype))

    return [am.get_values(_arr, _mask) for am in an_masks]


def mmm_dao(
    sky,
    highbad=None,
    min_nsky=20,
    maxiter=50,
    readnoise=0,
):
    """Use the MMM-DAO algorithm to estimate the sky value.

    The MMM-DAO algorithm is a robust estimator of the sky value. This
    algorithm is based on the MMM algorithm, which is a modification of the
    DAOFIND algorithm in DAOPHOT. This algorithm is described in Section 2.2
    of `Stetson (1987) <https://ui.adsabs.harvard.edu/abs/1987PASP...99..191S/abstract>`_.

    Parameters
    ----------
    sky : array-like
        The sky values. This is usually the difference between the image and
        the object mask.

    highbad : bool, optional
        If True, the sky values are assumed to be high values, and the
        algorithm will search for low values. If False, the sky values are
        assumed to be low values, and the algorithm will search for high
        values. If None, the algorithm will search for both high and low
        values. The default is None.

    min_nsky : float or int, optional
        The minimum number of sky values to use. The default is 20. (`minsky` in DAOPHOT MMM)

    maxiter : int, optional
        The maximum number of iterations. The default is 50.

    readnoise : float, optional
        The read noise of the image. The default is 0.

    Returns
    -------
    sky : float
        The estimated sky value.

    """
    sky = np.array(sky)
    if (nsky := sky.size) < min_nsky:
        sigma = -1.
        skew = 0.
        raise ValueError(f"Input vector must contain at least {min_nsky = } elements.")

    # do the MMM sky estimation similar to DAOPHOT (MMM.pro of https://idlastro.gsfc.nasa.gov/ftp/pro/idlphot/mmm.pro)
    sky = np.sort(sky)
    skymid = 0.5*sky[(nsky-1)//2] + 0.5*sky[nsky//2]
    cut1 = min([skymid-sky[0], sky[-1]-skymid])
    if highbad is not None:
        cut1 = min(cut1, highbad-skymid)
    cut2 = skymid + cut1
    cut1 = skymid - cut1
    goodmask = (cut1 <= sky) & (sky <= cut2)
    if (ngood := np.count_nonzero(goodmask)) == 0:
        sigma = -1.
        skew = 0.
        raise ValueError(f"No sky values survived: {cut1:.4f}<=sky<={cut2:.4f}")

    skydelta = sky[goodmask] - skymid
    deltasum = np.sum(skydelta)
    deltasumsq = np.sum(skydelta**2)
    _goodidx = np.where(goodmask)[0]
    max_idx = _goodidx[-1]
    min_idx = _goodidx[0] - 1

    # compute mean and standard deviation
    skymed = 0.5*(sky[(min_idx+max_idx+1)//2] + sky[(min_idx+max_idx)//2 + 1])

    """
 cut1 = min( [skymid-sky[0],sky[nsky-1] - skymid] )
 if N_elements(highbad) EQ 1 then cut1 = cut1 < (highbad - skymid)
 cut2 = skymid + cut1
 cut1 = skymid - cut1

; Select the pixels between Cut1 and Cut2

 good = where( (sky LE cut2) and (sky GE cut1), Ngood )
 if ( Ngood EQ 0 ) then begin
      sigma=-1.0 &  skew = 0.0
      message,/CON, NoPrint=Silent, $
           'ERROR - No sky values fall within ' + strtrim(cut1,2) + $
	   ' and ' + strtrim(cut2,2)
      return
   endif

 delta = sky[good] - skymid  ;Subtract median to improve arithmetic accuracy
 sum = total(delta,/double)
 sumsq = total(delta^2,/double)

 maximm = max( good,MIN=minimm )  ;Highest value accepted at upper end of vector
 minimm = minimm -1               ;Highest value reject at lower end of vector

; Compute mean and sigma (from the first pass).

 skymed = 0.5*sky[(minimm+maximm+1)/2] + 0.5*sky[(minimm+maximm)/2 + 1] ;median
 skymn = float(sum/(maximm-minimm))                            ;mean
 sigma = sqrt(sumsq/(maximm-minimm)-skymn^2)             ;sigma
 skymn = skymn + skymid         ;Add median which was subtracted off earlier


;    If mean is less than the mode, then the contamination is slight, and the
;    mean value is what we really want.
skymod =  (skymed LT skymn) ? 3.*skymed - 2.*skymn : skymn

; Rejection and recomputation loop:

 niter = 0
 clamp = 1
 old = 0
START_LOOP:
   niter = niter + 1
   if ( niter GT mxiter ) then begin
      sigma=-1.0 &  skew = 0.0
      message,/CON, NoPrint=Silent, $
           'ERROR - Too many ('+strtrim(mxiter,2) + ') iterations,' + $
           ' unable to compute sky'
      return
   endif

   if ( maximm-minimm LT minsky ) then begin    ;Error?

      sigma = -1.0 &  skew = 0.0
      message,/CON,NoPrint=Silent, $
            'ERROR - Too few ('+strtrim(maximm-minimm,2) +  $
           ') valid sky elements, unable to compute sky'
      return
   endif

; Compute Chauvenet rejection criterion.

    r = alog10( float( maximm-minimm ) )
    r = max( [ 2., ( -0.1042*r + 1.1695)*r + 0.8895 ] )

; Compute rejection limits (symmetric about the current mode).

    cut = r*sigma + 0.5*abs(skymn-skymod)
    if integer then cut = cut > 1.5
    cut1 = skymod - cut   &    cut2 = skymod + cut
;
; Recompute mean and sigma by adding and/or subtracting sky values
; at both ends of the interval of acceptable values.

    redo = 0B
    newmin = minimm
    tst_min = sky[newmin+1] GE cut1      ;Is minimm+1 above current CUT?
    done = (newmin EQ -1) and tst_min    ;Are we at first pixel of SKY?
    if ~done then  $
        done =  (sky[newmin>0] LT cut1) and tst_min
    if ~done then begin
        istep = 1 - 2*fix(tst_min)
        repeat begin
                newmin = newmin + istep
                done = (newmin EQ -1) || (newmin EQ nlast)
                if ~done then $
                    done = (sky[newmin] LE cut1) and (sky[newmin+1] GE cut1)
        endrep until done
        if tst_min then delta = sky[newmin+1:minimm] - skymid $
                   else delta = sky[minimm+1:newmin] - skymid
        sum = sum - istep*total(delta,/double)
        sumsq = sumsq - istep*total(delta^2,/double)
        redo = 1b
        minimm = newmin
     endif
;
   newmax = maximm
   tst_max = sky[maximm] LE cut2           ;Is current maximum below upper cut?
   done = (maximm EQ nlast) and tst_max    ;Are we at last pixel of SKY array?
   if ~done then $
       done = ( tst_max ) && (sky[(maximm+1)<nlast] GT cut2)
    if ~done then begin                 ;Keep incrementing NEWMAX
       istep = -1 + 2*fix(tst_max)         ;Increment up or down?
       Repeat begin
          newmax = newmax + istep
          done = (newmax EQ nlast) or (newmax EQ -1)
          if ~done then $
                done = ( sky[newmax] LE cut2 ) and ( sky[newmax+1] GE cut2 )
       endrep until done
       if tst_max then delta = sky[maximm+1:newmax] - skymid $
               else delta = sky[newmax+1:maximm] - skymid
       sum = sum + istep*total(delta,/double)
       sumsq = sumsq + istep*total(delta^2,/double)
       redo = 1b
       maximm = newmax
    endif
;
; Compute mean and sigma (from this pass).
;
   nsky = maximm - minimm
   if ( nsky LT minsky ) then begin    ;Error?
       sigma = -1.0 &  skew = 0.0
       message,NoPrint=Silent, /CON, $
               'ERROR - Outlier rejection left too few sky elements'
       return
   endif

   skymn = float(sum/nsky)
   sigma = float( sqrt( (sumsq/nsky - skymn^2)>0 ))
    skymn = skymn + skymid


;  Determine a more robust median by averaging the central 20% of pixels.
;  Estimate the median using the mean of the central 20 percent of sky
;  values.   Be careful to include a perfectly symmetric sample of pixels about
;  the median, whether the total number is even or odd within the acceptance
;  interval

        center = (minimm + 1 + maximm)/2.
        side = round(0.2*(maximm-minimm))/2.  + 0.25
        J = round(CENTER-SIDE)
        K = round(CENTER+SIDE)

;  In case  the data has a large number of of the same (quantized)
;  intensity, expand the range until both limiting values differ from the
;  central value by at least 0.25 times the read noise.

        if keyword_set(readnoise) then begin
          L = round(CENTER-0.25)
          M = round(CENTER+0.25)
          R = 0.25*readnoise
          while ((J GT 0) && (K LT Nsky-1) && $
            ( ((sky[L] - sky[J]) LT R) || ((sky[K] - sky[M]) LT R))) do begin
             J--
             K++
        endwhile
        endif
   skymed = total(sky[j:k])/(k-j+1)

;  If the mean is less than the median, then the problem of contamination
;  is slight, and the mean is what we really want.

   dmod = skymed LT skymn ?  3.*skymed-2.*skymn-skymod : skymn - skymod

; prevent oscillations by clamping down if sky adjustments are changing sign
   if dmod*old LT 0 then clamp = 0.5*clamp
   skymod = skymod + clamp*dmod
   old = dmod
   if redo then goto, START_LOOP

;
 skew = float( (skymn-skymod)/max([1.,sigma]) )
 nsky = maximm - minimm

 if keyword_set(DEBUG) or ( N_params() EQ 1 ) then begin
        print, '% MMM: Number of unrejected sky elements: ', strtrim(nsky,2), $
              '    Number of iterations: ',  strtrim(niter,2)
        print, '% MMM: Mode, Sigma, Skew of sky vector:', skymod, sigma, skew
 endif

 return
 end"""
