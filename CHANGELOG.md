# Changelog

## [0.3] (unreleased)

### New modules / functions
* `aputil` — new module with low-level aperture utilities that bypass `photutils` high-level overhead:
  * `fast_circ_apmask` — circular aperture overlap mask (~3x faster than `photutils` `BoundingBox`)
  * `fast_circ_anmask` — circular annulus overlap mask
  * `fast_circ_apanmask` — simultaneous aperture + annulus masks in a single bounding-box pass
  * `fast_ellip_apmask` — rotated elliptical aperture overlap mask
  * `fast_ellip_anmask` — rotated elliptical annulus overlap mask

### New modules / functions (continued)
* `seputil.sep_extract_iterative` — iterative background + extraction loop:
  runs `n_iter` rounds of `sep_back` → `sep_extract`, using the segmentation
  map from each pass (optionally dilated by `seg_dilate` pixels) as a source
  mask for the next background estimation.

### Bugfixes
* `seputil._sep_extract`: `seg_remove_mask` was applying `seg & ~mask` (bitwise
  AND on integer label array), which corrupted or zeroed label values for all
  sources. Fixed to use `seg[mask] = 0` (boolean index assignment).

### Major changes
* `radprof`
  * New `radcum_profile` function: cumulative radial profile using circular apertures, with optional variance/error propagation, pixel counting, and last-radius normalization.
  * New `ee_radius` function: finds the radius encircling a given fraction of total flux (encircled-energy radius) via Brent's method.
  * `ee_radius`: new `sum_is_unity` option — treats the image as already normalized so `fraction` is used directly as the target flux.
* `background.annul2values`
  * Fast path for `CircularAnnulus` via `fast_circ_anmask` (~1.4–1.5x faster).
  * Fast path for `EllipticalAnnulus` via `fast_ellip_anmask`.
  * Bugfix: no longer raises when the sky annulus contains no valid pixels.
* `background.quick_sky_circ`: now accepts `mask` and `**kwargs` forwarded to `sky_fit`.

### API Changes
* `radprof.radcum_profile`: new parameters `var`, `err`, `return_var`, `add_npix`, `norm_by_last`.

## [0.2.1]

### Major changes
* `background.sky_fit`
  * ``mode_option`` is removed. Instead of ``method="mode", mode_option="sex"``, use ``method="sex"``, etc.
  * Sky sigma-clipping can now be skipped. Use `sky_clipper` as a user-given function, `None`, or the default `utils.sigma_clipper`.
* `radprof.radial_profile`
  * New `add_center` (default False) option. Adds (r, y) = (0, center_pixel) for convenience.
  * Fix: standard deviation could be negative when `norm_by_center` is `True` and central pixel value is negative. Now fixed by using the absolute value of central pixel.


## [0.2]

### Major changes

* Uses `photutils` >= 2.0
  * There were some breaking changes made in `photutils`, so `ysphotutilpy` is now incompatible with previous versions, unfortunately. This includes Aperture objects, SourceGrouper, etc.
*

### API Changes
* `queryutil.organize_ps1_and_isnear`: Argument changed from `group_min_separation` → `group_minsep`
* all `crit_separation` used for the old DAOGROUP are now renamed to `min_separation`.