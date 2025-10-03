# Changelog

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