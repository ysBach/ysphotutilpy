# Changelog

## [0.2.1]

### Major changes
* `background.sky_fit`
  * ``mode_option`` is removed. Instead of ``method="mode", mode_option="sex"``, use ``method="sex"``, etc.
  * Sky sigma-clipping can now be skipped. Use `sky_clipper` as a user-given function, `None`, or the default `utils.sigma_clipper`.
*

## [0.2]

### Major changes

* Uses `photutils` >= 2.0
  * There were some breaking changes made in `photutils`, so `ysphotutilpy` is now incompatible with previous versions, unfortunately. This includes Aperture objects, SourceGrouper, etc.
*

### API Changes
* `queryutil.organize_ps1_and_isnear`: Argument changed from `group_min_separation` → `group_minsep`
* all `crit_separation` used for the old DAOGROUP are now renamed to `min_separation`.