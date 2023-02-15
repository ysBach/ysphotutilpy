# ysphotutilpy
[![DOI](https://zenodo.org/badge/159082834.svg)](https://zenodo.org/badge/latestdoi/159082834)

Simple photometry utilities.


Install by

```
$ pip install ysphotutilpy
```

or

```
$ cd <where you want to download this package>
$ git clone https://github.com/ysBach/ysphotutilpy
$ cd ysphotutilpy
$ git pull && pip install -e .
```
From the second time, **just run the last line**.


**NOTE**: Please understand this package may undergo severe backward-incompatible changes. This is a package I use for (1) my own research and (2) education (see, e.g., [SNU_AOclass](https://github.com/ysBach/SNU_AOclass/)).

For (1): I always include a proper "snapshot" of this package to the publication, so any backward incompatible changes won't affect the reproducibility of the published results.

For (2): Each semester, students will try to download/install the newest versions of other packages (especially ``ccdproc``, ``photutils``, etc). But these packages sometimes introduce "breaking changes" to their source codes, and thus, I have to modify my codes to "work" with the newest versions of such packages (e.g., as of 2022 Apr, I suddenly had to drop ``photutils <= 1.3`` due to its [internal API change](https://github.com/astropy/photutils/issues/1335) in version 1.4 & as of 2022 Dec, I also suddenly had to drop ``photutils <= 1.6`` due to the [same reason](https://github.com/astropy/photutils/commit/799e0b0aca361b8deb5f506a91af1c890075af77)). Thus, backward incompatible changes are inevitable for me.

My justification is that this is not an "officially recommended" package by, e.g., STScI, but rather it is just a personal toolbox. Even for the name of this package, I tried not to occupy "photutilpy", which must be used by better-organized ones (such as photutils).

Enjoy astronomical data reduction!

Some useful urls:

* PS1 Different table columns: [PS1 DB obj&det tables](https://outerspace.stsci.edu/display/PANSTARRS/PS1+Database+object+and+detection+tables)
  * Among these, [PS1 Mean table](https://outerspace.stsci.edu/display/PANSTARRS/PS1+MeanObjectView+table+fields) is the most widely used (default in MAST query).
  * [Explanation on ``objInfoFlag`` column](https://outerspace.stsci.edu/display/PANSTARRS/PS1+Object+Flags#PS1ObjectFlags-ObjectInfoFlagsvalues,e.g.,columnobjInfoFlagintableObjectThin)