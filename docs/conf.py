# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from importlib.metadata import version as get_version

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "ysphotutilpy"
copyright = "2024, Yoonsoo P. Bach"
author = "Yoonsoo P. Bach"

try:
    release = get_version("ysphotutilpy")
    version = ".".join(release.split(".")[:2])
except Exception:
    release = "0.0.0"
    version = "0.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_copybutton",
    "numpydoc",
]

# Napoleon settings for NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = False
napoleon_use_rtype = True
napoleon_preprocess_types = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True,
    "ignore-module-all": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Mock imports for modules that might not be available
# Only mock truly optional or hard-to-install C-extensions not in dependencies
autodoc_mock_imports = [
    "sep",
]

# Diagnostic to check if the package is importable during the build
try:
    import ysphotutilpy
    print(f"DEBUG: Successfully imported ysphotutilpy {ysphotutilpy.__version__ if hasattr(ysphotutilpy, '__version__') else 'unknown'}")
except Exception as e:
    print(f"DEBUG: Failed to import ysphotutilpy: {e}")

# Autosummary
autosummary_generate = True

# Default role for backticks (allows `PanSTARRS1` instead of :obj:`PanSTARRS1`)
default_role = "obj"

# -- Intersphinx configuration -----------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "photutils": ("https://photutils.readthedocs.io/en/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "ccdproc": ("https://ccdproc.readthedocs.io/en/latest/", None),
    "astroquery": ("https://astroquery.readthedocs.io/en/latest/", None),
}

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
}

# Suppress warnings for missing references in external packages
nitpicky = False

# -- Templates ---------------------------------------------------------------
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Source suffix -----------------------------------------------------------
source_suffix = ".rst"
master_doc = "index"

