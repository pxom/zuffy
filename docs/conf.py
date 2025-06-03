import os
import sys
sys.path.insert(0, os.path.abspath('..')) # Go up one directory from docs
from importlib.metadata import version as get_version

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Zuffy'
copyright = "2025, P O'Mahony"
author = "P O'Mahony"
release = get_version('zuffy')
version = ".".join(release.split(".")[:3])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# start pom additions

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon', # If you use Google/NumPy style docstrings
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',   

    "sphinx.ext.autosummary",
    #"sphinx_design",
    #"sphinx-prompt",
    #"sphinx_gallery.gen_gallery",
    #"numpydoc",     
]
html_theme = 'sphinx_rtd_theme'

napoleon_google_docstring = True # Set to False if using NumPy style
napoleon_numpy_docstring = False # Set to True if using NumPy style
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# end pom additions

templates_path = ['_templates']
exclude_patterns = ["_build", "_templates", "Thumbs.db", ".DS_Store"]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

