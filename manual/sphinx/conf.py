#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# BOUT++ documentation build configuration file, created by
# sphinx-quickstart on Mon Jan 23 17:21:15 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import argparse

try:
    from breathe import apidoc

    has_breathe = True
except ImportError:
    print("breathe module not installed")
    has_breathe = False

import os
import subprocess
import sys

sys.path.append("../../tools/pylib")

# Are we running on readthedocs?
on_readthedocs = os.environ.get("READTHEDOCS") == "True"

if on_readthedocs:
    from unittest.mock import MagicMock

    class Mock(MagicMock):
        __all__ = [
            "foo",
        ]

        @classmethod
        def __getattr__(cls, name):
            return MagicMock()

    MOCK_MODULES = [
        "netCDF4",
        "mayavi2",
        "enthought",
        "enthought.mayavi",
        "enthought.mayavi.scripts",
        "enthought.tvtk",
        "enthought.tvtk.api",
        "scipy",
        "scipy.ndimage",
        "scipy.interpolate",
        "scipy.integrate",
        "tvtk",
        "tvtk.tools",
        "tvtk.api",
        "scipy.ndimage.filters",
        "scipy.ndimage.morphology",
        "scipy.spatial",
        "past",
        "crosslines",
    ]
    sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)
    print(os.environ)
    print(sys.argv)
    python = sys.argv[0]
    pydir = "/".join(python.split("/")[:-2])
    os.system("which clang-format")
    os.system("which clang-format-6.0")
    os.system(
        "git submodule update --init --recursive ../../externalpackages/mpark.variant"
    )
    pwd = "/".join(os.getcwd().split("/")[:-2])
    os.system("git submodule update --init --recursive ../../externalpackages/fmt")
    cmake = (
        "cmake  . -DBOUT_USE_FFTW=ON"
        + " -DBOUT_USE_LAPACK=OFF"
        + " -DBOUT_ENABLE_PYTHON=ON"
        + " -DBOUT_UPDATE_GIT_SUBMODULE=OFF"
        + " -DBOUT_TESTS=OFF"
        + " -DBOUT_ALLOW_INSOURCE_BUILD=ON"
        + f" -DPython3_ROOT_DIR={pydir}"
        + f" -Dmpark_variant_DIR={pwd}/externalpackages/mpark.variant/"
        + f" -Dfmt_DIR={pwd}/externalpackages/fmt/"
    )
    # os.system("mkdir ../../build")
    os.system("echo " + cmake)
    x = os.system("cd ../.. ;" + cmake)
    assert x == 0
    x = os.system("cd ../.. ; make -j 2 -f Makefile")
    assert x == 0


# readthedocs currently runs out of memory if we actually dare to try to do this
if has_breathe:
    # Run doxygen to generate the XML sources
    if on_readthedocs:
        subprocess.call("cd ../doxygen; doxygen Doxyfile_readthedocs", shell=True)
    else:
        subprocess.call("cd ../doxygen; doxygen Doxyfile", shell=True)
    # Now use breathe.apidoc to autogen rst files for each XML file
    apidoc_args = argparse.Namespace(
        destdir="_breathe_autogen/",
        dryrun=False,
        force=True,
        notoc=False,
        outtypes=("file"),
        project="BOUT++",
        rootpath="../doxygen/bout/xml",
        suffix="rst",
        members=True,
        quiet=False,
    )
    apidoc_args.rootpath = os.path.abspath(apidoc_args.rootpath)
    if not os.path.isdir(apidoc_args.destdir):
        if not apidoc_args.dryrun:
            os.makedirs(apidoc_args.destdir)
    apidoc.recurse_tree(apidoc_args)
    for key, value in apidoc.TYPEDICT.items():
        apidoc.create_modules_toc_file(key, value, apidoc_args)

    # -- Options for breathe extension ----------------------------------------

    breathe_projects = {"BOUT++": "../doxygen/bout/xml"}
    breathe_default_project = "BOUT++"
    breathe_default_members = ("members",)

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
]

if has_breathe:
    extensions.append("breathe")

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# How to parse markdown files
try:
    from recommonmark.parser import CommonMarkParser

    source_parsers = {".md": CommonMarkParser}
except ImportError:
    print("recommonmark module not installed")
    source_parsers = {}

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "BOUT++"
author = "B. Dudson and The BOUT++ team"
copyright = f"2017-2023, {author}"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = "5.1"
# The full version, including alpha/beta/rc tags.
release = "5.1.1"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# Tell sphinx what the primary language being documented is.
primary_domain = "cpp"

# Tell sphinx what the pygments highlight language should be.
highlight_language = "cpp"

# Turn on figure numbering
numfig = True

# The default role for text marked up `like this`
default_role = "any"

# Handle multiple parameters on one line correctly (in Python docs)
napoleon_use_param = False

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = dict(
    repository_url="https://github.com/boutproject/BOUT-dev",
    repository_branch="master",
    path_to_docs="manual/sphinx",
    use_edit_page_button=True,
    use_repository_button=True,
    use_issues_button=True,
    home_page_in_toc=False,
)


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


def setup(app):
    app.add_css_file("css/custom.css")


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "BOUTdoc"


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    "preamble": r"""
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\makeatletter
\def\UTFviii@defined#1{%
  \ifx#1\relax
      ?%
  \else\expandafter
    #1%
  \fi
}
\makeatother


""",
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "BOUT.tex", "BOUT++ Documentation", "The BOUT++ team", "manual"),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "bout", "BOUT++ Documentation", [author], 1)]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "BOUT",
        "BOUT++ Documentation",
        author,
        "BOUT",
        "One line description of project.",
        "Miscellaneous",
    ),
]
