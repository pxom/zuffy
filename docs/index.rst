.. Zuffy documentation master file, created by
   sphinx-quickstart on Tue May 27 20:50:01 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. raw: html

..   <table><tr><td><img style="float:left;padding-right:0px;vertical-align:top;border:none" src="https://raw.githubusercontent.com/pxom/zuffy/master/assets/zuffy_logo_small_nb_gr.png" alt="Zuffy Logo" width="80"/></td><td><h2>Zuffy - Fuzzy Pattern Trees with Genetic Programming</h2></td></tr></table>

**Version**: |version|  **Date**: |today| 

.. image:: ../assets/zuffy_logo_small_nb_gr.png
   :width: 80px
   :class: align-left
   :alt: Zuffy logo showing a tree in a lightbulb

Zuffy - Fuzzy Pattern Trees induced by Genetic Programming
==========================================================


An Open Source sci-kit compatible classifier.

This package is part of an academic research exercise and it is designed to help a
Machine Learning developer to explore Fuzzy Pattern Trees (FPT).

FPTs are models that have high explainability and they help the researcher
explore a dataset where vague or ambiguous data may be present.

This kit will analyse a dataset and fuzzify the input features to generate
a collection of fuzzy sets.  These sets become new features which are 
combined with fuzzy operators to build a pattern tree.

The package uses Genetic Programming to induce the tree and it requires the
SymbolicClassifer from the open source gplearn package to support the 
genetic evolution process.


.. This is the root of the documentation and all significant help should be reached from links on this page.
   Decide on the formatting style:
   This is ``red text with a grey frame around it``.

   What other `this is in italics` options are there?

   How do I write in **bold**, ``strikethrough``, :sup:`Super` Script *italics* and show [code examples]?

   How to show an image?


   .. https://github.com/ericholscher/sphinx-tutorial/blob/master/cheatsheet.rst


   Literal code block::

      Contents are indented.

      ::

      The :: marker is omitted here.

   =======
   Level 1
   =======

   Level 2
   -------

   Level 1a
   ^^^^^^^^

   Level 2a
   ++++++++

   .. .. [cit1] A global citation.

   .. attention:: Attention

      Attention, Attention, Attention!

   .. warning:: Warning!

      Warning, Warning!


   Add your content using ``reStructuredText`` syntax. See the
   `reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
   documentation for details.

   Table of Contents appears below here.

..   source/index
   source/quick_start
   source/user_guide
   

.. toctree::
   :maxdepth: 10
   :caption: Contents:

   source/zuffy
   source/zuffy_fit_iterator
   source/fuzzy_transformer
   source/visuals
   source/_visuals_color_assignment
   source/_fpt_operators
   source/Examples
