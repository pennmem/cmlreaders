.. _intro:

************
Introduction
************

Purpose
========

CMLReaders is a library for easily finding, loading, and saving data that is
specific to the Computational Memory Lab at the University of Pennsylvania.
It abstracts away the need to remember a multitude of file locations, prevents
the need for analysis code to be littered with try/except blocks, and unifies
the API for loading lab specific data. For more information on lab
specific data types, see the :ref:`data_guide` section.

Features
========

* Easily find the location of a file on RHINO
* Single API for loading data
* Convert from one file output format to another

Description
===========

The :mod:`cmlreaders` library contains a number of lab-specific data readers.
However, for the vast majority of use cases, users should be able to use
:class:`cmlreaders.readers.CMLReader` to load data.

In situations where the user would like to load data into a non-default
format or convert between different output formats, the individual readers can
be used directly. All readers are built on top of
:class:`cmlreaders.readers.BaseCMLReader`. These readers implement various
as_x() methods to load data into memory using different representations such
as a dictionary, dataframe, recarray, python class, etc. Readers also implement
different to_y() methods for saving data to CSV, JSON, or HDF5 output formats.
Different readers implement different loading and saving methods, so consult the
:ref:`api` documentation for more information.

