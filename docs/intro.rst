.. _intro:

************
Introduction
************

Purpose
========
CMLReaders is a library for easily finding, loading, and saving data types that
are specific to the Computational Memory Lab at the University of Pennsylvania.
It abtracts away the need to remember a multitude of file locations, prevents
the need for analysis code to be littered with try/except blocks, and unifies
the API for loading and saving lab specific data.

Features
========
* Easily find the location of a file on RHINO
* Supports loading  lab-specific data into memory in analysis-friendly formats
* Convert from one file output format to another
* Single API for loading/saving data. No need to use data-specific readers

Description
===========
The :mod:`cmlreaders` library contains a number of lab-specific data readers.
However, for the vast majority of use cases, users should be able to use
:class:`cmlreaders.readers.CMLReader` to load and save data. All readers
implement the interface defined in :class:`cmlreaders.readers.BaseCMLReader`,
which includes methods to load data into memory as a dictionary, recarray, or
Pandas dataframe. There is a save method corresponding to each of the loaders:
Each format includes a method for saving, i.e. JSON, CSV, and HDF5. It is not
guaranteed that all loading/saving methods will be implemented for each reader.
Internally, the readers use :class:`cmlreaders.PathFinder` to identify where
files are located.

