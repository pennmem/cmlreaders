CML Data Readers
================

.. image:: https://img.shields.io/travis/pennmem/cml_data_readers.svg
   :target: https://travis-ci.org/pennmem/cml_data_readers

.. image:: https://codecov.io/gh/pennmem/cml_data_readers/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/pennmem/cml_data_readers

.. image:: https://img.shields.io/badge/docs-here-brightgreen.svg
   :target: https://pennmem.github.io/pennmem/cml_data_readers/html/index.html
   :alt: docs

This repository contains a collection of utility classes for reading
lab-specific file formats. Many of these readers have previously existed in
other repositories, but are being migrated to a single repository since they
are logically related.

This is a work in progress and should be used with caution until the v1.0
release as the API will likely undergo numerous changes up until that point.


Installation
------------

.. code-block:: shell-session

    conda create -y -n environment_name python=3
    source activate environment_name
    conda install -c pennmem cml_data_readers


Testing
-------

Since this repository is specific to the data formats of the lab, almost all
tests require RHINO access. To run the test suite from a computer with RHINO
mounted:

.. code-block:: shell-sessions

    pytest cml_data_readers/ --rhino-root [path_to_mount_point] --cov=html

Upon completion, the coverage report will be saved into htmlcov/ in the top
level directory of the project.


API Proposal
------------
The goal of the cml_data_readers package is twofold:

1. Abstract away the inconsistencies of data stored on RHINO. In particular,
naming conventions, locations, and file types have changed over time. To the
greatest extent possible, end users should be allowed to remain oblivious to
these inconsistencies. This will be achieved primarily through the
:class:`PathFinder`

2. Provide a unified API for reading/writing data that is lab-specific. In its
current state, lab members use a variety of different readers largely located
in PTSA:
    - JsonIndexReader
    - CMLEventReader
    - H5RawReader (if the eeg storage format is CML specific)
    - LocReader
    - Tal Reader

From an end-user perspective, interacting with CML-specific data formats would
require:
1. Instantiating a generic reader with information about the type of data
to be read, i.e. subject/experiment/session/montage/etc.
2. Loading the desired data, which returns a generic data object type
3. Converting the generic data object type into the desired format

Internally, this generic reader class will use :class:`PathFinder` and the
data-specific readers to retrieve the data from disk and store in memory. To be
more concerete, a session could look like the following:

.. code-block:: python

    # subject, experiment, session given as examples. Additional or fewer
    # parameters could also be given including localization number,
    # montage number, etc.
    reader = CMLReader(subject=subject, experiment=experiment, session=session)

    # First step is to load the desired data. This is an exmample, many other
    # data types would be supported
    abstract_data_object = reader.load('events')

    # Second step is to get the data in memory in the desired format
    event_df = abstract_data_object.as_dataframe()
    event_json = abstract_data_object.as_json()
    event_csv = abstract_data_object.as_csv()

    # I really hesitate to even have this as an option since they should be
    # banned from use
    event_recarray = abstract_data_object.as_recarray()

    # Optional step is to output the data in a different format
    event_df.to_csv('filename')
    event_df.to_json('filename')
    event_df.to_hdf('filename')




