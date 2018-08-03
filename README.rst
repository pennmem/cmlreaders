CML Data Readers
================

.. image:: https://travis-ci.org/pennmem/cmlreaders.svg?branch=master
    :target: https://travis-ci.org/pennmem/cmlreaders

.. image:: https://codecov.io/gh/pennmem/cmlreaders/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/pennmem/cmlreaders

.. image:: https://img.shields.io/badge/docs-here-brightgreen.svg
   :target: https://pennmem.github.io/pennmem/cmlreaders/html/index.html
   :alt: docs

This repository contains a collection of utility classes for reading
lab-specific file formats. Many of these readers have previously existed in
other repositories, but are being migrated to a single repository since they
are logically related.

More information and usage examples are provided in the documentation_.

.. _documentation: https://pennmem.github.io/cmlreaders/html/index.html


Installation
------------

The following assumes you have already installed Anaconda_ or Miniconda_.

.. _Anaconda: https://www.anaconda.com/distribution/
.. _Miniconda: https://conda.io/miniconda.html

.. code-block:: shell-session

    conda create -y -n environment_name python=3
    source activate environment_name
    conda install -c pennmem cmlreaders

.. note::

    In some installations using Anaconda (as opposed to miniconda), it may be
    necessary to also install the ``bottleneck`` package::

        conda install -y bottleneck


Usage examples
--------------

What follows are some basic usage examples. See the documentation for a more
complete guide to getting started.

Get the index of all RAM data:

.. code-block:: python

    >>> from cmlreaders import CMLReader
    >>> ix = CMLReader.get_data_index("r1")

Find unique experiments performed by R1111M:

.. code-block:: python

    >>> ix[ix["subject"] == "R1111M"]["experiment"].unique()
    array(['FR1', 'FR2', 'PAL1', 'PAL2', 'PS2', 'catFR1'], dtype=object)

Load montage pair data from FR1 session 0:

.. code-block:: python

    >>> reader = CMLReader("R1111M", "FR1", 0)
    >>> pairs = reader.load("pairs")
    >>> pairs.columns()
    Index(['contact_1', 'contact_2', 'label', 'is_stim_only', 'type_1', 'type_2',
           'avg.dural.region', 'avg.dural.x', 'avg.dural.y', 'avg.dural.z',
           'avg.region', 'avg.x', 'avg.y', 'avg.z', 'dk.region', 'dk.x', 'dk.y',
           'dk.z', 'id', 'ind.dural.region', 'ind.dural.x', 'ind.dural.y',
           'ind.dural.z', 'ind.region', 'ind.x', 'ind.y', 'ind.z', 'is_explicit',
           'stein.region', 'stein.x', 'stein.y', 'stein.z', 'tal.region', 'tal.x',
           'tal.y', 'tal.z', 'wb.region', 'wb.x', 'wb.y', 'wb.z'],
          dtype='object')


Select word onset events:

.. code-block:: python

    >>> events = reader.load("events")
    >>> words = events[events["type"] == "WORD"]
    >>> len(words)
    288

Load EEG 100 ms before and after word onset using the bipolar referencing
scheme:

.. code-block:: python

    >>> eeg = reader.load_eeg(events=words, rel_start=-100, rel_stop=100, scheme=pairs)
    >>> eeg.data.shape
    (288, 141, 100)


Testing
-------

Since this repository is specific to the data formats of the lab, almost all
tests require RHINO access. To run the test suite from a computer with RHINO
mounted:

.. code-block:: shell-session

    pytest cmlreaders/ --rhino-root [path_to_mount_point] --cov=html

Upon completion, the coverage report will be saved into htmlcov/ in the top
level directory of the project.
