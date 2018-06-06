CML Data Readers
================

.. image:: https://img.shields.io/travis/pennmem/cmlreaders.svg
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

This is a work in progress and should be used with caution until the v1.0
release as the API will likely undergo numerous changes up until that point.

More information and usage examples are provided in the documentation_.

.. _documentation: https://pennmem.github.io/cmlreaders/html/index.html


Installation
------------

.. code-block:: shell-session

    conda create -y -n environment_name python=3
    source activate environment_name
    conda install -c pennmem cmlreaders

Main Features
-------------
* Easily find the location of a file on RHINO
* Single API for loading CML-specific data
* Convert from one file output format to another


Testing
-------

Since this repository is specific to the data formats of the lab, almost all
tests require RHINO access. To run the test suite from a computer with RHINO
mounted:

.. code-block:: shell-session

    pytest cmlreaders/ --rhino-root [path_to_mount_point] --cov=html

Upon completion, the coverage report will be saved into htmlcov/ in the top
level directory of the project.


