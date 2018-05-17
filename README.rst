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


EEG reading API proposal
------------------------

A proposed API for reading EEG looks like the following:

.. code-block:: python

   from cmlreaders import CMLReader, RereferencingNotPossibleError

   # create a reader for one subject/experiment/session
   # this abstracts away where things are stored; a root path may need to be
   # specified
   reader = CMLReader(subject='R1111M', experiment='FR1', session=0)

   # load events as a DataFrame and filter for some event type
   events = reader.load('events')
   word_events = events[events.type == 'WORD']

   # get electrode contact info as a DataFrame
   # this will have contact labels, locations, regions, coordinates, etc.
   contacts = reader.load('contacts')

   # load eeg data for WORD events
   # pre/post: milliseconds before/after event onset/offset to include
   eeg = reader.load('eeg', events=word_events, pre=100, post=100)

   # or load all eeg data using all recorded channels
   all_eeg = reader.load('eeg')

   # or specify only contacts that are located in the MTL
   # require_monopolar will raise an exception if monopolar is not possible
   subset_eeg = reader.load('eeg',
                            contacts=contacts[contacts.region == 'MTL'],
                            require_monopolar=True)

   # get pairs from neurorad pipeline or whatever
   pairs = reader.load('pairs')

   # try to re-reference
   try:
       reref = eeg.rereference(pairs)
   except RereferencingNotPossibleError:
       print("oops, this was recorded in bipolar mode")


The idea here is to use pandas DataFrames wherever possible for all the benefits
they give us (selection, saving to lots of different formats, etc.). The one
exception is the EEG data which will be returned as something like a
:class:`ptsa.data.TimeSeries` object.


Testing
-------

Since this repository is specific to the data formats of the lab, almost all
tests require RHINO access. To run the test suite from a computer with RHINO
mounted:

.. code-block:: shell-sessions

    pytest cmlreaders/ --rhino-root [path_to_mount_point] --cov=html

Upon completion, the coverage report will be saved into htmlcov/ in the top
level directory of the project.


