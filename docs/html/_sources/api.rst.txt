.. _api:

===
API
===

CML Reader
==========

.. autoclass:: cmlreaders.CMLReader
    :members:

Custom Readers
==============

.. automodule:: cmlreaders.readers.readers
    :members:

.. automodule:: cmlreaders.readers.eeg
    :members:

PathFinder
==========

The :class:`cmlreaders.PathFinder` class can be used to identify the location of
various file types on RHINO. In an ideal world, all historic data would be
processed to have consistent file names, locations, and types. However, because
this has not been the case and individuals analyzing the data have come to
expect and deal with these inconsistencies, the safer approach is to leave the
data in its original form and attempt to abstract away these underlying
inconsistencies for future users.

.. autoclass:: cmlreaders.PathFinder
    :members:


Path and File Constants
-----------------------

:class:`cmlreaders.PathFinder` internally uses the :mod:`cmlreaders.constants`
module. The usefulness of :class:`cmlreaders.PathFinder` relies on these
constants being well-maintained.



