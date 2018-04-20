Computation Memory Lab Data Readers
===================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Finding Files on RHINO
----------------------
The :class:`cml_data_readers.PathFinder` class can be used to identify the location of various
file types on RHINO. In an ideal world, all historic data would be processed
to have consistent file names, locations, and types. However, because this has
not been the case and individuals analyzing the data have come to expect and
deal with these inconsistencies, the safer approach is to leave the data in
its original form and attempt to abstract away these underlying inconsistencies
for future users.

.. autoclass:: cml_data_readers.PathFinder
    :members:


Path and File Constants
-----------------------
:class:`cml_data_readers.PathFinder` internally uses the :mod:`cml_data_readers.constants`
module. The usefulness of :class:`cml_data_readers.PathFinder` relies on these
constants being well-maintained.

