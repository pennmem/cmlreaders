Developer's Guide
=================

Adding new search paths
-----------------------

All of the file path information is contained in the ``rhino_paths`` dictionary
contained within :mod:`cmlreaders.constants`. The keys of the dictionary are the
supported data types and the values are python lists whose elements correspond
to the possible locations of where the data can be found. Paths within the list
should appear in the preferred search order since
:class:`cmlreaders.path_finder.PathFinder` is set up to return the first path
where an existing file is found. When adding a new search path, place the
new path in the desired location within the list of search paths.

Adding new data types
---------------------

To add support for a new data type, first add the data type shortcut name
to :mod:`cmlreaders.constants` with a list of possible locations on rhino
as the value. Next, add this data type to the appropriate list:

- subject_files
- localization_files
- montage_files
- session_files
- host_pc_files
- used_classifier_files

These lists are used by :class:`cmlreaders.path_finder.PathFinder`.

Ideally, a reader already exists that can manage the new data
type. If this is the case, add the new data type to the ``data_types`` class
member of the reader class that should be used. If a new reader is required,
see the following section.

Adding new readers
------------------

New readers are added by extending :class:`cmlreaders.readers.BaseCMLReader` and
implementing one or more of the ``as_xyz`` methods. The default output format
when calling ``load`` is set by using the class variable ``default_representation``
which defaults to ``dataframe``. For example, say you want to create a new
reader that defaults to using a ``dict`` as output and should be used for some
data type, X. At a minimum, you will need to define a ``data_types`` list that
contains X, and set ``default_representation`` to ``dict``. If there are
additional data types that should use this reader, those should also be added
to the ``data_types`` list.

.. code-block:: python

    class MyReader(BaseCMLReader):
        data_types = ['X']
        default_representation = 'dict'

        def as_dict(self):
            return {'for': 'great justice'}

Once the reader works, test cases for the data types using the new reader
should be added to :mod:`cmlreaders.test.test_cmlreader`. These are in addition
to the test cases that should already exist for the new reader. For examples,
see :mod:`cmlreaders.test.test_readers`.



