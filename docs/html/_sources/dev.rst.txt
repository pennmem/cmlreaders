Developer's Guide
=================

Adding new readers
------------------

New readers are added by extending :class:`cmlreaders.readers.BaseCMLReader` and
implementing one or more of the ``as_xyz`` methods. The default output format
when calling ``load`` is set by using the class variable ``default_representation``
which defaults to ``dataframe``. For example, say you want to create a new
reader that defaults to using a ``dict`` as output, the minimum that needs to be
done is to override ``as_dict`` and setting that as the default:

.. code-block:: python

    class MyReader(BaseCMLReader):
        default_representation = 'dict'

        def as_dict(self):
            return {'for': 'great justice'}

Once the reader works, it must be enabled in the general :class:`cmlreaders.CMLReader`:

1. Add an entry for the class to the :attr:`cmlreaders.reader` dict
2. Add a test in :mod:`cmlreaders.test.test_cmlreader`
