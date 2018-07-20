Shortcuts for common queries
============================

Since most data other than EEG data can be naturally represented by a pandas
:class:`DataFrame`, querying and selecting subsets of data is generally very
easy. Nevertheless, common queries can be tedious to have to write out every
time, so shortcuts exist in the form of custom pandas accessors__.

__ https://pandas.pydata.org/pandas-docs/stable/extending.html

For example, we can get all word events for some events either by directly
masking an events :class:`DataFrame` or by using the shortcut accessor
``.events.words``:

.. code-block:: ipython

    In [1]: from cmlreaders import CMLReader

    In [2]: subjects = ["R1111M", "R1286J"]

    In [3]: experiments = ["FR1"]

    In [4]: all_events = CMLReader.load_events(subjects, experiments)

    In [5]: all_events[all_events["type"] == "WORD"][:2]
    Out[5]:
        answer                    eegfile  eegoffset  ...  subject       test  type
    27    -999  R1111M_FR1_0_22Jan16_1638     100520  ...   R1111M  [0, 0, 0]  WORD
    28    -999  R1111M_FR1_0_22Jan16_1638     101829  ...   R1111M  [0, 0, 0]  WORD

    [2 rows x 24 columns]

    In [6]: all_events.events.words[:2]
    Out[6]:
        answer                    eegfile  eegoffset  ...  subject       test  type
    27    -999  R1111M_FR1_0_22Jan16_1638     100520  ...   R1111M  [0, 0, 0]  WORD
    28    -999  R1111M_FR1_0_22Jan16_1638     101829  ...   R1111M  [0, 0, 0]  WORD

    [2 rows x 24 columns]


Available accessors
-------------------

Upon importing :mod:`cmlreaders`, the following accessors are automatically
registered with pandas.

.. autoclass:: cmlreaders._accessors.events.EventsAccessor
    :members:
