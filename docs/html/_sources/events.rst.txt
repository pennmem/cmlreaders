Events
======

Experimental events are stored in a tabular format with a set of common fields.
These common fields are supplemented by a number of other fields that may be
specific to a particular experiment or common amongst a subset of experiments.

Common fields
-------------

The following fields should be available in events of all experiment types:

.. csv-table::
    :file: events_common.csv
    :header-rows: 1
    :widths: auto

Stimulation parameters
^^^^^^^^^^^^^^^^^^^^^^

As noted above, stimulation parameters are stored as a list of dictionaries.
Each dictionary contains the following fields:

.. csv-table::
    :file: stim_params.csv
    :header-rows: 1
    :widths: auto

Some additional fields may also exist, including burst fields which are not
often used (``burst_freq``, ``n_bursts``) and ``stim_on`` which is 1 for an
event with stimulation and 0 otherwise.

FR events
---------

Event types that can be found in most FR-like experiments are:

.. csv-table::
    :file: fr_event_types.csv
    :header-rows: 1
    :widths: auto

Fields common in most FR-like experiments include:

.. csv-table::
    :file: fr_event_fields.csv
    :header-rows: 1
    :widths: auto
