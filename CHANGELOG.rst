Changes
=======

Upcoming
--------

* Significantly improved read speed for split EEG data (#137)


Version 0.8.0
-------------

**2018-07-19**

* Added support for loading pyFR data (#117)
* Simplified EEG loading by removing the option to load directly via epochs
  (#125)
* Renamed the class holding results from ``CMLReader.load_eeg`` to
  ``EEGContainer`` to avoid confusion with the PTSA ``TimeSeries`` class (#126)
* Added a new ``CMLReader.load_events`` classmethod to load events from
  multiple subjects and/or experiments (#129)
* Added support for loading multisession EEG data (#130)


Version 0.7.2
-------------

**2018-07-17**

* Improved conversion of EEG data to PTSA format (#107)
* Fixed loading events for PS2 and PS4 (#110, #112)
* Improved error message when trying to load EEG with an empty events DataFrame
  (#114)


Version 0.7.1
-------------

**2018-07-12**

New feature:

* Results of ``get_data_index`` are now cached using ``functools.lru_cache``
  (#101).

Bug fix:

* Magic importing of reader classes didn't work if not in a specific working
  directory (#104). Fixed in PR #105.


Version 0.7.0
-------------

**2018-07-06**

User-facing changes:

* Localization and montage numbers are now converted to integers instead of
  being strings (#91)
* Fixed loading of montage data for subjects with a montage number other than
  0 (#95)
* Added preliminary support for loading ltp data (#97)

Other changes:

* CI testing updated to use an environment variable to specify what Python
  version to run (#93)
* Test data gets written to a temporary directory instead of polluting the
  ``cmlreaders.test.data`` package (#96)
* Reader classes are automatically discovered instead of having to specify them
  in ``cmlreaders/readers/__init__.py`` (#99)


Version 0.6.0
-------------

**2018-06-29**

This release fixes several bugs with EEG reading when passing a referencing
scheme and improves performance when loading pairs/contacts data. Highlights:

* Adds and improves existing test cases for rereferencing EEG data
* Improved load speed of ``pairs.json``/``contacts.json`` by about 2 orders of
  magnitude (#89)
* Speeds up loading of split EEG data when specifying a referencing scheme by
  only loading the required data (#85)


Version 0.5.0
-------------

**2018-06-26**

New features:

* Allow globally setting the root directory with an environment variable (#46)
* Added a function to check if EEG data can be rereferenced
* Automatically determine montage and localization numbers when possible (#77)
* Added a ``fromfile`` method to classes based on ``BaseCMLReader`` to more
  easily directly load specific data types (#79)

Improvements:

* Added support for reading EEG data from restarted sessions (#68)
* Improved the ergonomics of passing a ``scheme`` keyword argument to
  rereference EEG data (#70)
* Make channel filtering via the ``scheme`` keyword argument more explicit (#80)

Bug fixes:

* Handle loading PS4 events (#47)
* Fixed paths with respect to montage/localization confusion (#62)
* Fixed the ``CSVReader`` to correctly read jacksheets (#65)
* Handle gaps in contact numbers when reading EEG data (#63)


Version 0.4.0
-------------

**2018-06-06**

* Implemented custom TimeSeries representation that can be converted to PTSA
  or MNE format
* Implemented EEG reader with support for loading a full session or event-based
  subset
* Updated getting started guide and documentation


Version 0.3.1
-------------

**2018-05-24**

* Minor bugfix to allow conda package to build correctly

Version 0.3.0
-------------

**2018-05-24**

* Updated API to use .load() and .get_reader()
* Added Json, Montage, Localization, Event, Classifier, ReportData, and
  ElectrodeCateogry readers
* Refactored base reader class to use a metaclass for registering new readers

Version 0.2.0
-------------

**2018-05-15**

* Implemented basic Text and CSV readers
* Somewhat stabilized the API/internals
* Improved error message when files are not found

Version 0.1.1
-------------

**2018-04-26**

* Minor API/name changes
* Renamed package for Pep8 compliance

Version 0.1.0
-------------

**2018-04-20**

* Initial alpha release for internal developer use

