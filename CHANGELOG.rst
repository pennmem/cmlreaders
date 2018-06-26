Changes
=======

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

