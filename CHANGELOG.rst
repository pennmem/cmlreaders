Changes
=======

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

