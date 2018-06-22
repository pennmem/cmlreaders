from cmlreaders import exc
from cmlreaders.base_reader import BaseCMLReader


class BaseReportDataReader(BaseCMLReader):
    """
        Reader class for classifier summary data produced in reporting pipeline

    Notes
    -----
    By default, a python class is returned. For report data read with this class
    a python object is the only supported return type. The returned class will
    be `ramutils.reports.summary.ClassifierSummary`

    """
    data_types = ["classifier_summary"]
    default_representation = 'pyobject'

    def __init__(self, data_type, subject, experiment, session, localization,
                 file_path=None, rootdir="/", **kwargs):
        super(BaseReportDataReader, self).__init__(data_type, subject=subject,
                                                   experiment=experiment,
                                                   session=session,
                                                   localization=localization,
                                                   file_path=file_path,
                                                   rootdir=rootdir)
        self.data_type = data_type

        try:
            from ramutils.reports.summary import ClassifierSummary
        except ImportError:
            raise exc.UnmetOptionalDependencyError("Install ramutils to use this reader")

        self.pyclass_mapping = {
            'classifier_summary': ClassifierSummary,
        }

    def as_pyobject(self):
        """ Return data as a python object specific to this data type """
        if self.data_type in self.pyclass_mapping:
            return self.pyclass_mapping[self.data_type].from_hdf(self._file_path)

    def as_dataframe(self):
        raise exc.UnsupportedRepresentation("Unable to represent this data as a dataframe")

    def to_hdf(self, file_name):
        pyobj = self.as_pyobject()
        pyobj.to_hdf(file_name)


class ReportSummaryDataReader(BaseReportDataReader):
    """
        Reader class for session and math summary data produced in the reporting
        pipeline

    Notes
    -----
    By default, a python class is returned based on the type of data. It could
    be one of

    - `ramutils.reports.summary.MathSummary`
    - `ramutils.reports.summary.FRStimSessionSummary`

    """

    data_types = ["session_summary", "math_summary"]
    default_representation = "pyobject"

    def __init__(self, data_type, subject, experiment, session, localization,
                 file_path=None, rootdir="/", **kwargs):
        super(BaseReportDataReader, self).__init__(data_type, subject=subject,
                                                   experiment=experiment,
                                                   session=session,
                                                   localization=localization,
                                                   file_path=file_path,
                                                   rootdir=rootdir)
        self.data_type = data_type
        self.subject = subject
        self.experiment = experiment
        self.session = session

        try:
            from ramutils.reports.summary import FRStimSessionSummary, \
                MathSummary
            from ramutils.utils import is_stim_experiment
        except ImportError:
            raise exc.UnmetOptionalDependencyError("Install ramutils to use this reader")

        self.pyclass_mapping = {
            'math_summary': MathSummary,
            'fr_stim_summary': FRStimSessionSummary,
        }

        self.is_stim_experiment = is_stim_experiment

    def as_pyobject(self):
        if self.data_type == 'math_summary':
            return super(ReportSummaryDataReader, self).as_pyobject()

        stim_experiment = self.is_stim_experiment(self.experiment)

        # TODO: Loading record-only data is a bit more complicated since it is
        # not tied to a particular session
        if not stim_experiment:
            raise exc.UnsupportedExperimentError("Only stim report data is currently supported. The readers in ramutils can still be used")

        summary_obj = self.pyclass_mapping['fr_stim_summary']

        return summary_obj.from_hdf(self._file_path)

    def as_dataframe(self):
        pyobj = self.as_pyobject()
        return pyobj.to_dataframe()
