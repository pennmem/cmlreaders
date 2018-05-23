from typing import Optional
from . import readers


__all__ = ['CMLReader']


class CMLReader(object):
    """ Generic reader for all CML-specific files

    Notes
    -----
    At import, all the readers from :mod:`cmlreaders.readers` will register the
    data types that should correspond to that reader by updating the
    reader_names dictionary. reader_names is a dict whose keys are one of
    the data types understood by :class:`cmlreaders.PathFinder` and defined in
    :mode:`cmlreaders.constants`. Values are the name of the reader class
    that should be used for loading/reading the data type. When an instance of
    :class:`cmlreaders.cmlreader.CMLReader` is instantiated, a new dictionary is
    created that maps the data types to the actual reader class, rather than
    just the class name. In essence, :class:`cmlreaders.cmlreader.CMLReader` is
    a factory that routes the requests for loading a particular data type to
    the reader defined to handle that data.

    """
    reader_names = {}

    def __init__(self, subject: Optional[str] =None,
                 experiment: Optional[str] = None,
                 session: Optional[int] = None,
                 localization: Optional[int] = 0,
                 montage: Optional[int] = 0,
                 rootdir: Optional[str] = "/"):

        self.subject = subject
        self.experiment = experiment
        self.session = session
        self.localization = localization
        self.montage = montage
        self.rootdir = rootdir
        self.readers = {k: getattr(readers, v) for k, v in self.reader_names.items()}

    def get_reader(self, data_type, file_path=None):
        """ Return an instance of the reader class for the given data type """

        return self.readers[data_type](data_type,
                                       subject=self.subject,
                                       experiment=self.experiment,
                                       session=self.session,
                                       localization=self.localization,
                                       montage=self.montage,
                                       file_path=file_path,
                                       rootdir=self.rootdir)

    def load(self, data_type, file_path=None):
        """ Load requested data into memory """
        if data_type not in self.readers:
            raise NotImplementedError("There is no reader to support the "
                                      "requested file type")

        # By default we want task + math events when requesting events
        if data_type == 'events':
            data_type = 'all_events'

        return self.readers[data_type](data_type,
                                       subject=self.subject,
                                       experiment=self.experiment,
                                       session=self.session,
                                       localization=self.localization,
                                       montage=self.montage,
                                       file_path=file_path,
                                       rootdir=self.rootdir).load()
