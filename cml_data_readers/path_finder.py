""" Module for mapping file types to their locations on RHINO """
import os
import glob
from .constants import *


class InvalidFileTypeRequest(Exception):
    """ Exception for requests that are not supported """


class PathFinder(object):
    """ Class for finding files on RHINO """
    def __init__(self, subject, rootdir='/', experiment=None, session=None, localization=None, montage=None):
        """ Instantiates a PathFind object using the known information 

        Parameters:
        -----------
        subject: str
            Subject ID
        
        Keyword Argments:
        -----------------
        rootdir: str
            Root directory for RHINO
        experiment: str or None
            RAM experiment name. If none, then the data is assumed to be constant across experiments
        session: int or None
            Session number. If none, then the data is assumed to be constant across sessions
        localization: int
            Localization number
        montage: int or None
            Montage number
        """
        self.subject = subject
        self.rootdir = rootdir
        self.experiment = experiment
        self.session = session
        self.localization = localization
        self.montage = montage
        self._paths = rhino_paths

    @property
    def path_info(self):
        return self._paths

    @property
    def requestable_files(self):
        """ All files that can be requested with `PathFinder.find_file` """
        return list(self._paths.keys())
    
    @property
    def localization_files(self):
        """ All localization related files """
        return self.localization_files

    @property
    def montage_files(self):
        """ All files that vary by montage number """
        return self.montage_files
    
    @property
    def subject_files(self):
        """ All files that vary only by subject """
        return self.subject_files


    def find_file(self, file_type):
        """ Given a specific file type, find the corresponding file on RHINO and return the full path 
    
        Parameters:
        -----------
        file_type: The type of file to load. The given name should match one of the
                   keys from rhino_paths
        Returns:
        --------
        path: str
            The path of the file found based on the request

        """
        if file_type not in rhino_paths:
            raise InvalidFileTypeRequest("Unknown file type")

        expected_path = self._lookup_file(file_type)

        return expected_path

    def _lookup_file(self, file_type):
        """ This will handle the special cases before passing the data through 
            to _find_single_path 
        """
        subject_localization = self.subject

        if type(self.localization) == int:
            localization = str(self.localization)

        # Some files/locations append the localization number, so to abstract that away
        # from the user, we handle this internally
        if localization != '0':
            subject_localization = "_".join([self.subject, self.localization])

        paths_to_check = self._paths[file_type]
        timestamped_dir = None

        # Only check the host_pc folder if necessary
        if (file_type in host_pc_files) or (file_type in used_classifier_files):
            ramulator_session_folder = self._paths['ramulator_session_folder'][0].format(
                subject=subject_localization, experiment=self.experiment, session=self.session)

            timestamped_dir = self._get_most_recent_ramulator_folder(ramulator_session_folder,
                                                                     self.subject, 
                                                                     self.experiment, 
                                                                     self.session)
            # The user can also just request the folder
            if file_type == 'ramulator_session_folder':
                return ramulator_session_folder
        
        expected_path = self._find_single_path(paths_to_check,
                                               subject=self.subject,
                                               subject_localization=subject_localization,
                                               timestamped_dir=timestamped_dir,
                                               experiment=self.experiment,
                                               session=self.session,
                                               localization=self.localization,
                                               montage=self.montage)
        return expected_path

    def _get_most_recent_ramulator_folder(self, base_folder_path, subject, experiment, session):
        timestamped_directories = glob.glob(os.path.join(self.rootdir, base_folder_path))
        
        # Sort such that most recently modified appears first
        print(timestamped_directories)
        timestamped_directories = sorted(timestamped_directories, key=os.path.getmtime, reverse=True)

        if len(timestamped_directories) == 0:
            raise RuntimeError("No timestamped folder found in host_pc folder")
        
        if len(timestamped_directories) > 1:
            raise RuntimeWarning("Multiple timestamped directories found. The most recent will be returned")

        # Only return the values from the final "/" to the end
        latest_directory = timestamped_directories[0][timestamped_directories[0].rfind("/") + 1 :]
        
        return latest_directory

    def _find_matching_files(self, wildcard_path):
        return glob.glob(wildcard_path)

    def _find_single_path(self, paths, **kwargs):
        final_paths = []
        for path in paths:
            if "*" in path:
                final_paths.extend(self._find_matching_files(os.path.join(self.rootdir, 
                                                                          path.format(**kwargs))))
            else:
                final_paths.extend([path])
        
        found_files = []
        for path in final_paths:
            expected_path = os.path.join(self.rootdir,
                                         path.format(**kwargs))
            if os.path.exists(expected_path):
                found_files.append(expected_path)

        if len(found_files) == 0:
            raise FileNotFoundError("Unable to find the requested file in any of the expected locations")

        if len(found_files) > 1:
            raise RuntimeWarning('Multiple files found. Returning the first file found')

        return found_files[0]
        
