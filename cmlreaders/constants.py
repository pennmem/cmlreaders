
rhino_paths = {
    'protocols_database': ['protocols/r1.json'],
    # Localization-level (subject + localization)
    'voxel_coordinates': [
        "data10/RAM/subjects/{subject_localization}/tal/VOX_coords_mother.txt",
    ],
    'prior_stim_results': [
        'data10/eeg/freesurfer/subjects/{subject_localization}/prior_stim/{subject}_allcords.csv',
    ],
    'electrode_coordinates': [
        'data10/RAM/subjects/{subject_localization}/tal/coords/electrode_coordinates.csv',
        'data10/RAM/subjects/{subject_localization}/tal/electrode_coordinates.csv',
    ],
    'jacksheet': [
        'data10/RAM/subjects/{subject_localization}/docs/jacksheet.txt',
    ],
    'area': [
        'data10/RAM/subjects/{subject_localization}/docs/area.txt',
    ],
    'electrode_categories': [
        'data10/RAM/subjects/{subject_localization}/docs/electrode_categories.txt',
    ],
    'good_leads': [
        'data10/RAM/subjects/{subject_localization}/tal/good_leads.txt',
    ],
    'leads': [
        'data10/RAM/subjects/{subject_localization}/tal/leads.txt',
    ],
    'classifier_excluded_leads': [
        'data10/RAM/subjects/{subject_localization}/tal/classifier_excluded_leads.txt',
    ],
    'localization': [
        'protocols/r1/subjects/{subject}/localizations/{localization}/neuroradiology/current_processed/localization.json',
    ],

    # Montage level (subject + localization + montage)
    'pairs': [
        'protocols/r1/subjects/{subject}/localizations/{localization}/montages/{montage}/neuroradiology/current_processed/pairs.json',
    ],
    'contacts': [
        'protocols/r1/subjects/{subject}/localizations/{localization}/montages/{montage}/neuroradiology/current_processed/contacts.json',
    ],

    # Report Data
    'session_summary': [
        'scratch/report_database/{subject}_{experiment}_{session}_session_summary.h5',
    ],
    'classifier_summary': [
        'scratch/report_database/{subject}_{experiment}_{session}_classifier_session_{session}.h5',
    ],
    'math_summary': [
        'scratch/report_database/{subject}_{experiment}_{session}_math_summary.h5',
    ],
    'target_selection_table': [
        'scratch/report_database/{subject}_{experiment}_*_target_selection_table.csv',
    ],
    'trained_classifier': [
        'scratch/report_database/{subject}_retrained_classifier.zip',
        'scratch/report_database/{subject}_{experiment}_all_retrained_classifier.zip',
    ],

    # Session Data
    'all_events': ['protocols/r1/subjects/{subject}/experiments/{experiment}/sessions/{session}/behavioral/current_processed/all_events.json'],
    'task_events': ['protocols/r1/subjects/{subject}/experiments/{experiment}/sessions/{session}/behavioral/current_processed/task_events.json'],
    'math_events': ['protocols/r1/subjects/{subject}/experiments/{experiment}/sessions/{session}/behavioral/current_processed/math_events.json'],
    'ps4_events': ['protocols/r1/subjects/{subject}/experiments/{experiment}/sessions/{session}/behavioral/current_processed/ps4_events.json'],
    'sources': ['protocols/r1/subjects/{subject}/experiments/{experiment}/sessions/{session}/ephys/current_processed/sources.json'],

    # Ramulator-related information
    'experiment_log': [
        'data10/RAM/subjects/{subject}/behavioral/{experiment}/experiment.log',
    ],
    'session_log': [
        'data10/RAM/subjects/{subject}/behavioral/{experiment}/session_{session}/session.log',
    ],
    'ramulator_session_folder': [
        'data10/RAM/subjects/{subject}/behavioral/{experiment}/session_{session}/host_pc/*',
    ],

    # There can be multiple timestamped folders for the host pc files for when
    # a session is restarted
    'event_log': [
        'data10/RAM/subjects/{subject}/behavioral/{experiment}/session_{session}/host_pc/{timestamped_dir}/event_log.json',
    ],
    'experiment_config': [
        'data10/RAM/subjects/{subject}/behavioral/{experiment}/session_{session}/host_pc/{timestamped_dir}/experiment_config.json',
    ],
    'raw_eeg': [
        'data10/RAM/subjects/{subject}/behavioral/{experiment}/session_{session}/host_pc/{timestamped_dir}/eeg_timeseries.h5',
    ],
    'odin_config': [
        'data10/RAM/subjects/{subject}/behavioral/{experiment}/session_{session}/host_pc/{timestamped_dir}/config_files/{subject}_*.csv',
    ],

    # There can be multiple classifiers if artifcat detection was enabled and
    # the classifier needed to be retrained. The order is important here In
    # general, the files should be listed in order of preference so that the
    # first file found is returned
    'used_classifier': [
        'data10/RAM/subjects/{subject}/behavioral/{experiment}/session_{session}/host_pc/{timestamped_dir}/config_files/retrained_classifier/{subject}-classifier.zip',
        'data10/RAM/subjects/{subject}/behavioral/{experiment}/session_{session}/host_pc/{timestamped_dir}/config_files/{subject}-classifier.zip',
    ],
    'excluded_pairs': [
        'data10/RAM/subjects/{subject}/behavioral/{experiment}/session_{session}/host_pc/{timestamped_dir}/config_files/retrained_classifier/excluded_pairs.json',
        'data10/RAM/subjects/{subject}/behavioral/{experiment}/session_{session}/host_pc/{timestamped_dir}/config_files/excluded_pairs.json'
    ],
    'all_pairs': [
        'data10/RAM/subjects/{subject}/behavioral/{experiment}/session_{session}/host_pc/{timestamped_dir}/config_files/pairs.json',
        'data10/RAM/subjects/{subject}/behavioral/{experiment}/session_{session}/host_pc/{timestamped_dir}/config_files/retrained_classifier/pairs.json'
    ],
}

# Maintain separate lists of the file types depending on what information is
# required to be able to find them
localization_files = [
    'voxel_coordinates',
    'prior_stim_results',
    'electrode_coordinates',
    'jacksheet',
    'good_leads',
    'leads',
    'area',
    'classifier_excluded_leads',
    'localization',
    'electrode_categories',
]

# All files that change when a montage changes
montage_files = ['pairs', 'contacts']

# All files that are constant by subject
subject_files = ['target_selection_file', 'trained_classifier']

# All files that vary at the session level
session_files = [
    'session_summary',
    'classifier_summary',
    'math_summary',
    'used_classifier',
    'excluded_pairs',
    'all_pairs',
    'experiment_log',
    'session_log',
    'event_log',
    'experiment_config',
    'raw_eeg',
    'odin_config',
    'all_events',
    'task_events',
    'math_events',
    'ps4_events'
]

# All files that require some extra work to identify
host_pc_files = [
    'event_log',
    'experiment_config',
    'raw_eeg',
    'odin_config',
    'used_classifier',
    'excluded_pairs',
    'all_pairs',
]

# Files related to in-session classifier retraining
used_classifier_files = ['used_classifier', 'excluded_pairs', 'all_pairs']
