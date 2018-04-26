# Computational Memory Lab Data Guide
Data collected and processed by the Computational Memory lab is stored in a flat file system attached to the RHINO computing cluster. Although the data is largely consistent as far as locations and naming, there are enough special cases that it can be a hassle for analysts to combine all the information needed for their analyses. The goal of this guide is to inform readers of what data is available and how to read this data using existing tools developed internally by different members of the lab.

## RAM
The DARPA-funded Restoring Active Memory (RAM) project is the primary culprit when it comes to the proliferation of various data files. This following sections enumerate the different data types that are stored and is divided by the level of aggregations, at which, the data is stored.

### Subject
All subject-level data is consistent across all experiments and sessions that a participant completed as well as any re-implants (localization changes) or montage changes that a subject may have experienced.

| Data Type | Name | Description | Format | Associated Repositories | Recommended Reader | Tutorials/Examples |
|:---------:|:----:|:----------- |:------:|:-----------------------:|:------------------:|:------------------:|
|Target Selection Table| target_selection_file | Generated as part of producing reports for record-only sessions. Contains information about subsequent memory effects and other metadata by electrode (location, type, label, etc.) | csv | [ramutils](https://github.com/pennmem/ram_utils) | [`pandas.read_csv()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html)
|Trained Classifier| trained_classifier | A serialized version of a classifier that has been trained using record-only session data. When loaded, the trained model can be used to make out of sample predictions given new powers. | zip | [classiflib](https://github.com/pennmem/classiflib) | [`classiflib.ClassifierContainer.load()`](https://pennmem.github.io/classiflib/html/index.html#loading-a-classifier) | N/A |


### Localization
If a subject has had electrodes implanted/removed after the initial surgery, these files will change since they are localization specific. For the sake of analysis, a subject that is re-localized is often treated as a new subject because classifier features are localization-dependent and are not easily combined when the number/type/location of electrodes changes.

| Data Type | Name | Description | Format | Associated Repositories | Recommended Reader | Tutorials/Examples |
|:---------:|:----:|:----------- |:------:|:-----------------------:|:------------------:|:------------------:|
|Localization|localization|Contains known information about each contact in an subject's montage. This includes coordinates in various spaces, atlas locations, type of contact, and more.| json | [neurorad](https://github.com/pennmem/neurorad_pipeline) | [`ptsa.data.readers.LocReader`](https://github.com/pennmem/ptsa_new/blob/master/ptsa/data/readers/localization.py) | N/A|
|Voxel Coordinates| voxel_coordinates| Contains voxel coordinates, contact labels, and types. This information is contained with the localization file. | txt | N/A| [`pandas.read_csv()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html)  | N/A|
| Prior Stim Results | prior_stim_results | List-level normalized delta recall for all stimulation sessions prior to when localization was completed for the current subject. Used by the 3D brain visualization application to plot prior stim results | csv | [brain_viz](https://github.com/pennmem/brain_viz) | [`pandas.read_csv()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) | N/A |
| Electrode Coordinates | electrode_coordinates | Contains freesurfer average coordinates for each contact; both monopolar and bipolar. Used by 3D brain visualization application to plot contact locations | csv | [brain_viz](https://github.com/pennmem/brain_viz) | [`pandas.read_csv()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) | N/A |
| Jacksheet | jacksheet | Contains all monopolar contact labels | txt | [bptools](https://github.com/pennmem/bptools) | [`pandas.read_csv()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) | N/A |
| Area | area | Surface area of contacts by electrode | txt |[bptools](https://github.com/pennmem/bptools) |[`pandas.read_csv()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html)  | N/A
| Leads | leads | Indices of all contacts | txt |[bptools](https://github.com/pennmem/bptools)  | N/A | N/A |
| Good Leads | good_leads | Indices of all contacts not flagged by clinicians as bad leads | txt | [bptools](https://github.com/pennmem/bptools) | N/A | N/A |
|Classifier Excluded Leads | classifier_excluded_leads | List of contact labels that should be excluded from the classifier for the purposes of training. | txt | [ramutils](https://github.com/pennmem/ram_utils) | N/A | N/A |
| Electrode Categories | electrode_categories | Lists of contacts in that are either in a seizure onset zone or exhibit interical spikes | txt | N/A | N/A | N/A |


### Montage
Montage-related files change whenever a subject's montage changes. This can happen when ...

| Data Type | Name | Description | Format | Associated Repositories | Recommended Reader | Tutorials/Examples |
|:---------:|:----:|:----------- |:------:|:-----------------------:|:------------------:|:------------------:|
| Pairs | pairs | Metadata for each bipolar contact in a subject's montage. Contains atlas locations, coordinates, and type information. | json | [neurorad](https://github.com/pennmem/neurorad_pipeline), [ramutils](https://github.com/pennmem/ram_utils) | [`ptsa.data.readers.TalReader`](https://github.com/pennmem/ptsa_new/blob/master/ptsa/data/readers/tal.py) | [PTSA readers](https://pennmem.github.io/ptsa_new/html/api/data/readers.html) |
| Contacts | contacts | Metadata for each monopolar contact in a subject's montage. Contains atlas locations, coordinates, and type information | json |[neurorad](https://github.com/pennmem/neurorad_pipeline) | [`ptsa.data.readers.TalReader`](https://github.com/pennmem/ptsa_new/blob/master/ptsa/data/readers/tal.py) | [PTSA readers](https://pennmem.github.io/ptsa_new/html/api/data/readers.html) |


### Session

| Data Type | Name | Description | Format | Associated Repositories | Recommended Reader | Tutorials/Examples |
|:---------:|:----:|:----------- |:------:|:-----------------------:|:------------------:|:------------------:|
| Session Summary | session_summary | File containing metadata related to a single session from a RAM experiment. Generated as part of the post-hoc reporting pipeline. Contains all information to regenerate a report for that session | h5 | [ramutils](https://github.com/pennmem/ram_utils) | [`ramutils.reports.summary.SessionSummary`](https://pennmem.github.io/ram_utils/html/data.html#underlying-data) | N/A |
| Classifier Summary | classifier_summary | File containing metadata related to classifier performance. For stim sessions, there is one classifier summary per session. For record-only sessions, classifier summaries are combined by classifier type (encoding vs joint encoding and retrieval) | h5 | [ramutils](https://github.com/pennmem/ram_utils) | [`ramutils.reports.summary.ClassifierSummary`](https://pennmem.github.io/ram_utils/html/data.html#underlying-data) | N/A | 
| Math Summary | math_summary | File containing metadata related to the distractor period from a RAM experiment session. | h5 | [ramutils](https://github.com/pennmem/ram_utils) | [`ramutils.reports.summary.MathSummary`](https://pennmem.github.io/ram_utils/html/data.html#underlying-data) | N/A |
| Used Clasifier | used_classifier | Serialized classifier that was active during the session. | zip | [classiflib](https://github.com/pennmem/classiflib) | [`classiflib.ClassifierContainer.load()`](https://pennmem.github.io/classiflib/html/index.html#loading-a-classifier) | N/A |
| Excluded Pairs | excluded_pairs | List of bipolar pairs that were excluded from classifier training | json | [bptools]() | N/A | N/A |
| All Pairs | all_pairs | The full set of bipolar pairs including pairs that were not actually used for classification | json | [bptools]() | N/A | N/A |
| Experiment Log | experiment_log | ? | ? | ? | ? | ? |
| Session Log | session_log | ? | ? | ? | ? | ? | ? |
| Event Log | event_log | ? | ? | ? | ? | ? | ? |
| Experiment Config | experiment_config | ? | ? | ? | ? | ? | ? | ? |
| EEG | raw_eeg | ? | ? | ? | ? | ? | ? | ? |
| Odin Config | odin_config | ? | ? | ? | ? | ? | ? | ? |
