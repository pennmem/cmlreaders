RAM Public Data Description
===========================

.. note:: This document was converted from the Word document that accompanies
          the RAM public data releases.

This document contains a listing and description of the various data
included in this public data archive, with requests made via the URL:
http://memory.psych.upenn.edu/RAM_Public_Data

Instructions to download and extract data:

-  Replace <date> with the data of the specific release you are
   downloading/extracting

-  Download all the files named “Release\_<date>.tar.gz.\_\_” into a
   common directory.

-  Combine and untar all the split files containing event, eeg, and
   localization data:

   -  Max/Linux: Using the terminal, run “cat Release\_<date>.tar.gz.\*
      \| tar xzvf –“ in the directory with the downloaded files.

   -  Windows: Use `Winrar <http://www.rarlab.com/>`__ or a similar
      application.

-  Download “Release\_Metadata\_<date>.tar.gz” and untar:

   -  Max/Linux: Using the terminal, run command “tar -xzvf
      Release\_Metadata\_<date>.tar.gz“.

   -  Windows: Use `Winrar <http://www.rarlab.com/>`__ or a similar
      application.

Description of data contents follow:

MetaData
--------

**design\_documents** word documents containing detailed descriptions of
the implementations of behavioral tasks FR1/2/3, CatFR1/2/3, PAL1/2/3, YC1/2,
TH1/TH3

**session\_notes** for each subject, contains tester notes and comments
for each session

**electrode\_categories** for each subject, lists electrode contacts in
the seizure onset zone, exhibiting frequent interictal activity, residing in a
brain lesion, or labelled as bad/broken by the clinical staff

**RAM\_subject\_demographics.csv** for each subject, the age (at testing),
gender, race, ethnicity, and handedness

Directories
-----------

.. csv-table::
    :delim: ;
    :header: Directory,Description
    :widths: auto

    ``/protocols/<protocol>``;``<protocol>`` is the code for the protocol under which an experiment took place. In this archive, the protocol is ‘r1’
    ``subjects/<subject>``;``<subject>`` is the 6-character code representing a participant in an experiment. All data related to the subject is placed under this directory
    ``experiments/<experiment>``;``<experiment>`` is the name of the task in which the subject participated. One of: FR1, FR2, FR3, catFR1, catFR2, catFR3, PAL1, PAL2, PAL3, YC1, YC2, TH1, TH3
    ``sessions/<session>``;``<session>`` is a zero-indexed identifier for a session of an experiment
    ``localizations/<localization>``;``<localization>`` is a zero-indexed number that identifies a reimplant. Data concerning localization of electrodes for this implant is contained in this directory.
    ``montages/<montage>``;``<montage>`` is a zero-indexed number that identifies a montage change. Note that montage numbers do not reset in the case of a new localization.

Files
-----

.. csv-table::
    :header: File,Description
    :delim: ;
    :widths: auto

    ``protocols/r1.json``;A complete index of all information stored within the archive. JSON objects for each subject-session contain paths to the relevant files.
    ``sessions/#/behavioral/current_processed/events.json``;Lists of JSON objects containing information about the behavioral events and stimulus presentations that occurred during an experiment
    ``sessions/#/ephys/current_processed/sources.json``;List of JSON objects containing information about the eeg files contained in the noreref directory.
    ``sessions/#/ephys/current_processed/noreref``;Each file contains raw binary, EEG data split by channel (denoted by suffix, i.e. .001 = Channel 1, etc.).
    ``montages/#/neuroradiology/current_processed/pairs.json``;JSON objects containing localization information for neighboring pairs of implanted electrodes
    ``montages/#/neuroradiology/current_processed/contacts.json``;JSON objects containing localization information for each individual implanted electrode
    ``#/surf/lh.pial, #/surf/rh.pial``;Freesurfer files for each hemisphere containing the reconstructed surface of each subject’s brain

Events
------

JSON events structures are present for each session. Each events
structure consists of an array of objects, in which the object consists
of fields that specify information about the event in question. The
following fields exist across every experiment:

.. csv-table::
    :header: Term,Description
    :widths: auto

    Protocol,"The protocol number under which the subject was run. For the subjects contained in this archive, the protocol will always be ‘r1’"
    Subject,The code of the subject that participated in this session
    Montage,"A decimal number representing the localization and montage numbers of the subject during this experiment. Localization 1, montage 2 is represented as 1.2"
    Experiment,Experiment name
    Session,Session number
    Type,A label for the event. Event types differ across experiments.
    mstime,Epoch time in ms at which the event took place
    msoffset,The amount of time in ms that it took for the call in question to return. Represents an uncertainty in the timing of the event
    eegoffset,The number of samples into the eeg file at which the event took place
    eegfile,The basename of the eeg file that contains the recordings for this session. Eeg files can be located in the corresponding ephys folder for the session
    exp_version,Experiment software version number
    stim_params,"A list of the stimulation parameters that were applied during or around that event. If no stimulation was applied, this value is an empty list"

The ``stim_params`` list contains the following fields:

.. csv-table::
    :header: Field,Description
    :widths: auto

    ``anode_number``,Channel number of the stimulated anode. This number corresponds to a localization made in pairs.json and contacts.json for this montage.
    ``cathode_number``,Channel number of the stimulated cathode
    ``anode_label``,Channel label of the stimulated anode
    ``cathode_label``,Channel label of the stimulated cathode
    ``amplitude``,Amplitude of stimulation in microamps
    ``pulse_freq``,The frequency of the train of pulses used for stimulation
    ``n_pulses``,The number of stimulation pulses delivered in the stimulation train
    ``burst_freq``,Not used in this data set
    ``n_bursts``,Not used in this data set
    ``pulse_width``,Width of an individual pulse in microseconds
    ``stim_on``,Whether stimulation was being applied during this specific event
    ``stim_duration``,Approximate duration of the stimulation (in milliseconds)

The remaining portion of this document describes the specific fields that
pertain to each experiment.

FR and catFR
~~~~~~~~~~~~

.. csv-table::
    :header: Event type,Description
    :widths: auto

    ``SESS_START``/``SESS_END``,The start and end of a session
    ``COUNTDOWN_START``/``COUNTDOWN_END``,The start and end of the countdown period that occurs prior to each list
    ``DISTRACT_START``/``DISTRACT_END``,The start and ends of the math distractor period that occurs between encoding and retrieval. “PRACTICE” indicates that the distractor period occurred on the initial practice list.
    ``REC_START``/``REC_END``,The start and end of the recall period.
    ``REC_WORD``,The recall of a word
    ``REC_WORD_VV``,Production of a non-word vocalization
    ``STIM_ON``,Indicates the onset of stimulation
    ``TRIAL``,The start of a trial
    ``WORD``,The presentation of a word

.. csv-table::
    :header: Field,Description
    :widths: auto

    ``list``,The number of the current list. -1 indicates a practice list
    ``serialpos``,The serial position (at encoding) of the currently presented or recalled item
    ``word``,The currently presented or recalled word
    ``wordno``,The number in the wordpool of the currently presented or recalled item
    ``recalled``,"A boolean flag to indicate during WORD events whether the currently presented word was subsequently recalled, and during REC_WORD events whether the current presented word was a successful recall"
    ``rectime``,"During REC_WORD events, the amount of time elapsed (in ms) since the beginning of the recall period"
    ``intrusion``,"During REC_WORD events, -1 indicates an extra-list intrusion and a positive number N indicates that the word presentation occurred N lists back"
    ``stim_list``,"During FR2/3, indicates that stimulation occurred on a given list"
    ``is_stim``,"During FR2/3, Indicates that stimulation occurred during an item’s presentation"
    ``category``,(catFR only) the category that the currently presented word belongs to
    ``category_num``,(catFR only) a numerical identifier for the category of the current word

PAL
~~~

.. csv-table::
    :header: Event type,Description
    :widths: auto

    ``SESS_START/SESS_END``,The start and end of a session
    ``ENCODING_START``,The start of the encoding period
    ``MATH_START``/``MATH_END``,The start and end of the math distractor period
    ``REC_START``/``REC_END``,The start and end of the recall period for individual items during retrieval
    ``REC_EVENT``,A recall or vocalization
    ``STUDY_PAIR``,The presentation of a pair of items during encoding
    ``STUDY_ORIENT``,The appearance of the orient cue before words during encoding
    ``TEST_PROBE``,The presentation of a word during retrieval
    ``TEST_ORIENT``,The appearance of the orient cue before words during retrieval
    ``TEST_START``,The start of the retrieval period

.. csv-table::
    :header: Field,Description
    :widths: auto

    ``resp_word``,The word that was recalled for the current pair
    ``probe_word``,The word that was shown as the probe for the current pair
    ``probepos``,The position in which the probe from the current pair was presented at  retrieval
    ``cue_direction``,Whether the top (1) or bottom (0) item was presented as the probe
    ``is_stim``,Whether stimulation occurred during the given event
    ``resp_pass``,Whether the current item was responded to with PASS as the only recall
    ``RT``,The amount of time (in ms) that elapsed between the presentation of the probe and the recall
    ``serialpos``,The position in which the current pair was presented during encoding
    ``stim_list``,Whether stimulation was applied during the current list
    ``correct``,Whether a correct recall was made during the retrieval period
    ``study_1``,The word that was presented at the top during encoding
    ``study_2``,The word that was presented at the bottom during encoding
    ``vocalization``,Whether the current recall was a non-word vocalization
    ``stim_type``,Whether stimulation on this list occurred at encoding or retrieval
    ``intrusion``,"On incorrect recalls, -1 if the word was an extra-list intrusion, 0 if the word came from the current list, and N (N>1) if the word was presented N lists back"
    ``list``,The number of the current list (-1 for practice)
    ``expecting_word``,The word that was intended to be recalled during retrieval

YC
~~~

.. csv-table::
    :header: Event type,Description
    :widths: auto

    ``NAV_LEARN``,A trial in which the subject is driven automatically to the target object with the object visible
    ``NAV_PRACTICE_LEARN``,"Same as NAV_LEARN, but considered practice"
    ``NAV_PRACTICE_TEST``,"Same as NAV_TEST, but considered practice"
    ``NAV_TEST``,A trial in which the subject drives to where they believe the invisible target object is located

.. csv-table::
    :header: Field,Description
    :widths: auto

    obj_locs,XY coordinate of the target object
    stimulus_num,The current trial count within a session
    resp_reaction_time,Length of time (seconds) to initiate movement
    start_locs,XY coordinate of the starting location
    env_size,"Array representing the bounds of the environent [minimum x, maximum x, minimum y, maximum y]"
    resp_path_length,Number of units traversed on the path taken between start_locs and resp_locs
    resp_dist_err,Euclidean distance between the target location (obj_locs) and the response location (resp_locs)
    is_stim,"Indicates whether the current NAV_LEARN, NAV_LEARN, NAV_TEST set was stimulated or unstimulated"
    resp_performance_factor,"Normalized distance between the target location (obj_locs) and the response location (resp_locs).  0 is a perfect response, 1 is the worst possible response."
    recalled,"Indicates whether the euclidean distance error for the current NAV_LEARN, NAV_LEARN, NAV_TEST set was below the median of the subject's distance errors"
    resp_locs,XY coordinate of the response location
    path,"Contains the subarrays 'x', 'y', 'direction', and 'time' detailing the path taken between start_locs and resp_locs"
    resp_travel_time,"Once movement is initiated, the length of time (seconds) spent navigating"
    block_num,"The number of the current NAV_LEARN, NAV_LEARN, NAV_TEST set"
    stimulus,The identity of the current target object
    paired_block,Indicates the block used to counterbalance the current block
    block,"The number of the current block. One block is composed of one pair of NAV_LEARN, NAV_LEARN, NAV_TEST sets"

TH and THR
~~~~~~~~~~

.. csv-table::
    :header: Event type,Description
    :widths: auto

    ``CHEST``,The opening of a treasure chest. The chest can either be filled with a study object or empty
    ``REC``,The moment a response position is chosen

.. csv-table::
    :header: Field,Description
    :widths: auto

    trial,"trial number, zero indexed. Values will range from 0 – 39 in a full session"
    chestNum,"chest number within a given trial (1 – 3 or 4, depending on listLength. One indexed)."
    block,"block number, zero indexed. Values will range from 0 – 4 in a full session."
    listLength,(2 or 3) – Indicates how many filled chests were present for the current trial.
    radius_size,Float (constant across all events) indicating the size (in VR units) of the selection circle radius.
    is_stim,"(0 or 1) – Indicates whether electrical stimulation was received during this event. For the TH1 task, this is always 0."
    stim_list,"(0 or 1) – Indicates whether electrical stimulation was received during this trial. For the TH1 task, this is always 0."
    locationX,X-coordinate of current chest position.
    locationY,Y-coordinate of current chest position.
    item_name,"String identifying the current item. If empty, the chest contained no item."
    navStartLocationX,X-coordinate of starting position for the current trial.
    navStartLocationY,Y-coordinate of starting position for the current trial.
    isRecFromNearSide,"(0 or 1) - Indicates whether the correct item location is in the near half or far half of the field, relative to the retrieval viewpoint location."
    isRecFromStartSide,(0 or 1) - Indicates whether the retrieval viewpoint location is the same side of the field as where the trial started.
    reactionTime,Float indicating the amount of time (in ms) between when the item probe was given and when the response location was selected.
    confidence,"(0, 1, 2) – Indicates whether the subject selected the low, medium, or high confidence response for the item."
    recStartLocationX,X-coordinate of the retrieval period viewpoint for the current trial.
    recStartLocationY,Y-coordinate of the retrieval period viewpoint for the current trial.
    distErr,Float indicating the Euclidean distance between the true chest location for this item and the response location.
    recalled,(0 or 1) – Indicates whether the chosen response location fell within radius_size of the correct location.
    normErr,"Float (between 0 and 1) indicating normalized distance error, where 0 is a perfect response and 1 is the worst possible response, given the object’s location."
    chosenLocationX,X-coordinate of current response position.
    chosenLocationY,Y-coordinate of current response position.

Montage information (pairs and contacts)
----------------------------------------

The ``pairs.json`` and ``contacts.json`` files contains a “contacts” object,
which in turn contains an object for each contact, with the contact
label as the key. The fields within the contact are:

.. csv-table::
    :header: Field,Description
    :widths: auto

    ``atlases.avg``,Registered to an internally made average brain
    ``atlases.avg.dural``,"Registered to the average brain, snapped to the dural surface"
    ``atlases.ind``,Registered to the subjects individual brain
    ``atlases.ind.dural``,"Registered to the subjects individual brain, snapped to the dural surface"
    ``atlases.mni``,Coordinates in MNI space
    ``atlases.tal``,Coordinates in Talairach space
    ``channel``,Channel number of the contact
    ``code``,Label for the contact
    ``type``,"One of “s”, “g”, or “d”, representing strips, grids, or depths respectively"
