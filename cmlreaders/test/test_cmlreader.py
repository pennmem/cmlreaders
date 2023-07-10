from contextlib import ExitStack
import functools
import os
from unittest.mock import patch

import pandas as pd
from pkg_resources import resource_filename
import pytest

from cmlreaders import CMLReader, exc, get_data_index
from cmlreaders.path_finder import PathFinder
from cmlreaders.test.utils import patched_cmlreader

datafile = functools.partial(resource_filename, "cmlreaders.test.data")


class TestCMLReader:
    @pytest.mark.parametrize("protocol", ["all", "r1"])
    def test_get_data_index(self, protocol):
        if protocol == "all":
            path = resource_filename("cmlreaders.test.data", "r1.json")
        else:
            path = resource_filename("cmlreaders.test.data", protocol + ".json")

        with patch.object(PathFinder, "find", return_value=path):
            ix = CMLReader.get_data_index(protocol)
            assert all(ix == get_data_index(protocol))

    @pytest.mark.parametrize(
        "subject,experiment,session,localization,montage",
        [
            ("R1278E", "catFR1", 0, 0, 1),
            ("R1278E", "catFR1", None, 0, 1),
            ("R1278E", "PAL1", None, 2, 2),
            ("R1278E", "PAL3", 2, 2, 2),
            ("R1278E", "TH1", 0, 0, 0),
            ("R1278E", "TH1", None, 0, 0),
            ("LTP093", "ltpFR2", 0, None, None),
        ],
    )
    def test_determine_localization_or_montage(
        self, subject, experiment, session, localization, montage
    ):
        with patched_cmlreader():
            reader = CMLReader(subject=subject, experiment=experiment, session=session)
            assert reader.montage == montage
            assert reader.localization == localization

    @pytest.mark.rhino
    # FIXME add an intrcranial subject to test
    @pytest.mark.parametrize(
        "subject,experiment,session,localization",
        [
            ("LTP093", "ltpFR2", 0, None),
        ],
    )
    @pytest.mark.parametrize(
        "file_type",
        [
            "voxel_coordinates",
            "classifier_excluded_leads",
            "jacksheet",
            "mni_coordinates",
            "good_leads",
            "leads",
            "electrode_coordinates",
            "prior_stim_results",
            "target_selection_table",
            "electrode_categories",
            "classifier_summary",
            "math_summary",
            "session_summary",
            "pairs",
            "contacts",
            "localization",  # 'baseline_classifier', 'used_classifier',
            "events",
            "all_events",
            "task_events",
            "math_events",
        ],
    )
    def test_load_from_rhino(
        self, subject, experiment, session, localization, file_type, rhino_root
    ):
        if subject.startswith("LTP"):
            reader = CMLReader(
                subject=subject,
                experiment=experiment,
                session=session,
                localization=localization,
                rootdir=rhino_root,
            )
            if "ltp" not in reader.reader_protocols[file_type]:
                with pytest.raises(exc.UnsupportedProtocolError):
                    reader.load(file_type)
                return

        if file_type in [
            "electrode_categories",
            "classifier_summary",
            "math_summary",
            "session_summary",
            "baseline_classifier",
        ]:
            subject = "R1111M"
            experiment = "FR2"
            session = 0

        if file_type in ["used_classifier"]:
            subject = "R1409D"
            experiment = "FR6"
            session = 0
            localization = 0

        if subject.startswith("LTP") and file_type in ["contacts", "pairs"]:
            pytest.xfail("unsure if montage data exists for LTP")

        reader = CMLReader(
            subject=subject,
            localization=localization,
            experiment=experiment,
            session=session,
            rootdir=rhino_root,
        )
        reader.load(file_type)

    # FIXME: find a good way to test ramutils-requiring things
    @pytest.mark.parametrize(
        "file_type",
        [
            "voxel_coordinates.txt",
            "classifier_excluded_leads.txt",
            "jacksheet.txt",
            "good_leads.txt",
            "leads.txt",
            "electrode_coordinates.csv",
            "prior_stim_results.csv",
            "target_selection_table.csv",
            # 'classifier_summary.h5',
            # 'math_summary.h5',
            # 'session_summary.h5',
            "pairs.json",
            "contacts.json",
            "localization.json",
            # "baseline_classifier.zip",
            # "used_classifier.zip",
            "all_events.json",
            "math_events.json",
            "task_events.json",
        ],
    )
    def test_load(self, file_type):
        with patched_cmlreader(datafile(file_type)):
            data_type = os.path.splitext(file_type)[0]
            reader = CMLReader(
                subject="R1405E", localization=0, experiment="FR5", session=1
            )
            reader.load(data_type=data_type)

    # FIXME: find a good way to test ramutils-requiring things
    @pytest.mark.parametrize(
        "file_type",
        [
            "voxel_coordinates",
            "classifier_excluded_leads",
            "jacksheet",
            "good_leads",
            "leads",
            "electrode_coordinates",
            "prior_stim_results",
            "target_selection_table",
            # 'classifier_summary', 'math_summary', 'session_summary',
            "pairs",
            "contacts",
            "localization",
            "baseline_classifier",
            "used_classifier",
            "all_events",
            "math_events",
            "task_events",
        ],
    )
    def test_get_reader(self, file_type):
        with patched_cmlreader():
            reader = CMLReader(
                subject="R1405E", localization=0, experiment="FR1", session=0, montage=0
            )
            reader_obj = reader.get_reader(file_type)
            assert isinstance(reader_obj, reader.readers[file_type])

    def test_load_unimplemented(self):
        with patched_cmlreader():
            reader = CMLReader(
                subject="R1405E", localization=0, experiment="FR1", session=0, montage=0
            )
            with pytest.raises(NotImplementedError):
                reader.load("fake_data")

    @pytest.mark.rhino
    @pytest.mark.parametrize(
        "subject,experiment,session",
        [
            ("R1354E", "PS4_FR5", 1),
            ("R1354E", "PS4_FR", 1),
            ("R1111M", "PS2", 0),
            ("R1025P", "PS1", 0),
        ],
    )
    def test_ps_events(self, subject, experiment, session, rhino_root):
        reader = CMLReader(subject, experiment, session, rootdir=rhino_root)
        events = reader.load("events")
        task_events = reader.load("task_events")
        assert all(events == task_events)


class TestLoadMontage:
    @staticmethod
    def assert_categories_correct(
        df: pd.DataFrame, categories: dict, read_categories: bool
    ):
        if read_categories:
            assert "category" in df.columns
            for category, labels in categories.items():
                mask = df["label"].isin(labels)
                assert all([category in entry for entry in df[mask]["category"]])
        else:
            assert "category" not in df.columns

    @pytest.mark.parametrize("kind", ["contacts", "pairs"])
    @pytest.mark.parametrize("read_categories", [True, False])
    def test_read_categories(self, kind, read_categories):
        from cmlreaders.path_finder import PathFinder
        from cmlreaders.readers.electrodes import (
            ElectrodeCategoriesReader,
            MontageReader,
        )

        cpath = datafile("R1111M_electrode_categories.txt")
        categories = ElectrodeCategoriesReader.fromfile(cpath)
        mpath = datafile("R1111M_{}.json".format(kind))

        with ExitStack() as stack:
            stack.enter_context(patched_cmlreader())
            stack.enter_context(patch.object(PathFinder, "find", return_value=""))
            stack.enter_context(
                patch.object(ElectrodeCategoriesReader, "load", return_value=categories)
            )
            stack.enter_context(patch.object(MontageReader, "_file_path", mpath))

            reader = CMLReader("R1111M", "FR1", 0, 0, 0)
            df = reader.load(kind, read_categories=read_categories)

        self.assert_categories_correct(df, categories, read_categories)

    # from PathFinder finding multiple files
    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    @pytest.mark.rhino
    @pytest.mark.parametrize("kind", ["contacts", "pairs"])
    @pytest.mark.parametrize("read_categories", [True, False])
    def test_read_categories_rhino(self, kind, read_categories, rhino_root):
        reader = CMLReader("R1111M", "FR1", 0, 0, 0, rootdir=rhino_root)
        df = reader.load(kind, read_categories=read_categories)

        if read_categories:
            categories = reader.load("electrode_categories")
        else:
            categories = None

        self.assert_categories_correct(df, categories, read_categories)


@pytest.mark.rhino
class TestLoadAggregate:
    @pytest.mark.parametrize(
        "subjects,experiments,unique_sessions",
        [
            (None, None, None),
            ("R1111M", "FR1", 4),
            (["R1111M", "R1260D"], ["FR1"], 5),
            (["R1111M"], None, 22),
            (["R1111M"], ["PS2"], 6),
            (None, ["FR2"], 79),
            ("R1289C", ["TH1"], 3),
        ],
    )
    def test_load_events(self, subjects, experiments, unique_sessions, rhino_root):
        if subjects is experiments is None:
            with pytest.raises(ValueError):
                CMLReader.load_events(subjects, experiments, rootdir=rhino_root)
            return

        events = CMLReader.load_events(subjects, experiments, rootdir=rhino_root)
        size = len(events.groupby(["subject", "experiment", "session"]).size())
        assert size == unique_sessions
