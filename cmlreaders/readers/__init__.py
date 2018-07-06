# flake8: noqa

# TODO: find a way to automatically import all readers

from .eeg import EEGReader

from .electrodes import (
    ElectrodeCategoriesReader,
    LocalizationReader,
    MontageReader,
)

from .readers import *

from .reports import (
    BaseRAMReportDataReader,
    RAMReportClassifierSummaryReader,
    RAMReportSummaryDataReader,
)
