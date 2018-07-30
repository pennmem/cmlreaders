from contextlib import contextmanager
import os
from random import shuffle

import pytest

from cmlreaders import exc
from cmlreaders.util import DefaultTuple, get_protocol, get_root_dir, is_rerefable


@contextmanager
def set_cml_root(path):
    orig = os.environ.copy()

    if path is not None:
        os.environ["CML_ROOT"] = path
    else:
        # We need to remove an existing environment variable if present
        os.environ.pop("CML_ROOT", None)
    yield
    os.environ = orig


@pytest.mark.parametrize("subject", [
    "R1111M", "LTP001", "TJ001", "FR001", "CH001", "DNE123", None
])
def test_get_protocol(subject):
    if subject is None:
        with pytest.raises(ValueError):
            get_protocol(subject)
        return

    if subject.startswith("DNE"):
        with pytest.raises(exc.UnknownProtocolError):
            get_protocol(subject)
    else:
        protocol = get_protocol(subject)

        if subject.startswith("R1"):
            assert protocol == "r1"
        elif subject.startswith("LTP"):
            assert protocol == "ltp"
        else:
            assert protocol == "pyfr"


@pytest.mark.parametrize("path", [None, "/some/path"])
@pytest.mark.parametrize("with_env_var", [True, False])
def test_get_root_dir(path, with_env_var):
    with set_cml_root("/override" if with_env_var else None):

        root = get_root_dir(path)

        if with_env_var:
            if path is None:
                assert root == os.environ["CML_ROOT"]
            else:
                assert root == path
        else:
            if path is None:
                assert root == "/"
            else:
                assert root == path


@pytest.mark.rhino
@pytest.mark.parametrize("subject,experiment,session,result", [
    ("R1361C", "FR1", 0, False),
    ("R1353N", "PAL1", 0, True),
    ("R1111M", "catFR1", 0, True),
    ("R1409D", "FR6", 0, False),
])
def test_isrerefable(subject, experiment, session, result, rhino_root):
    assert is_rerefable(subject, experiment, session, rootdir=rhino_root) == result


class TestDefaultTuple:
    def test_getitem(self):
        invals = list(range(4)) + [None]
        shuffle(invals)
        index = invals.index(None)

        # use default... default
        dtuple = DefaultTuple(invals)
        assert dtuple[index] == 0

        # use a custom default
        dtuple = DefaultTuple(invals, default="hi")
        assert dtuple[index] == "hi"
