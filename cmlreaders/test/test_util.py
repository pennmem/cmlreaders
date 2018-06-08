from contextlib import contextmanager
import os
import pytest

from cmlreaders.util import get_root_dir, is_rerefable


@contextmanager
def set_cml_root(path):
    orig = os.environ.copy()

    if path is not None:
        os.environ["CML_ROOT"] = path
    yield
    os.environ = orig


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
