from copy import copy
from pkg_resources import resource_filename
import pytest

from cmlreaders import cache
from cmlreaders.base_reader import BaseCMLReader
from cmlreaders.readers.electrodes import MontageReader


@pytest.fixture(params=[True, False])
def caching_enabled(request):
    orig_state = copy(cache.enabled)
    cache.enabled = request.param
    yield request.param
    cache.enabled = orig_state


@pytest.fixture
def contacts_reader():
    path = resource_filename("cmlreaders.test.data", "R1006P_contacts.json")
    mr = MontageReader("contacts", "R1006P", file_path=path)
    yield mr
    mr.clear_cache()
    BaseCMLReader._instances.clear()


@pytest.fixture
def pairs_reader():
    path = resource_filename("cmlreaders.test.data", "R1006P_pairs.json")
    mr = MontageReader("pairs", "R1006P", file_path=path)
    yield mr
    mr.clear_cache()
    BaseCMLReader._instances.clear()


def test_clear_all(caching_enabled, contacts_reader, pairs_reader):
    contacts_reader.load()
    pairs_reader.load()

    assert len(BaseCMLReader._instances) == 2

    if caching_enabled:
        assert contacts_reader._result is not None
        assert pairs_reader._result is not None

    cache.clear_all()

    assert contacts_reader._result is None
    assert pairs_reader._result is None


def test_enable(caching_enabled):
    cache.enable()
    assert cache.enabled


@pytest.mark.parametrize("clear", [True, False])
def test_disable(caching_enabled, clear):
    cache.disable(clear)
    assert not cache.enabled


def test_in_memory_caching(caching_enabled):
    """Test in-memory caching."""
    path = resource_filename("cmlreaders.test.data", "R1006P_contacts.json")
    mr = MontageReader("contacts", "R1006P", file_path=path)
    df = mr.load()

    if caching_enabled:
        assert mr._result is df
    else:
        assert mr._result is None
