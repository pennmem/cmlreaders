from copy import copy
from pkg_resources import resource_filename
import pytest

from cmlreaders import cache
from cmlreaders.readers.electrodes import MontageReader


@pytest.fixture(params=[True, False])
def caching_enabled(request):
    orig_state = copy(cache.enabled)
    cache.enabled = request.param
    yield request.param
    cache.enabled = orig_state


def test_clear_all():
    with pytest.raises(NotImplementedError):
        cache.clear_all()


def test_enable(caching_enabled):
    cache.enable()
    assert cache.enabled


@pytest.mark.parametrize("clear", [True, False])
def test_disable(caching_enabled, clear):
    if clear:
        with pytest.raises(NotImplementedError):
            cache.disable(clear)
    else:
        cache.disable(clear)


def test_in_memory_caching(caching_enabled):
    """Test in-memory caching."""
    path = resource_filename("cmlreaders.test.data", "R1006P_contacts.json")
    mr = MontageReader("contacts", "R1006P", file_path=path)
    df = mr.load()

    if caching_enabled:
        assert mr._result is df
    else:
        assert mr._result is None
