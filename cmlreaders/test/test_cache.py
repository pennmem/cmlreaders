import pytest

from cmlreaders import cache


class TestCache:
    def test_clear_all(self):
        with pytest.raises(NotImplementedError):
            cache.clear_all()
