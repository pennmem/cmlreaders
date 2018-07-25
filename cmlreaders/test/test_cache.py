import pandas as pd
import pytest

from cmlreaders.base_reader import BaseCMLReader
from cmlreaders import cache


class TestCache:
    def setup_method(self):
        cache._cached_readers.clear()

    def make_reader_class(self):
        class DummyReader(BaseCMLReader):
            data_types = ["dummy"]
            default_representation = "dataframe"

            def as_dataframe(self):
                return pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1]})

        return cache.cache(DummyReader)

    def test_caching(self):
        cls = self.make_reader_class()
        instance = cls("dummy")

        assert hasattr(instance, "cached_result")
        assert instance.cached_result is None

        df = instance.load()
        assert all(df == instance.cached_result)
        df2 = instance.load()
        assert all(df == df2)

        instance.clear()
        assert instance.cached_result is None

    def test_clear_all(self):
        obj1 = self.make_reader_class()("dummy")
        obj2 = self.make_reader_class()("dummy")
        assert len(cache._cached_readers) == 2

        obj1.load()
        obj2.load()

        assert obj1.cached_result is not None
        assert obj2.cached_result is not None

        cache.clear_all()

        assert obj1.cached_result is None
        assert obj2.cached_result is None
