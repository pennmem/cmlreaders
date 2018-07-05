from pathlib import Path
import pytest

from cmlreaders.data_index import get_data_index, read_index


@pytest.mark.rhino
@pytest.mark.parametrize("kind", ["r1", "ltp"])
def test_read_index(kind, rhino_root):
    index = Path(rhino_root).joinpath("protocols", "{}.json".format(kind))
    data = read_index(index)
    if kind == "r1":
        assert "R1111M" in data
    else:
        assert "LTP093" in data


@pytest.mark.rhino
@pytest.mark.parametrize("kind", ["r1", "ltp", "all"])
def test_get_data_index(kind, rhino_root):
    df = get_data_index(kind, rootdir=rhino_root)

    if kind in ["r1", "all"]:
        assert any(df.subject == "R1111M")
        assert df[df.subject == "R1111M"].experiment.count() == 22
    if kind in ["ltp", "all"]:
        assert any(df.subject == "LTP093")
        assert df[df.subject == 'LTP093'].experiment.count() == 24
    if kind != "ltp":
        assert df["localization"].dtype == int
        assert df["montage"].dtype == int
