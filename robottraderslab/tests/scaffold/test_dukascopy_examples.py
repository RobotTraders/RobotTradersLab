from pathlib import Path

import pytest

from robottraderslab.scaffold import main as new


@pytest.fixture(autouse=True)
def _require_the_dukascopy_plugin() -> None:
    pytest.importorskip("robottraderslab_dukascopy")


@pytest.mark.parametrize(
    ("example", "notebook"),
    [
        ("dukascopy-impulse", "run_impulse_forex.ipynb"),
        ("dukascopy-envelope", "run_envelope_forex.ipynb"),
    ],
)
def test_new_creates_a_bot_folder_with_its_notebook(tmp_path, example, notebook):
    new(example, Path("first-bot"), tmp_path)

    assert {path.name for path in (tmp_path / "first-bot").iterdir()} == {
        "bot.toml",
        notebook,
    }
