from pathlib import Path

import pytest

from robottraderslab.scaffold import ExampleError
from robottraderslab.scaffold.command import main


def test_lists_installed_examples_when_no_name_given(monkeypatch, capsys):
    monkeypatch.setattr(
        "robottraderslab.scaffold.command.list_examples",
        lambda: ["breakout_demo", "momentum_demo"],
    )

    main(None)

    assert capsys.readouterr().out == "breakout_demo\nmomentum_demo\n"


def test_reports_when_no_examples_installed(monkeypatch, capsys):
    monkeypatch.setattr("robottraderslab.scaffold.command.list_examples", lambda: [])

    main(None)

    assert capsys.readouterr().out == "No examples installed.\n"


def test_copies_the_named_example_into_the_workspace(monkeypatch, capsys):
    requested: list[tuple[str, Path]] = []

    def _copy(name: str, destination: Path) -> Path:
        requested.append((name, destination))
        return destination / name

    monkeypatch.setattr("robottraderslab.scaffold.command.copy_example", _copy)

    main("momentum_demo")

    assert requested == [("momentum_demo", Path("workspace"))]
    expected = Path("workspace") / "momentum_demo"
    assert capsys.readouterr().out == f"Example copied to {expected}\n"


def test_an_unknown_example(monkeypatch):
    monkeypatch.setattr("robottraderslab.scaffold.command.copy_example", _raise_unknown)

    with pytest.raises(ExampleError, match="ghost_demo"):
        main("ghost_demo")


def _raise_unknown(name: str, destination: Path) -> Path:
    raise ExampleError(f"Unknown example: {name}")
