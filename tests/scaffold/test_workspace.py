import tomllib
from pathlib import Path

import pytest

from robottraderslab.scaffold.workspace import lay_out_workspace, main

_EXAMPLES = ("secrets.example.toml", "registry.example.toml")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path / "workspace"


def test_writes_both_examples_into_a_workspace_it_creates(workspace):
    written = lay_out_workspace(workspace)

    assert written == [workspace / name for name in _EXAMPLES]


def test_marks_the_workspace_as_the_root(workspace):
    lay_out_workspace(workspace)

    assert (workspace / ".rtlab").is_dir()


def test_a_second_run_keeps_what_the_root_already_holds(workspace):
    lay_out_workspace(workspace)
    lock = workspace / ".rtlab" / "bot.lock"
    lock.write_text("", encoding="utf-8")

    lay_out_workspace(workspace)

    assert lock.is_file()


def test_each_example_is_toml_declaring_nothing_yet(workspace):
    lay_out_workspace(workspace)

    for name in _EXAMPLES:
        assert tomllib.loads((workspace / name).read_text(encoding="utf-8")) == {}


def test_an_example_already_there_is_written_afresh(workspace):
    workspace.mkdir()
    stale = workspace / "secrets.example.toml"
    stale.write_text("stale", encoding="utf-8")

    lay_out_workspace(workspace)

    assert stale.read_text(encoding="utf-8") != "stale"


def test_a_filled_file_beside_the_examples_is_untouched(workspace):
    workspace.mkdir()
    filled = workspace / "secrets.toml"
    filled.write_text('[[secrets]]\nname = "mine"\n', encoding="utf-8")

    lay_out_workspace(workspace)

    assert filled.read_text(encoding="utf-8") == '[[secrets]]\nname = "mine"\n'


def test_main_names_each_example_it_wrote(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    main()

    assert capsys.readouterr().out == "".join(
        f"Wrote {Path('workspace') / name}\n" for name in _EXAMPLES
    )
