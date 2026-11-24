from pathlib import Path

import pytest

from robottraderslab._internal import (
    mark_root,
    scheduler_file,
    tool_file,
    workspace_root,
)


@pytest.mark.parametrize(
    ("owner", "suffix", "expected"),
    [
        ("workspace/pilot/bot.toml", ".lock", "workspace/pilot/.rtlab/bot.lock"),
        ("registry.toml", ".launches.json", ".rtlab/registry.launches.json"),
    ],
)
def test_a_file_the_engine_keeps_sits_in_the_tool_directory_beside_its_owner(
    tmp_path, owner, suffix, expected
):
    placed = tool_file(tmp_path / owner, suffix)

    assert placed == tmp_path / expected


def test_a_registry_files_sit_in_the_nearest_tool_directory_at_or_above_it(tmp_path):
    mark_root(tmp_path / "workspace")
    registry = tmp_path / "workspace" / "kraken" / "registry.toml"

    placed = scheduler_file(registry, ".lock")

    assert placed == tmp_path / "workspace" / ".rtlab" / "kraken" / "registry.lock"


def test_a_registry_with_no_tool_directory_above_it_gets_one_beside_it(tmp_path):
    registry = tmp_path / "kraken" / "registry.toml"

    placed = scheduler_file(registry, ".launches.json")

    assert placed == tmp_path / "kraken" / ".rtlab" / "registry.launches.json"


def test_the_workspace_is_the_nearest_folder_above_the_bot_holding_a_tool_directory(
    tmp_path,
):
    mark_root(tmp_path / "workspace")
    mark_root(tmp_path / "workspace" / "kraken")
    owner = tmp_path / "workspace" / "kraken" / "trend" / "bot.toml"

    assert workspace_root(owner) == tmp_path / "workspace" / "kraken"


def test_the_tool_directory_of_the_bots_own_folder_marks_no_workspace(tmp_path):
    mark_root(tmp_path / "workspace")
    mark_root(tmp_path / "workspace" / "pilot")
    owner = tmp_path / "workspace" / "pilot" / "bot.toml"

    assert workspace_root(owner) == tmp_path / "workspace"


def test_a_bot_nested_in_the_folder_of_a_bot_that_has_run_keeps_the_outer_workspace(
    tmp_path,
):
    mark_root(tmp_path / "workspace")
    mark_root(tmp_path / "workspace" / "a")
    (tmp_path / "workspace" / "a" / "bot.toml").write_text("")
    owner = tmp_path / "workspace" / "a" / "b" / "bot.toml"

    assert workspace_root(owner) == tmp_path / "workspace"


def test_a_workspace_root_holding_a_bot_of_its_own_stays_the_workspace(tmp_path):
    mark_root(tmp_path / "workspace")
    (tmp_path / "workspace" / "bot.toml").write_text("")
    owner = tmp_path / "workspace" / "a" / "bot.toml"

    assert workspace_root(owner) == tmp_path / "workspace"


def test_a_bot_nested_two_folders_below_a_root_holding_a_bot_keeps_the_root(tmp_path):
    mark_root(tmp_path / "workspace")
    (tmp_path / "workspace" / "bot.toml").write_text("")
    mark_root(tmp_path / "workspace" / "a")
    (tmp_path / "workspace" / "a" / "bot.toml").write_text("")
    owner = tmp_path / "workspace" / "a" / "b" / "bot.toml"

    assert workspace_root(owner) == tmp_path / "workspace"


def test_a_bot_with_no_tool_directory_above_it_is_its_own_workspace(tmp_path):
    mark_root(tmp_path / "pilot")
    owner = tmp_path / "pilot" / "bot.toml"

    assert workspace_root(owner) == tmp_path / "pilot"


def test_the_workspace_of_a_relative_path_is_the_resolved_folder(tmp_path, monkeypatch):
    mark_root(tmp_path / "workspace")
    monkeypatch.chdir(tmp_path / "workspace")

    assert workspace_root(Path("pilot") / "bot.toml") == tmp_path / "workspace"
