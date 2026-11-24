import logging
from collections.abc import Callable
from pathlib import Path

import pytest

from robottraderslab._internal import (
    drop_dead_locks,
    hold_instance_lock,
    hold_scheduler_lock,
)
from robottraderslab_core.engine import try_lock


@pytest.fixture
def owner(tmp_path: Path) -> Path:
    return tmp_path / "bot.toml"


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / ".rtlab").mkdir()
    return tmp_path


def _bot_config(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    config = directory / "envelope-bitget-demo.toml"
    config.write_text("[strategy]\n")
    return config


def test_a_second_holder_while_the_first_holds(owner):
    with hold_instance_lock(owner) as first, hold_instance_lock(owner) as second:
        pass

    assert first
    assert not second


def test_the_next_holder_after_the_block_ends(owner):
    with hold_instance_lock(owner):
        pass

    with hold_instance_lock(owner) as held:
        pass

    assert held


def test_the_next_holder_after_a_crash_in_the_block(owner):
    with pytest.raises(RuntimeError), hold_instance_lock(owner):
        raise RuntimeError("the cycle crashed")

    with hold_instance_lock(owner) as held:
        pass

    assert held


def test_a_relative_and_an_absolute_path_to_the_owner_share_the_lock(
    owner, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    with hold_instance_lock(Path("bot.toml")), hold_instance_lock(owner) as second:
        pass

    assert not second


def test_the_lock_file_lands_in_the_tool_directory_inside_the_config_folder(
    workspace,
):
    config = _bot_config(workspace / "pilot")

    with hold_instance_lock(config):
        pass

    assert (workspace / "pilot" / ".rtlab" / "envelope-bitget-demo.lock").is_file()


def test_a_tree_holding_no_tool_directory_gets_one_beside_the_config(tmp_path):
    config = _bot_config(tmp_path / "pilot")

    with hold_instance_lock(config):
        pass

    assert (tmp_path / "pilot" / ".rtlab" / "envelope-bitget-demo.lock").is_file()


def test_two_bots_named_alike_in_two_folders_hold_their_locks_at_once(workspace):
    pilot = _bot_config(workspace / "pilot")
    limit_pilot = _bot_config(workspace / "pilot" / "limit")

    with (
        hold_instance_lock(pilot) as pilot_held,
        hold_instance_lock(limit_pilot) as limit_held,
    ):
        pass

    assert pilot_held
    assert limit_held


def test_an_earlier_lock_and_the_folders_it_leaves_empty_are_swept(
    workspace, create_earlier_lock, names_in_tool_directory
):
    config = _bot_config(workspace / "kraken" / "trend")
    earlier = create_earlier_lock(workspace, config)

    drop_dead_locks(workspace / "registry.toml", [config], [])

    assert not earlier.exists()
    assert names_in_tool_directory(workspace) == []


def test_an_earlier_folder_holding_an_unrelated_file_keeps_it(
    workspace, create_earlier_lock
):
    config = _bot_config(workspace / "pilot")
    earlier = create_earlier_lock(workspace, config)
    notes = earlier.parent / "notes.txt"
    notes.write_text("kept")

    drop_dead_locks(workspace / "registry.toml", [config], [])

    assert not earlier.exists()
    assert notes.read_text() == "kept"


def test_a_lock_that_cannot_be_opened_is_reported_and_the_sweep_goes_on(
    workspace, create_earlier_lock, caplog
):
    config = _bot_config(workspace / "pilot")
    earlier = create_earlier_lock(workspace, config)
    (workspace / ".rtlab" / "stale.lock").mkdir()

    with caplog.at_level(logging.WARNING):
        drop_dead_locks(workspace / "registry.toml", [config], [])

    assert not earlier.exists()
    assert "Cannot sweep the lock file" in caplog.text


def test_the_lock_of_another_registry_in_the_swept_folder_is_kept(workspace):
    other_registry = workspace / "kraken" / "other.toml"
    other_registry.parent.mkdir()
    other_registry.write_text("")
    with hold_scheduler_lock(other_registry):
        pass

    drop_dead_locks(workspace / "registry.toml", [], [])

    assert (workspace / ".rtlab" / "kraken" / "other.lock").is_file()


def test_a_lock_of_a_config_deleted_beside_the_swept_folder_is_swept(
    workspace, create_lock_file
):
    _bot_config_with_lock_deleted(workspace, create_lock_file)

    drop_dead_locks(workspace / "registry.toml", [], [])

    assert not (workspace / ".rtlab" / "envelope-bitget-demo.lock").exists()


def test_a_lock_recording_no_owner_is_kept(workspace):
    stale = workspace / ".rtlab" / "from-an-older-engine.lock"
    stale.write_text("")

    drop_dead_locks(workspace / "scheduler.toml", [], [])

    assert stale.is_file()


def test_a_lock_of_a_config_beside_the_swept_folder_is_kept(
    workspace, create_lock_file
):
    config = _bot_config(workspace)
    create_lock_file(config)

    drop_dead_locks(workspace / "registry.toml", [config], [])

    assert (workspace / ".rtlab" / "envelope-bitget-demo.lock").is_file()


def test_a_lock_of_a_config_in_a_folder_named_in_another_alphabet_is_kept(
    tmp_path, create_lock_file
):
    config = _bot_config(tmp_path / "Ð¿Ð¸Ð»Ð¾Ñ‚")
    create_lock_file(config)

    drop_dead_locks(tmp_path / "registry.toml", [config], [])

    assert (tmp_path / "Ð¿Ð¸Ð»Ð¾Ñ‚" / ".rtlab" / "envelope-bitget-demo.lock").is_file()


def test_a_held_lock_of_the_earlier_placement_is_kept(workspace, create_earlier_lock):
    config = _bot_config(workspace / "pilot")
    earlier = create_earlier_lock(workspace, config)

    with open(earlier, "a+", encoding="utf-8") as handle:
        assert try_lock(handle)
        drop_dead_locks(workspace / "registry.toml", [config], [])

    assert earlier.is_file()


def test_the_lock_of_a_bot_running_from_its_working_copy_is_kept_without_its_file(
    workspace, create_lock_file
):
    config = _bot_config_with_lock_deleted(workspace / "pilot", create_lock_file)

    drop_dead_locks(workspace / "registry.toml", [config], [config])

    assert (workspace / "pilot" / ".rtlab" / "envelope-bitget-demo.lock").is_file()


def _bot_config_with_lock_deleted(
    directory: Path, create_lock_file: Callable[[Path], None]
) -> Path:
    config = _bot_config(directory)
    create_lock_file(config)
    config.unlink()
    return config
