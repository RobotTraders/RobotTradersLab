from pathlib import Path

import pytest

from robottraderslab._core import drop_dead_locks, hold_instance_lock


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


def test_the_lock_file_lands_in_the_tool_directory_the_workspace_holds(workspace):
    config = _bot_config(workspace / "pilot")

    with hold_instance_lock(config):
        pass

    assert (workspace / ".rtlab" / "pilot" / "envelope-bitget-demo.lock").is_file()


def test_a_config_nested_under_the_root_lands_in_the_matching_branch(workspace):
    config = _bot_config(workspace / "pilot" / "limit")

    with hold_instance_lock(config):
        pass

    assert (
        workspace / ".rtlab" / "pilot" / "limit" / "envelope-bitget-demo.lock"
    ).is_file()


def test_the_nearest_tool_directory_above_the_config_wins(workspace):
    (workspace / "pilot" / ".rtlab").mkdir(parents=True)
    config = _bot_config(workspace / "pilot" / "limit")

    with hold_instance_lock(config):
        pass

    assert (
        workspace / "pilot" / ".rtlab" / "limit" / "envelope-bitget-demo.lock"
    ).is_file()


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


def test_a_lock_of_a_deleted_config_is_swept(workspace):
    config = _bot_config(workspace / "pilot")
    with hold_instance_lock(config):
        pass
    config.unlink()

    drop_dead_locks([config])

    assert not (workspace / ".rtlab" / "pilot" / "envelope-bitget-demo.lock").exists()


def test_a_lock_recording_no_owner_is_kept(workspace):
    stale = workspace / ".rtlab" / "from-an-older-engine.lock"
    stale.write_text("")

    drop_dead_locks([workspace / "scheduler.toml"])

    assert stale.is_file()


def test_a_lock_of_a_config_still_there_is_kept(workspace):
    config = _bot_config(workspace / "pilot")
    with hold_instance_lock(config):
        pass

    drop_dead_locks([config])

    assert (workspace / ".rtlab" / "pilot" / "envelope-bitget-demo.lock").is_file()


def test_a_lock_of_a_config_in_a_folder_named_in_another_alphabet_is_kept(workspace):
    config = _bot_config(workspace / "пилот")
    with hold_instance_lock(config):
        pass

    drop_dead_locks([config])

    assert (workspace / ".rtlab" / "пилот" / "envelope-bitget-demo.lock").is_file()


def test_a_held_lock_of_a_deleted_config_is_kept(workspace):
    config = _bot_config(workspace / "pilot")

    with hold_instance_lock(config):
        config.unlink()
        drop_dead_locks([config])

    assert (workspace / ".rtlab" / "pilot" / "envelope-bitget-demo.lock").is_file()
