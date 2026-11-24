import logging
import re
from functools import partial

import pytest
from live_bot_helpers import (
    SOL,
    THREE,
    TWO,
    bot_loaded_from,
    bot_toml,
    copy_of,
    save,
    save_bot,
    section,
    seed_bot,
)
from live_cycle_helpers import BTC, ETH, messages_at

from robottraderslab.exceptions import BotCriticalError
from robottraderslab.live.working_copy import (
    RunningConfig,
    TouchedSymbols,
    load_running_config,
    read_working_copy,
    touched_symbols,
)


def test_a_symbol_the_edit_no_longer_names_is_touched():
    touched = touched_symbols(bot_toml(*TWO).encode(), bot_toml(section(BTC)).encode())

    assert touched == TouchedSymbols(frozenset({ETH}), every_symbol=False)


def test_a_symbol_only_the_edit_names_is_touched():
    touched = touched_symbols(bot_toml(*TWO).encode(), bot_toml(*THREE).encode())

    assert touched == TouchedSymbols(frozenset({SOL}), every_symbol=False)


def test_a_symbols_sections_are_compared_in_order():
    working_copy = bot_toml(
        section(BTC, quantity=1.0), section(BTC, quantity=2.0), section(ETH)
    )
    swapped = bot_toml(
        section(BTC, quantity=2.0), section(BTC, quantity=1.0), section(ETH)
    )

    touched = touched_symbols(working_copy.encode(), swapped.encode())

    assert touched == TouchedSymbols(frozenset({BTC}), every_symbol=False)


def test_profile_sections_that_are_no_list_touch_every_symbol():
    touched = touched_symbols(
        b"[strategy]\nprofiles = 1\n", b"[strategy]\nprofiles = 2\n"
    )

    assert touched.every_symbol


def test_a_strategy_that_is_no_table_touches_every_symbol():
    touched = touched_symbols(bot_toml(*TWO).encode(), b"strategy = 1\n")

    assert touched.every_symbol


@pytest.mark.parametrize(
    ("read", "level"),
    [
        (read_working_copy, logging.WARNING),
        (partial(read_working_copy, level=logging.DEBUG), logging.DEBUG),
    ],
)
def test_a_copy_that_cannot_be_read_counts_as_none_and_logs_once_at_the_level_asked(
    tmp_path, caplog, read, level
):
    unreadable = tmp_path / "bot.working.toml"
    unreadable.mkdir()

    with caplog.at_level(logging.DEBUG):
        working_copy = read(unreadable)

    assert working_copy is None
    assert len(messages_at(caplog, level)) == 1
    assert len(caplog.records) == 1


def test_a_bot_with_no_working_copy_loads_its_bot_toml(tmp_path):
    bot_toml_text = save_bot(tmp_path, *TWO)

    loaded = load_running_config(tmp_path)

    assert loaded == RunningConfig(
        bot_loaded_from(tmp_path, bot_toml_text), bot_toml_text, None
    )


def test_a_bot_with_a_working_copy_loads_the_copy_over_a_different_bot_toml(tmp_path):
    copy_text = seed_bot(tmp_path, *TWO)
    save_bot(tmp_path, *THREE)

    loaded = load_running_config(tmp_path)

    assert loaded == RunningConfig(
        bot_loaded_from(tmp_path, copy_text), copy_text, copy_of(tmp_path).resolve()
    )


def test_a_bot_toml_that_is_not_valid_toml_leaves_the_copy_loaded(tmp_path):
    copy_text = seed_bot(tmp_path, *TWO)
    save(tmp_path, "[strategy\n")

    loaded = load_running_config(tmp_path)

    assert loaded == RunningConfig(
        bot_loaded_from(tmp_path, copy_text), copy_text, copy_of(tmp_path).resolve()
    )


def test_a_missing_bot_toml_leaves_the_copy_loaded(tmp_path):
    copy_text = seed_bot(tmp_path, *TWO)
    (tmp_path / "bot.toml").unlink()

    loaded = load_running_config(tmp_path)

    assert loaded == RunningConfig(
        bot_loaded_from(tmp_path, copy_text), copy_text, copy_of(tmp_path).resolve()
    )


def test_a_copy_that_is_not_valid_toml_counts_as_none_and_the_bot_toml_loads(
    tmp_path, caplog
):
    bot_toml_text = save_bot(tmp_path, *TWO)
    copy_of(tmp_path).parent.mkdir()
    copy_of(tmp_path).write_text("[strategy\n")

    with caplog.at_level(logging.WARNING):
        loaded = load_running_config(tmp_path)

    assert loaded == RunningConfig(
        bot_loaded_from(tmp_path, bot_toml_text), bot_toml_text, None
    )
    (warning,) = messages_at(caplog, logging.WARNING)
    assert str(copy_of(tmp_path).resolve()) in warning


def test_a_copy_the_loader_refuses_is_named_in_the_error(tmp_path):
    seed_bot(tmp_path, *TWO, head='name = "renamed"\n')
    save_bot(tmp_path, *TWO)

    with pytest.raises(BotCriticalError) as refused:
        load_running_config(tmp_path)

    assert str(refused.value).startswith(
        f"Working copy `{copy_of(tmp_path).resolve()}`, loaded as `bot.toml`: "
    )


def test_a_bot_with_neither_copy_nor_bot_toml_is_refused(tmp_path):
    with pytest.raises(BotCriticalError, match="There is no `bot.toml`"):
        load_running_config(tmp_path)


def test_a_path_naming_a_file_is_refused(tmp_path):
    save_bot(tmp_path, *TWO)

    with pytest.raises(BotCriticalError, match=re.escape("names a file")):
        load_running_config(tmp_path / "bot.toml")
