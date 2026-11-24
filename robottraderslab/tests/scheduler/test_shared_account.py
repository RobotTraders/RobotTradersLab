import logging
from collections.abc import Callable, Iterable
from pathlib import Path

import pytest

from robottraderslab.exceptions import BotCriticalError
from robottraderslab.live import working_copy_file
from robottraderslab.scheduler.shared_account import refuse_shared_account


def test_the_refusal_names_every_configuration_sharing_the_digest(write_bot):
    configs = [
        write_bot(file, name="grid", account="main")
        for file in ("first", "second", "third")
    ]

    with pytest.raises(BotCriticalError) as refusal:
        refuse_shared_account(configs)

    assert all(str(config) in str(refusal.value) for config in configs)


def test_bots_in_folders_named_alike_are_refused(write_bot):
    configs = [
        write_bot("alpha/grid", account="main"),
        write_bot("beta/grid", account="main"),
    ]

    with pytest.raises(BotCriticalError, match="a folder of a distinct name"):
        refuse_shared_account(configs)


def test_bots_in_folders_named_apart_are_accepted(write_bot):
    configs = [
        write_bot("alpha", account="main"),
        write_bot("beta", account="main"),
    ]

    refuse_shared_account(configs)


def test_bots_sharing_the_leading_digest_bytes_on_one_account_are_refused(write_bot):
    configs = [
        write_bot("first", name="jd", account="main"),
        write_bot("second", name="ks", account="main"),
    ]

    with pytest.raises(BotCriticalError, match="their folder names share an identity"):
        refuse_shared_account(configs)


def test_bots_sharing_the_leading_digest_bytes_on_different_accounts_are_accepted(
    write_bot,
):
    configs = [
        write_bot("first", name="jd", account="main"),
        write_bot("second", name="ks", account="spare"),
    ]

    refuse_shared_account(configs)


def test_bots_with_different_names_on_one_account_are_accepted(write_bot):
    configs = [
        write_bot("first", name="grid", account="main"),
        write_bot("second", name="scalp", account="main"),
    ]

    refuse_shared_account(configs)


def test_bots_with_one_name_on_different_accounts_are_accepted(write_bot):
    configs = [
        write_bot("first", name="grid", account="main"),
        write_bot("second", name="grid", account="spare"),
    ]

    refuse_shared_account(configs)


def test_bots_with_one_name_on_different_exchanges_are_accepted(write_bot):
    configs = [
        write_bot("first", name="grid", exchange="bitget"),
        write_bot("second", name="grid", exchange="kraken"),
    ]

    refuse_shared_account(configs)


def test_an_account_naming_none_is_the_exchanges_own(write_bot):
    configs = [
        write_bot("first", name="grid"),
        write_bot("second", name="grid", account="bitget"),
    ]

    with pytest.raises(BotCriticalError, match="their folder names share an identity"):
        refuse_shared_account(configs)


def test_bots_naming_no_account_are_accepted(write_bot):
    configs = [
        write_bot("first", name="grid", exchange=None),
        write_bot("second", name="grid", exchange=None),
    ]

    refuse_shared_account(configs)


@pytest.mark.parametrize(
    ("first_exchange", "venue"),
    [(None, "`pilot`, and"), ("bitget", "`pilot` on bitget, and")],
)
def test_bots_naming_one_secret_are_refused_whatever_exchange_one_declares(
    write_bot, first_exchange, venue
):
    configs = [
        write_bot("first", name="grid", exchange=first_exchange, secret="pilot"),
        write_bot("second", name="grid", exchange=None, secret="pilot"),
    ]

    with pytest.raises(BotCriticalError) as refusal:
        refuse_shared_account(configs)

    assert f"account {venue}" in str(refusal.value)
    assert all(str(config) in str(refusal.value) for config in configs)


def test_bots_naming_different_secrets_are_accepted(write_bot):
    configs = [
        write_bot("first", name="grid", exchange=None, secret="pilot"),
        write_bot("second", name="grid", exchange=None, secret="spare"),
    ]

    refuse_shared_account(configs)


def test_a_secret_wins_over_the_account_name(write_bot):
    configs = [
        write_bot("first", name="grid", account="main", secret="pilot"),
        write_bot("second", name="grid", account="main", secret="spare"),
    ]

    refuse_shared_account(configs)


def test_bots_naming_one_account_alone_are_refused(write_bot):
    configs = [
        write_bot(file, name="grid", exchange=None, account="kraken-pilot")
        for file in ("first", "second")
    ]

    with pytest.raises(BotCriticalError, match="their folder names share an identity"):
        refuse_shared_account(configs)


def test_a_non_ascii_name_is_read_as_utf_8(write_bot):
    configs = [
        write_bot("first", name="gridō", account="main"),
        write_bot("second", name="gridō", account="main"),
    ]

    with pytest.raises(BotCriticalError, match="their folder names share an identity"):
        refuse_shared_account(configs)


def test_a_configuration_that_is_not_toml_is_accepted(write_bot, tmp_path):
    broken = tmp_path / "broken.toml"
    broken.write_text("name = [\n")

    refuse_shared_account([broken, write_bot("sound", name="grid")])


def test_a_configuration_that_is_not_toml_is_logged(tmp_path, caplog):
    broken = tmp_path / "broken.toml"
    broken.write_text("name = [\n")

    with caplog.at_level(logging.ERROR):
        refuse_shared_account([broken])

    assert str(broken) in caplog.text


def test_a_name_that_is_not_a_string_is_accepted(tmp_path):
    numbered = tmp_path / "numbered.toml"
    numbered.write_text('name = 7\n[live.trading_account]\nexchange = "bitget"\n')

    refuse_shared_account([numbered])


def test_the_refusal_names_no_venue_when_the_bots_declare_different_ones(write_bot):
    configs = [
        write_bot("first", name="grid", exchange="bitget", secret="pilot"),
        write_bot("second", name="grid", exchange="kraken", secret="pilot"),
    ]

    with pytest.raises(BotCriticalError) as refusal:
        refuse_shared_account(configs)

    assert "bitget" not in str(refusal.value)
    assert "kraken" not in str(refusal.value)


def test_bots_declaring_one_symbol_on_one_account_are_refused(write_bot):
    configs = [
        write_bot("first", name="grid", account="main", symbols=("BTC/USDT:USDT",)),
        write_bot(
            "second",
            name="scalp",
            account="main",
            symbols=("ETH/USDT:USDT", "BTC/USDT:USDT"),
        ),
    ]

    with pytest.raises(BotCriticalError) as refusal:
        refuse_shared_account(configs)

    assert "account `main` on bitget" in str(refusal.value)
    assert str(refusal.value) == (
        f"`{configs[0]}` and `{configs[1]}` trade `BTC/USDT:USDT` "
        "on the account `main` on bitget; "
        "give each bot its own symbols or its own account"
    )


def test_bots_declaring_distinct_symbols_on_one_account_are_accepted(write_bot):
    configs = [
        write_bot("first", name="grid", account="main", symbols=("BTC/USDT:USDT",)),
        write_bot("second", name="scalp", account="main", symbols=("ETH/USDT:USDT",)),
    ]

    refuse_shared_account(configs)


def test_bots_declaring_one_symbol_on_different_accounts_are_accepted(write_bot):
    configs = [
        write_bot("first", name="grid", account="main", symbols=("BTC/USDT:USDT",)),
        write_bot("second", name="scalp", account="spare", symbols=("BTC/USDT:USDT",)),
    ]

    refuse_shared_account(configs)


def test_a_bot_declaring_one_symbol_on_several_profiles_is_accepted(write_bot):
    configs = [
        write_bot(
            "first",
            name="grid",
            account="main",
            symbols=("BTC/USDT:USDT", "BTC/USDT:USDT"),
        )
    ]

    refuse_shared_account(configs)


BTC = "BTC/USDT:USDT"
ETH = "ETH/USDT:USDT"
SOL = "SOL/USDT:USDT"
XRP = "XRP/USDT:USDT"
REMEDY = "delete the working copy or run its bot by hand"


@pytest.fixture
def bot_with_working_copy(write_bot_with_working_copy: Callable[..., Path]) -> Path:
    return write_bot_with_working_copy(
        "first", name="grid", account="main", symbols=(BTC,)
    )


def test_working_copies_declaring_one_symbol_on_one_account_are_refused_naming_the_copies(
    edit_first_bot, keep_working_copy, two_bots_on_one_symbol, second_symbol
):
    bot_files = list(two_bots_on_one_symbol)
    copy_files = [keep_working_copy(bot_file) for bot_file in bot_files]
    edit_first_bot(second_symbol)

    refusal_text = _refusal_text(bot_files)

    assert all(
        f"the working copy `{copy_file}` of `{bot_file}`" in refusal_text
        for copy_file, bot_file in zip(copy_files, bot_files, strict=True)
    )
    assert REMEDY in refusal_text


def test_an_edit_naming_another_bots_symbol_is_refused_for_that_bot_alone(
    edit_first_bot, two_bots_with_working_copies, first_symbol, second_symbol
):
    first, second = two_bots_with_working_copies
    edit_first_bot(first_symbol, second_symbol)

    edit_refusals = refuse_shared_account([first, second])

    assert list(edit_refusals) == [first]
    assert edit_refusals[first] == (
        f"`{first}` adds `{second_symbol}` on the account `main` on bitget, "
        "which `scalp` already trades; "
        "remove it from one of the two bots or give each its own account"
    )


def test_an_edit_naming_the_symbol_of_a_bot_without_a_working_copy_is_refused(
    write_bot, edit_first_bot, bot_with_working_copy
):
    second = write_bot("second", name="scalp", account="main", symbols=(ETH,))
    edit_first_bot(BTC, ETH)

    edit_refusals = refuse_shared_account([bot_with_working_copy, second])

    assert edit_refusals[bot_with_working_copy] == (
        f"`{bot_with_working_copy}` adds `{ETH}` on the account `main` on bitget, "
        "which `scalp` already trades; "
        "remove it from one of the two bots or give each its own account"
    )


def test_two_bots_edited_to_name_the_same_new_symbol_are_both_refused(
    edit_first_bot,
    edit_second_bot,
    two_bots_with_working_copies,
    first_symbol,
    second_symbol,
):
    first, second = two_bots_with_working_copies
    edit_first_bot(first_symbol, SOL)
    edit_second_bot(second_symbol, SOL)

    edit_refusals = refuse_shared_account([first, second])

    assert set(edit_refusals) == {first, second}
    assert edit_refusals[first] == (
        f"`{first}` adds `{SOL}` on the account `main` on bitget, "
        f"which the edited `{second}` adds too; "
        "remove it from one of the two bots or give each its own account"
    )
    assert edit_refusals[second] == (
        f"`{second}` adds `{SOL}` on the account `main` on bitget, "
        f"which the edited `{first}` adds too; "
        "remove it from one of the two bots or give each its own account"
    )


def test_two_bots_edited_to_name_different_new_symbols_are_accepted(
    edit_first_bot,
    edit_second_bot,
    two_bots_with_working_copies,
    first_symbol,
    second_symbol,
):
    first, second = two_bots_with_working_copies
    edit_first_bot(first_symbol, SOL)
    edit_second_bot(second_symbol, XRP)

    assert refuse_shared_account([first, second]) == {}


def test_a_bot_keeps_a_symbol_its_copy_runs_when_another_edit_names_it(
    edit_first_bot,
    edit_second_bot,
    two_bots_with_working_copies,
    first_symbol,
    second_symbol,
):
    first, second = two_bots_with_working_copies
    edit_first_bot(first_symbol, SOL)
    edit_second_bot(second_symbol, first_symbol)

    edit_refusals = refuse_shared_account([first, second])

    assert list(edit_refusals) == [second]
    assert edit_refusals[second] == (
        f"`{second}` adds `{first_symbol}` on the account `main` on bitget, "
        "which `grid` already trades; "
        "remove it from one of the two bots or give each its own account"
    )


def test_an_edit_moving_a_bot_to_an_account_where_its_name_is_taken_is_refused_once(
    write_bot, write_bot_with_working_copy
):
    first = write_bot_with_working_copy("first", name="grid", account="main")
    second = write_bot_with_working_copy("second", name="grid", account="spare")
    write_bot("first", name="grid", account="spare")

    edit_refusals = refuse_shared_account([first, second])

    assert edit_refusals[first] == (
        f"`{first}` adds the account `spare` on bitget, "
        "where the folder name of `grid` shares an identity with its own; "
        "give each a folder of a distinct name"
    )


def test_a_bot_in_conflict_on_two_symbols_gets_one_line_with_both_sentences(
    edit_first_bot, write_bot_with_working_copy
):
    first = write_bot_with_working_copy(
        "first", name="grid", account="main", symbols=(BTC,)
    )
    second = write_bot_with_working_copy(
        "second", name="scalp", account="main", symbols=(ETH, SOL)
    )
    edit_first_bot(BTC, ETH, SOL)

    edit_refusal = refuse_shared_account([first, second])[first]

    assert edit_refusal == (
        f"`{first}` adds `{ETH}` on the account `main` on bitget, "
        "which `scalp` already trades; "
        "remove it from one of the two bots or give each its own account. "
        f"`{first}` adds `{SOL}` on the account `main` on bitget, "
        "which `scalp` already trades; "
        "remove it from one of the two bots or give each its own account"
    )


def test_a_refused_edit_is_logged_at_debug_alone(
    edit_first_bot, two_bots_with_working_copies, caplog, first_symbol, second_symbol
):
    first, second = two_bots_with_working_copies
    edit_first_bot(first_symbol, second_symbol)

    edit_refusals = _refuse_logging_nothing_at_info([first, second], caplog)
    with caplog.at_level(logging.DEBUG):
        refuse_shared_account([first, second])

    assert f"The edit of {first} is refused: {edit_refusals[first]}" in caplog.text


def test_an_edited_file_that_is_not_toml_is_left_to_its_bot_and_logged_at_debug(
    bot_with_working_copy, caplog
):
    bot_with_working_copy.write_text("name = [\n")

    edit_refusals = _refuse_logging_nothing_at_info([bot_with_working_copy], caplog)
    with caplog.at_level(logging.DEBUG):
        refuse_shared_account([bot_with_working_copy])

    assert edit_refusals == {}
    assert f"Cannot read {bot_with_working_copy} to compare" in caplog.text


def test_a_file_that_is_gone_beside_a_working_copy_is_not_an_edit(
    bot_with_working_copy, caplog
):
    bot_with_working_copy.unlink()

    edit_refusals = _refuse_logging_nothing_at_info([bot_with_working_copy], caplog)

    assert edit_refusals == {}


def test_a_bot_whose_file_is_gone_is_compared_by_its_working_copy(
    keep_only_working_copy, two_bots_on_one_symbol
):
    first, second = two_bots_on_one_symbol
    keep_only_working_copy(first)

    refusal_text = _refusal_text([first, second])

    assert f"the working copy `{working_copy_file(first)}` of `{first}`" in refusal_text
    assert REMEDY in refusal_text


def test_a_working_copy_that_is_not_toml_counts_as_none(
    write_unusable_copy, two_bots_on_one_symbol
):
    first, second = two_bots_on_one_symbol
    write_unusable_copy(first)

    refusal_text = _refusal_text([first, second])

    assert f"`{first}`" in refusal_text
    assert "the working copy" not in refusal_text


def test_an_edit_conflicting_with_a_running_bot_and_another_edit_names_both(
    edit_first_bot, edit_second_bot, write_bot_with_working_copy
):
    first = write_bot_with_working_copy(
        "first", name="grid", account="main", symbols=(BTC,)
    )
    second = write_bot_with_working_copy(
        "second", name="scalp", account="main", symbols=(SOL,)
    )
    third = write_bot_with_working_copy(
        "third", name="swing", account="main", symbols=(ETH,)
    )
    edit_first_bot(BTC, ETH)
    edit_second_bot(SOL, ETH)

    edit_refusal = refuse_shared_account([first, second, third])[first]

    assert edit_refusal == (
        f"`{first}` adds `{ETH}` on the account `main` on bitget, "
        "which `swing` already trades and the edited "
        f"`{second}` adds too; "
        "remove it from all but one of the bots or give each its own account"
    )


def test_a_working_copy_that_is_not_toml_is_reported_by_its_bot_and_not_the_check(
    write_bot, write_unusable_copy, caplog
):
    first = write_bot("first", name="grid", account="main", symbols=(BTC,))
    write_unusable_copy(first)

    _refuse_logging_nothing_at_info([first], caplog)


def _refusal_text(bot_files: Iterable[Path]) -> str:
    with pytest.raises(BotCriticalError) as excinfo:
        refuse_shared_account(bot_files)
    return str(excinfo.value)


def _refuse_logging_nothing_at_info(
    bot_files: Iterable[Path], caplog: pytest.LogCaptureFixture
) -> dict[Path, str]:
    with caplog.at_level(logging.INFO):
        edit_refusals = refuse_shared_account(bot_files)
    assert caplog.text == ""
    return edit_refusals
