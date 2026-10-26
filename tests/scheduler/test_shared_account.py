import logging

import pytest

from robottraderslab.exceptions import BotCriticalError
from robottraderslab.scheduler.shared_account import refuse_shared_account


def test_the_refusal_names_every_configuration_sharing_the_digest(write_bot):
    configs = [
        write_bot(f"{file}.toml", name="grid", account="main")
        for file in ("first", "second", "third")
    ]

    with pytest.raises(BotCriticalError) as refusal:
        refuse_shared_account(configs)

    assert all(str(config) in str(refusal.value) for config in configs)


def test_bots_declaring_no_name_share_their_file_stem(write_bot):
    configs = [
        write_bot("alpha/grid.toml", account="main"),
        write_bot("beta/grid.toml", account="main"),
    ]

    with pytest.raises(BotCriticalError, match="distinct top-level `name`"):
        refuse_shared_account(configs)


def test_bots_sharing_the_leading_digest_bytes_on_one_account_are_refused(write_bot):
    configs = [
        write_bot("first.toml", name="jd", account="main"),
        write_bot("second.toml", name="ks", account="main"),
    ]

    with pytest.raises(BotCriticalError, match="two leading bytes `8368`"):
        refuse_shared_account(configs)


def test_bots_sharing_the_leading_digest_bytes_on_different_accounts_are_accepted(
    write_bot,
):
    configs = [
        write_bot("first.toml", name="jd", account="main"),
        write_bot("second.toml", name="ks", account="spare"),
    ]

    refuse_shared_account(configs)


def test_bots_with_different_names_on_one_account_are_accepted(write_bot):
    configs = [
        write_bot("first.toml", name="grid", account="main"),
        write_bot("second.toml", name="scalp", account="main"),
    ]

    refuse_shared_account(configs)


def test_bots_with_one_name_on_different_accounts_are_accepted(write_bot):
    configs = [
        write_bot("first.toml", name="grid", account="main"),
        write_bot("second.toml", name="grid", account="spare"),
    ]

    refuse_shared_account(configs)


def test_bots_with_one_name_on_different_exchanges_are_accepted(write_bot):
    configs = [
        write_bot("first.toml", name="grid", exchange="bitget"),
        write_bot("second.toml", name="grid", exchange="kraken"),
    ]

    refuse_shared_account(configs)


def test_an_account_naming_none_is_the_exchanges_own(write_bot):
    configs = [
        write_bot("first.toml", name="grid"),
        write_bot("second.toml", name="grid", account="bitget"),
    ]

    with pytest.raises(BotCriticalError, match="same bot digest"):
        refuse_shared_account(configs)


def test_bots_naming_no_account_are_accepted(write_bot):
    configs = [
        write_bot("first.toml", name="grid", exchange=None),
        write_bot("second.toml", name="grid", exchange=None),
    ]

    refuse_shared_account(configs)


@pytest.mark.parametrize(
    ("first_exchange", "venue"),
    [(None, "`pilot` under"), ("bitget", "`pilot` on bitget")],
)
def test_bots_naming_one_secret_are_refused_whatever_exchange_one_declares(
    write_bot, first_exchange, venue
):
    configs = [
        write_bot("first.toml", name="grid", exchange=first_exchange, secret="pilot"),
        write_bot("second.toml", name="grid", exchange=None, secret="pilot"),
    ]

    with pytest.raises(BotCriticalError) as refusal:
        refuse_shared_account(configs)

    assert f"account {venue}" in str(refusal.value)
    assert all(str(config) in str(refusal.value) for config in configs)


def test_bots_naming_different_secrets_are_accepted(write_bot):
    configs = [
        write_bot("first.toml", name="grid", exchange=None, secret="pilot"),
        write_bot("second.toml", name="grid", exchange=None, secret="spare"),
    ]

    refuse_shared_account(configs)


def test_a_secret_wins_over_the_account_name(write_bot):
    configs = [
        write_bot("first.toml", name="grid", account="main", secret="pilot"),
        write_bot("second.toml", name="grid", account="main", secret="spare"),
    ]

    refuse_shared_account(configs)


def test_bots_naming_one_account_alone_are_refused(write_bot):
    configs = [
        write_bot(f"{file}.toml", name="grid", exchange=None, account="kraken-pilot")
        for file in ("first", "second")
    ]

    with pytest.raises(BotCriticalError, match="same bot digest"):
        refuse_shared_account(configs)


def test_a_non_ascii_name_is_read_as_utf_8(write_bot):
    configs = [
        write_bot("first.toml", name="gridō", account="main"),
        write_bot("second.toml", name="gridō", account="main"),
    ]

    with pytest.raises(BotCriticalError, match="same bot digest"):
        refuse_shared_account(configs)


def test_a_configuration_that_is_not_toml_is_accepted(write_bot, tmp_path):
    broken = tmp_path / "broken.toml"
    broken.write_text("name = [\n")

    refuse_shared_account([broken, write_bot("sound.toml", name="grid")])


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


def test_a_name_that_is_not_a_string_is_logged(tmp_path, caplog):
    numbered = tmp_path / "numbered.toml"
    numbered.write_text('name = 7\n[live.trading_account]\nexchange = "bitget"\n')

    with caplog.at_level(logging.ERROR):
        refuse_shared_account([numbered])

    assert str(numbered) in caplog.text


def test_the_refusal_names_no_venue_when_the_bots_declare_different_ones(write_bot):
    configs = [
        write_bot("first.toml", name="grid", exchange="bitget", secret="pilot"),
        write_bot("second.toml", name="grid", exchange="kraken", secret="pilot"),
    ]

    with pytest.raises(BotCriticalError) as refusal:
        refuse_shared_account(configs)

    assert " on " not in str(refusal.value)


def test_bots_declaring_one_symbol_on_one_account_are_refused(write_bot):
    configs = [
        write_bot(
            "first.toml", name="grid", account="main", symbols=("BTC/USDT:USDT",)
        ),
        write_bot(
            "second.toml",
            name="scalp",
            account="main",
            symbols=("ETH/USDT:USDT", "BTC/USDT:USDT"),
        ),
    ]

    with pytest.raises(BotCriticalError) as refusal:
        refuse_shared_account(configs)

    assert "account `main` on bitget" in str(refusal.value)
    assert "symbol `BTC/USDT:USDT`" in str(refusal.value)
    assert all(str(config) in str(refusal.value) for config in configs)


def test_bots_declaring_distinct_symbols_on_one_account_are_accepted(write_bot):
    configs = [
        write_bot(
            "first.toml", name="grid", account="main", symbols=("BTC/USDT:USDT",)
        ),
        write_bot(
            "second.toml", name="scalp", account="main", symbols=("ETH/USDT:USDT",)
        ),
    ]

    refuse_shared_account(configs)


def test_bots_declaring_one_symbol_on_different_accounts_are_accepted(write_bot):
    configs = [
        write_bot(
            "first.toml", name="grid", account="main", symbols=("BTC/USDT:USDT",)
        ),
        write_bot(
            "second.toml", name="scalp", account="spare", symbols=("BTC/USDT:USDT",)
        ),
    ]

    refuse_shared_account(configs)


def test_a_bot_declaring_one_symbol_on_several_profiles_is_accepted(write_bot):
    configs = [
        write_bot(
            "first.toml",
            name="grid",
            account="main",
            symbols=("BTC/USDT:USDT", "BTC/USDT:USDT"),
        )
    ]

    refuse_shared_account(configs)
