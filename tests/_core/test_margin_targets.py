import asyncio
import logging
from unittest.mock import Mock, call

import pytest

from robottraderslab import Symbol
from robottraderslab._core import hold_margin_targets
from robottraderslab.exceptions import ExchangeCriticalError, ExchangeRecoverableError
from robottraderslab.exchanges import (
    FuturesExchangeProtocol,
    MarginMode,
    MarginSettings,
)
from robottraderslab.futures import FuturesAccount
from robottraderslab.strategies import (
    AccountRequirements,
    AccountSnapshot,
    AccountSnapshots,
)

BTC = Symbol.create("BTC/USDT:USDT")
ETH = Symbol.create("ETH/USDT:USDT")
ISOLATED_5X = MarginSettings(leverage=5.0, margin_mode=MarginMode.ISOLATED)
CROSS_3X = MarginSettings(leverage=3.0, margin_mode=MarginMode.CROSS)


@pytest.fixture
def exchange() -> Mock:
    return Mock(spec=FuturesExchangeProtocol)


@pytest.fixture
def account(exchange: Mock) -> FuturesAccount:
    return FuturesAccount(exchange, name="test")


def _hold(
    account: FuturesAccount, target: MarginSettings, held: MarginSettings
) -> AccountSnapshot:
    requirements = AccountRequirements()
    requirements.add(account, margin_targets={BTC: target})
    snapshot = AccountSnapshot(account_name=account.name, margin_settings={BTC: held})

    asyncio.run(hold_margin_targets(requirements, AccountSnapshots({"test": snapshot})))

    return snapshot


def test_a_symbol_at_its_targets_is_left_alone(account, exchange):
    _hold(account, ISOLATED_5X, ISOLATED_5X)

    exchange.set_margin_mode.assert_not_awaited()
    exchange.set_leverage.assert_not_awaited()


def test_a_drifted_symbol_gets_its_margin_mode_before_its_leverage(account, exchange):
    _hold(account, ISOLATED_5X, CROSS_3X)

    assert exchange.mock_calls == [
        call.set_margin_mode(BTC, MarginMode.ISOLATED),
        call.set_leverage(BTC, 5.0),
    ]


def test_a_changed_margin_mode_gets_the_leverage_even_when_the_read_matched(
    account, exchange
):
    _hold(
        account,
        ISOLATED_5X,
        MarginSettings(leverage=5.0, margin_mode=MarginMode.CROSS),
    )

    assert exchange.mock_calls == [
        call.set_margin_mode(BTC, MarginMode.ISOLATED),
        call.set_leverage(BTC, 5.0),
    ]


def test_a_symbol_at_its_margin_mode_gets_its_leverage_alone(account, exchange):
    _hold(
        account,
        ISOLATED_5X,
        MarginSettings(leverage=3.0, margin_mode=MarginMode.ISOLATED),
    )

    assert exchange.mock_calls == [call.set_leverage(BTC, 5.0)]


def test_every_drifted_symbol_of_the_account_is_set(account, exchange):
    requirements = AccountRequirements()
    requirements.add(account, margin_targets={BTC: ISOLATED_5X, ETH: ISOLATED_5X})
    held = MarginSettings(leverage=3.0, margin_mode=MarginMode.ISOLATED)
    snapshot = AccountSnapshot(
        account_name=account.name, margin_settings={BTC: held, ETH: held}
    )

    asyncio.run(hold_margin_targets(requirements, AccountSnapshots({"test": snapshot})))

    exchange.set_leverage.assert_any_await(BTC, 5.0)
    exchange.set_leverage.assert_any_await(ETH, 5.0)


def test_a_target_leverage_of_none_sets_no_leverage(account, exchange):
    target = MarginSettings(leverage=None, margin_mode=MarginMode.CROSS)

    _hold(account, target, CROSS_3X)

    exchange.set_leverage.assert_not_awaited()


def test_the_snapshot_carries_the_targets_the_venue_accepted(account):
    snapshot = _hold(account, ISOLATED_5X, CROSS_3X)

    assert snapshot.margin_settings(BTC) == ISOLATED_5X


def test_a_refused_setter_leaves_the_venue_setting_in_the_snapshot(
    account, exchange, caplog
):
    exchange.set_leverage.side_effect = ExchangeRecoverableError("position open")

    with caplog.at_level(logging.WARNING):
        snapshot = _hold(account, ISOLATED_5X, CROSS_3X)

    assert snapshot.margin_settings(BTC) == MarginSettings(
        leverage=3.0, margin_mode=MarginMode.ISOLATED
    )
    assert "SetLeverageAction (BTC/USDT:USDT)" in caplog.text


def test_a_venue_refusing_the_key_stops_the_run(account, exchange):
    exchange.set_margin_mode.side_effect = ExchangeCriticalError("invalid key")

    with pytest.raises(ExchangeCriticalError):
        _hold(account, ISOLATED_5X, CROSS_3X)
