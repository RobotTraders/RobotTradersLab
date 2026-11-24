import logging
from collections.abc import Collection, Mapping
from types import MappingProxyType
from unittest.mock import Mock, call

import pytest

from robottraderslab import Symbol
from robottraderslab._internal import assume_margin_targets, hold_margin_targets
from robottraderslab.exceptions import (
    BotCriticalError,
    ExchangeCriticalError,
    ExchangeRecoverableError,
)
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
from robottraderslab_core.engine import AccountProtocol

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


async def _hold(
    account: FuturesAccount, target: MarginSettings, held: MarginSettings
) -> AccountSnapshot:
    requirements, snapshot = _declared(account, target, held)

    await _hold_all(account, requirements, snapshot)

    return snapshot


async def test_a_symbol_at_its_targets_is_left_alone(account, exchange):
    await _hold(account, ISOLATED_5X, ISOLATED_5X)

    exchange.set_margin_mode.assert_not_awaited()
    exchange.set_leverage.assert_not_awaited()


async def test_a_drifted_symbol_gets_its_margin_mode_before_its_leverage(
    account, exchange
):
    await _hold(account, ISOLATED_5X, CROSS_3X)

    assert exchange.mock_calls == [
        call.set_margin_mode(BTC, MarginMode.ISOLATED),
        call.set_leverage(BTC, 5.0),
    ]


async def test_a_changed_margin_mode_gets_the_leverage_even_when_the_read_matched(
    account, exchange
):
    await _hold(
        account,
        ISOLATED_5X,
        MarginSettings(leverage=5.0, margin_mode=MarginMode.CROSS),
    )

    assert exchange.mock_calls == [
        call.set_margin_mode(BTC, MarginMode.ISOLATED),
        call.set_leverage(BTC, 5.0),
    ]


async def test_a_symbol_at_its_margin_mode_gets_its_leverage_alone(account, exchange):
    await _hold(
        account,
        ISOLATED_5X,
        MarginSettings(leverage=3.0, margin_mode=MarginMode.ISOLATED),
    )

    assert exchange.mock_calls == [call.set_leverage(BTC, 5.0)]


async def test_every_drifted_symbol_of_the_account_is_set(account, exchange):
    requirements = AccountRequirements()
    requirements.add(account, margin_targets={BTC: ISOLATED_5X, ETH: ISOLATED_5X})
    held = MarginSettings(leverage=3.0, margin_mode=MarginMode.ISOLATED)
    snapshot = AccountSnapshot(
        account_name=account.name, margin_settings={BTC: held, ETH: held}
    )

    await _hold_all(account, requirements, snapshot)

    exchange.set_leverage.assert_any_await(BTC, 5.0)
    exchange.set_leverage.assert_any_await(ETH, 5.0)


async def test_a_target_leverage_of_none_sets_no_leverage(account, exchange):
    target = MarginSettings(leverage=None, margin_mode=MarginMode.CROSS)

    await _hold(account, target, CROSS_3X)

    exchange.set_leverage.assert_not_awaited()


async def test_the_snapshot_carries_the_targets_the_venue_accepted(account):
    snapshot = await _hold(account, ISOLATED_5X, CROSS_3X)

    assert snapshot.margin_settings(BTC) == ISOLATED_5X


async def test_a_refused_setter_leaves_the_venue_setting_in_the_snapshot(
    account, exchange, caplog
):
    exchange.set_leverage.side_effect = ExchangeRecoverableError("position open")

    with caplog.at_level(logging.WARNING):
        snapshot = await _hold(account, ISOLATED_5X, CROSS_3X)

    assert snapshot.margin_settings(BTC) == MarginSettings(
        leverage=3.0, margin_mode=MarginMode.ISOLATED
    )
    assert "SetLeverageAction (BTC/USDT:USDT)" in caplog.text


async def test_a_venue_refusing_the_key_stops_the_run(account, exchange):
    exchange.set_margin_mode.side_effect = ExchangeCriticalError("invalid key")

    with pytest.raises(ExchangeCriticalError):
        await _hold(account, ISOLATED_5X, CROSS_3X)


def test_assumed_targets_reach_the_snapshot_with_no_setter_run(account, exchange):
    snapshot = _assume(account, ISOLATED_5X, CROSS_3X)

    assert snapshot.margin_settings(BTC) == ISOLATED_5X
    assert exchange.mock_calls == []


def test_an_assumed_target_leaving_the_leverage_keeps_the_leverage_read(account):
    target = MarginSettings(leverage=None, margin_mode=MarginMode.ISOLATED)

    snapshot = _assume(account, target, CROSS_3X)

    assert snapshot.margin_settings(BTC) == MarginSettings(
        leverage=3.0, margin_mode=MarginMode.ISOLATED
    )


def test_a_symbol_the_read_left_out_is_given_no_assumed_target(account):
    requirements = _requiring(account, ISOLATED_5X)
    snapshot = AccountSnapshot(account_name=account.name, margin_settings={})

    _assume_targets(account, requirements, snapshot)

    with pytest.raises(BotCriticalError, match="not declared"):
        snapshot.margin_settings(BTC)


def test_an_account_declaring_no_target_needs_no_snapshot_to_assume_none(account):
    requirements = AccountRequirements()
    requirements.add(account, symbols=[BTC])

    assume_margin_targets(requirements, AccountSnapshots({}))


def test_a_symbol_closed_to_every_order_keeps_the_settings_read(account):
    requirements, snapshot = _declared(account, ISOLATED_5X, CROSS_3X)

    _assume_targets(account, requirements, snapshot, closed={account: [BTC]})

    assert snapshot.margin_settings(BTC) == CROSS_3X


def _assume(
    account: FuturesAccount, target: MarginSettings, held: MarginSettings
) -> AccountSnapshot:
    requirements, snapshot = _declared(account, target, held)

    _assume_targets(account, requirements, snapshot)

    return snapshot


def _assume_targets(
    account: FuturesAccount,
    requirements: AccountRequirements,
    snapshot: AccountSnapshot,
    closed: Mapping[AccountProtocol, Collection[Symbol]] = MappingProxyType({}),
) -> None:
    assume_margin_targets(
        requirements, _account_snapshots(account, snapshot), closed=closed
    )


def _declared(
    account: FuturesAccount, target: MarginSettings, held: MarginSettings
) -> tuple[AccountRequirements, AccountSnapshot]:
    snapshot = AccountSnapshot(account_name=account.name, margin_settings={BTC: held})
    return _requiring(account, target), snapshot


async def _hold_all(
    account: FuturesAccount,
    requirements: AccountRequirements,
    snapshot: AccountSnapshot,
) -> None:
    await hold_margin_targets(requirements, _account_snapshots(account, snapshot))


def _requiring(account: FuturesAccount, target: MarginSettings) -> AccountRequirements:
    requirements = AccountRequirements()
    requirements.add(account, margin_targets={BTC: target})
    return requirements


def _account_snapshots(
    account: FuturesAccount, snapshot: AccountSnapshot
) -> AccountSnapshots:
    return AccountSnapshots({account.name: snapshot})
