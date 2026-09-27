from datetime import UTC, datetime
from typing import Any
from unittest.mock import Mock

import pytest

from robottraderslab import Symbol, TimeFrame
from robottraderslab.exceptions import StrategyCriticalError
from robottraderslab.exchanges import Balance, FuturesExchangeProtocol
from robottraderslab.strategies import (
    AccountSnapshot,
    AccountSnapshots,
    BookKeeper,
    OHLCVs,
    StrategyRequirements,
)
from robottraderslab.strategies.futures import (
    FuturesAccount,
    SizingRule,
)

BTC_USDT = Symbol.create("BTC/USDT:USDT")
PRICE = 100.0


class _ProfileLessStrategy:
    def __init__(self, *, account: FuturesAccount, **settings: Any) -> None:
        self._account = account
        self._sizing = SizingRule.from_settings(settings)

    async def setup(self, requirements: StrategyRequirements) -> None:
        pass

    def generate_trading_signals(self, ohlcvs: OHLCVs) -> None:
        pass

    def book_trading_actions(
        self,
        ohlcvs: OHLCVs,
        account_snapshots: AccountSnapshots,
        timestamp: datetime,
        bookkeeper: BookKeeper,
        triggered_timeframes: list[TimeFrame],
    ) -> None:
        bookkeeper.add(
            self._account.long_entry(BTC_USDT).size(
                self._sizing, PRICE, account_snapshots.of(self._account)
            )
        )


@pytest.fixture
def account() -> FuturesAccount:
    exchange = Mock(spec=FuturesExchangeProtocol)
    exchange.placement_reserve_rate = 0.0
    return FuturesAccount(exchange)


def _snapshots(account: FuturesAccount) -> AccountSnapshots:
    snapshot = AccountSnapshot(
        account_name=account.name,
        balances={"USDT": Balance(locked=0.0, total=10_000.0)},
        conversion_rates={BTC_USDT: 1.0},
    )
    return AccountSnapshots({account.name: snapshot})


class TestAStrategyWithoutProfiles:
    def test_sizes_with_the_rule_its_keys_name(self, account):
        strategy = _ProfileLessStrategy(
            account=account, total_balance_ratio=0.2, fast_length=20
        )
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            OHLCVs({}),
            _snapshots(account),
            datetime(2026, 1, 1, tzinfo=UTC),
            bookkeeper,
            [],
        )

        (entry,) = bookkeeper.list_actions()
        assert entry.quantity == pytest.approx(20.0)

    def test_keys_naming_no_rule(self, account):
        with pytest.raises(StrategyCriticalError, match="names no sizing rule"):
            _ProfileLessStrategy(account=account, fast_length=20)
