from datetime import UTC, datetime
from typing import Any

import pytest

from robottraderslab import Symbol, TimeFrame
from robottraderslab.exceptions import ProfileConfigError
from robottraderslab.strategies import (
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
async def account(make_venue) -> FuturesAccount:
    exchange = make_venue()
    account = FuturesAccount(exchange)
    await account._read_instruments([BTC_USDT])
    return account


class TestAStrategyWithoutProfiles:
    def test_sizes_with_the_rule_its_keys_name(self, account, make_account_snapshots):
        strategy = _ProfileLessStrategy(
            account=account, total_balance_ratio=0.2, fast_length=20
        )
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            OHLCVs({}),
            make_account_snapshots(
                account.name, 10_000.0, conversion_rates={BTC_USDT: 1.0}
            ),
            datetime(2026, 1, 1, tzinfo=UTC),
            bookkeeper,
            [],
        )

        (entry,) = bookkeeper.list_actions()
        assert entry.quantity == pytest.approx(20.0)

    def test_keys_naming_no_rule(self, account):
        with pytest.raises(ProfileConfigError, match="names no sizing rule"):
            _ProfileLessStrategy(account=account, fast_length=20)

    def test_a_value_the_rule_refuses(self, account):
        with pytest.raises(
            ProfileConfigError, match="notional: `notional` must be greater than 0"
        ):
            _ProfileLessStrategy(account=account, notional=-1)
