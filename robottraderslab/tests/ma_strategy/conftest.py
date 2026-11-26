from collections.abc import Callable, Mapping
from datetime import datetime
from unittest.mock import Mock

import pytest
from tracker_helpers import tracker_reporting

from robottraderslab import Symbol
from robottraderslab.futures import FuturesAccount, FuturesMarketOrderAction
from robottraderslab.ma_strategy.futures_ma_strategy import FuturesMAStrategy
from robottraderslab.strategies import (
    AccountSnapshots,
    BookKeeper,
    OHLCVs,
    PositionSide,
    TrackedPosition,
    TradingMode,
    TradingSystem,
)
from robottraderslab.strategies.futures import MarginSettings

BTC_USDT = Symbol.create("BTC/USDT:USDT")


@pytest.fixture
def mock_exchange(make_venue: Callable[..., Mock]) -> Mock:
    return make_venue()


@pytest.fixture
async def mock_account(mock_exchange: Mock) -> FuturesAccount:
    account = FuturesAccount(exchange=mock_exchange)
    await account._read_instruments([BTC_USDT])
    return account


@pytest.fixture
def backtest_trading_system() -> TradingSystem:
    return TradingSystem(trading_mode=TradingMode.BACKTEST)


@pytest.fixture
def make_booked_cycle(
    make_account_snapshots: Callable[..., AccountSnapshots],
) -> Callable[..., BookKeeper]:
    def book(
        ma_strategy: FuturesMAStrategy,
        ohlcvs: OHLCVs,
        booked_at: str = "2024-01-01 12:00:00",
        margin_settings: Mapping[Symbol, MarginSettings] | None = None,
    ) -> BookKeeper:
        """Book one candle of the strategy over an account holding no position.

        Args:
            booked_at: The candle's timestamp, as `datetime.fromisoformat` reads it.
            margin_settings: What the account reports for each symbol; None
                reports the leverage and margin mode of the strategy's profiles.
        """
        bookkeeper = BookKeeper()
        if margin_settings is None:
            margin_settings = {
                profile.symbol: MarginSettings(
                    leverage=profile.leverage, margin_mode=profile.margin_mode
                )
                for profile in ma_strategy.profiles
            }
        ma_strategy.book_trading_actions(
            ohlcvs,
            make_account_snapshots(
                ma_strategy.account.name,
                100_000.0,
                positions={},
                margin_settings=margin_settings,
            ),
            datetime.fromisoformat(booked_at),
            bookkeeper,
            [ma_strategy.profiles[0].timeframe],
        )
        return bookkeeper

    return book


@pytest.fixture
def only_order_action() -> Callable[[BookKeeper], FuturesMarketOrderAction]:
    def only(bookkeeper: BookKeeper) -> FuturesMarketOrderAction:
        """Return the one market order a bookkeeper holds."""
        order_actions = [
            action
            for action in bookkeeper.list_actions()
            if isinstance(action, FuturesMarketOrderAction)
        ]
        assert len(order_actions) == 1
        return order_actions[0]

    return only


@pytest.fixture
def make_booked_order(
    make_booked_cycle: Callable[..., BookKeeper],
    only_order_action: Callable[[BookKeeper], FuturesMarketOrderAction],
) -> Callable[..., FuturesMarketOrderAction]:
    def book(
        ma_strategy: FuturesMAStrategy,
        ohlcvs: OHLCVs,
        booked_at: str = "2024-01-01 12:00:00",
    ) -> FuturesMarketOrderAction:
        """Book one candle of the strategy and return the one market order it books.

        Args:
            booked_at: The candle's timestamp, as `datetime.fromisoformat` reads it.
        """
        return only_order_action(make_booked_cycle(ma_strategy, ohlcvs, booked_at))

    return book


@pytest.fixture
def hold_position() -> Callable[[FuturesMAStrategy, PositionSide, float], None]:
    def hold(
        ma_strategy: FuturesMAStrategy, position_side: PositionSide, quantity: float
    ) -> None:
        """Make the strategy's position tracker report a position held on one side."""
        ma_strategy._position_tracker = tracker_reporting(
            TrackedPosition(side=position_side, quantity=quantity)
        )

    return hold
