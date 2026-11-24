import asyncio
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any
from unittest.mock import Mock, create_autospec

import pandas as pd
import pytest

from robottraderslab import Symbol, TimeFrame
from robottraderslab.exchanges import (
    OrderSide,
    PositionSide,
    PositionSnapshot,
    VenueFill,
)
from robottraderslab.futures import FuturesAccount
from robottraderslab.live.livebot import LiveBot
from robottraderslab.strategies import (
    AccountSnapshots,
    BookKeeper,
    OHLCVs,
    StrategyProtocol,
    StrategyRequirements,
)
from robottraderslab_core.exchanges import OHLCVProviderProtocol

BTC = Symbol.create("BTC/USDT:USDT")
ETH = Symbol.create("ETH/USDT:USDT")


def long_position(symbol: Symbol, quantity: float) -> PositionSnapshot:
    return PositionSnapshot(
        symbol=symbol,
        side=PositionSide.LONG,
        quantity=quantity,
        average_entry_price=100.0,
        entry_time=datetime(2024, 1, 1, tzinfo=UTC),
        leverage=1.0,
        liquidation_price=1.0,
        stop_loss_price=None,
        take_profit_price=None,
    )


def make_bot_folder(directory: Path, name: str) -> Path:
    """Create the folder of a bot, holding the secrets file `creds.toml` that
    defines the account secret `my_account`.
    """
    folder = directory / name
    folder.mkdir()
    (folder / "creds.toml").write_text(
        '[[secrets]]\nname = "my_account"\napi_key = "shh"\n'
    )
    return folder


def candles(timestamps: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0] * len(timestamps),
            "high": [110.0] * len(timestamps),
            "low": [90.0] * len(timestamps),
            "close": [105.0] * len(timestamps),
            "volume": [1000.0] * len(timestamps),
        },
        index=pd.to_datetime(timestamps),
    )


class BookingStrategy(StrategyProtocol):
    market_type = "futures"

    def __init__(self, book: Callable[[BookKeeper], None]) -> None:
        self._book = book

    def generate_trading_signals(self, ohlcvs: OHLCVs) -> None:
        pass

    def book_trading_actions(
        self,
        ohlcvs: OHLCVs,
        account_snapshots: AccountSnapshots,
        timestamp: datetime,
        bookkeeper: BookKeeper,
        triggered_timeframes: list,
    ) -> None:
        self._book(bookkeeper)


class Booking(BookingStrategy):
    def __init__(
        self,
        account: FuturesAccount,
        book: Callable[[BookKeeper], None],
        **declared: bool,
    ) -> None:
        super().__init__(book)
        self._account = account
        self._declared = declared

    async def setup(self, requirements):
        for symbol in (BTC, ETH):
            requirements.ohlcv.add(symbol, "1d")
        requirements.account.add(self._account, symbols=[BTC, ETH], **self._declared)


def make_provider(
    dropped: Mapping[Symbol, BaseException] = MappingProxyType({}),
) -> Mock:
    """A candle provider answering every pair but those it fails for."""

    def fetch(symbol: Symbol, timeframe: TimeFrame) -> Any:
        if symbol in dropped:
            raise dropped[symbol]
        return candles(["2024-01-01"])

    provider = Mock(spec=OHLCVProviderProtocol)
    provider.fetch_ohlcv.side_effect = fetch
    return provider


def livebot(strategy: StrategyProtocol, *exit_only: Symbol, **options) -> LiveBot:
    return LiveBot(
        strategy,
        make_provider(),
        exit_only_symbols=frozenset(exit_only),
        **options,
    )


def run(strategy: StrategyProtocol, *exit_only: Symbol) -> None:
    asyncio.run(livebot(strategy, *exit_only).run())


def booking(
    account: FuturesAccount, book: Callable[[BookKeeper], None], **declared: bool
) -> StrategyProtocol:
    return Booking(
        account, book, **{"positions": True, "open_orders": True, **declared}
    )


def market_orders(exchange: Mock) -> list[tuple[Symbol, OrderSide, float]]:
    return [
        (placed.kwargs["symbol"], placed.kwargs["side"], placed.kwargs["quantity"])
        for placed in exchange.place_market_order.await_args_list
    ]


def cancelled_ids(exchange: Mock) -> list[str]:
    return [placed.args[1] for placed in exchange.cancel_order_by_id.await_args_list]


def requirements_failing_to_refresh(error: BaseException) -> StrategyRequirements:
    requirements = StrategyRequirements()
    requirements.tracker.refresh_all = create_autospec(
        requirements.tracker.refresh_all, side_effect=error
    )
    return requirements


async def bought(**_kwargs: object) -> VenueFill:
    return VenueFill(order_id="order-1", symbol=BTC, side=OrderSide.BUY, quantity=1.0)


def messages_at(caplog: pytest.LogCaptureFixture, level: int) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.levelno == level]


def logged_at_or_above(caplog: pytest.LogCaptureFixture, level: int) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.levelno >= level]
