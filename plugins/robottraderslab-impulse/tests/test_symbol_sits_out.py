import asyncio
import logging
from datetime import UTC, datetime
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest
from robottraderslab_core.engine import name_process

from robottraderslab import Symbol, TimeFrame
from robottraderslab.exchanges import (
    Execution,
    OHLCVProviderProtocol,
    OrderProtocol,
    OrderSide,
    OrderType,
    PlacedOrder,
    PositionSide,
    PositionSnapshot,
    VenueFill,
    next_client_order_id,
)
from robottraderslab.live.livebot import LiveBot
from robottraderslab.strategies import OHLCVs, TradingMode, TradingSystem
from robottraderslab.strategies.futures import FuturesAccount
from robottraderslab_impulse import ImpulseStrategy

BTC = Symbol.create("BTC/USDT:USDT")
ETH = Symbol.create("ETH/USDT:USDT")
TIMEFRAME: TimeFrame = "1m"
FIRST_OPEN = datetime(2024, 1, 1, tzinfo=UTC)
CLOSES = [100.0] * 12 + [101.0]
HELD_QUANTITY = 40.0
PROFILE: dict[str, Any] = {
    "timeframe": TIMEFRAME,
    "trix_length": 3,
    "signal_length": 3,
    "trend_length": 3,
    "total_balance_ratio": 0.25,
    "leverage": 5.0,
    "margin_mode": "isolated",
}
BTC_PROFILE = {**PROFILE, "symbol": str(BTC)}
BTC_STACKED_PROFILE = {**BTC_PROFILE, "tag": "stacked"}
BTC_REFUSED_PROFILE = {**BTC_PROFILE, "tag": "refused", "stop_los_pct": 0.05}
ETH_PROFILE = {**PROFILE, "symbol": str(ETH)}


class _RecordingImpulse(ImpulseStrategy):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.signalled: list[str] = []
        self.booked: list[str] = []

    def generate_profile_signals(self, profile: Any, ohlcvs: OHLCVs) -> None:
        self.signalled.append(str(profile.symbol))
        super().generate_profile_signals(profile, ohlcvs)

    def book_profile_actions(
        self,
        profile: Any,
        ohlcvs: OHLCVs,
        account_snapshots: Any,
        timestamp: datetime,
        bookkeeper: Any,
    ) -> None:
        self.booked.append(str(profile.symbol))
        super().book_profile_actions(
            profile, ohlcvs, account_snapshots, timestamp, bookkeeper
        )


def _candles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": CLOSES,
            "high": CLOSES,
            "low": CLOSES,
            "close": CLOSES,
            "volume": [1000.0] * len(CLOSES),
        },
        index=pd.date_range(FIRST_OPEN, periods=len(CLOSES), freq="1min"),
    )


def _hold_btc_with_a_resting_stop_loss(exchange: Mock) -> Mock:
    own_id = next_client_order_id("1m-stacked")
    exchange.get_open_positions.return_value = {
        BTC: PositionSnapshot(
            symbol=BTC,
            side=PositionSide.LONG,
            quantity=HELD_QUANTITY,
            average_entry_price=100.0,
            entry_time=FIRST_OPEN,
            leverage=5.0,
            liquidation_price=0.0,
            stop_loss_price=None,
            take_profit_price=None,
        )
    }
    stop_loss = Mock(spec=OrderProtocol)
    stop_loss.order_id = "stop-loss-1"
    stop_loss.symbol = BTC
    stop_loss.kind = OrderType.STOP_LOSS
    stop_loss.trigger_price = 95.0
    stop_loss.client_order_id = own_id
    exchange.get_open_orders.return_value = [stop_loss]
    exchange.get_executions_since.return_value = [
        Execution(
            execution_id="fill-1",
            order_id="order-1",
            symbol=BTC,
            side=OrderSide.BUY,
            price=100.0,
            quantity=HELD_QUANTITY,
            timestamp=FIRST_OPEN,
            client_order_id=own_id,
        )
    ]
    return stop_loss


async def _bought(**_kwargs: object) -> VenueFill:
    return VenueFill(order_id="order-1", symbol=ETH, side=OrderSide.BUY, quantity=1.0)


@pytest.fixture
def exchange(mock_exchange: Mock) -> Mock:
    name_process("impulse-kraken", datetime(2026, 10, 22, tzinfo=UTC))
    mock_exchange.place_market_order.return_value = PlacedOrder(
        order_id="order-1", get_fill=_bought
    )
    return mock_exchange


@pytest.fixture
def stop_loss(exchange: Mock) -> Mock:
    return _hold_btc_with_a_resting_stop_loss(exchange)


@pytest.fixture
def strategy(exchange: Mock) -> _RecordingImpulse:
    return _RecordingImpulse(
        account=FuturesAccount(exchange=exchange, name="test"),
        trading_system=TradingSystem(trading_mode=TradingMode.LIVE),
        config_dir=None,
        profiles=[BTC_PROFILE, BTC_REFUSED_PROFILE, ETH_PROFILE],
    )


@pytest.fixture
def ran_cycle(
    strategy: _RecordingImpulse, stop_loss: Mock, caplog: pytest.LogCaptureFixture
) -> None:
    provider = Mock(spec=OHLCVProviderProtocol)
    provider.fetch_ohlcv.return_value = _candles()
    with caplog.at_level(logging.ERROR):
        asyncio.run(LiveBot(strategy, provider).run())


def _placed_symbols(exchange: Mock) -> set[Symbol]:
    return {
        call.kwargs["symbol"] for call in exchange.place_market_order.await_args_list
    }


@pytest.mark.usefixtures("ran_cycle")
class TestRefusedProfileOnASharedSymbol:
    def test_no_profile_of_the_symbol_runs_a_slot(self, strategy):
        assert strategy.signalled == [str(ETH)]
        assert strategy.booked == [str(ETH)]

    def test_the_profile_on_another_symbol_trades(self, exchange):
        assert _placed_symbols(exchange) == {ETH}

    def test_the_refused_profile_and_the_profile_left_out_with_it_are_named(
        self, caplog
    ):
        assert [record.getMessage() for record in caplog.get_records("setup")] == [
            "_RecordingImpulse: profile 2 (BTC/USDT:USDT 1m): stop_los_pct: "
            "unknown key; this profile sits out the run",
            "_RecordingImpulse: profile 1 (BTC/USDT:USDT 1m): left out of the run "
            "with profile 2, which was refused on the same symbol",
        ]

    def test_the_stop_loss_resting_on_the_symbol_is_left_resting(self, exchange):
        exchange.cancel_order_by_id.assert_not_awaited()
