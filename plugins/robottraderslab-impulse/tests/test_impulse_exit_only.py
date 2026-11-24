import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest
from impulse_helpers import write_signal_columns
from robottraderslab_core.ohlcv import OHLCVRow

from robottraderslab import Symbol, TimeFrame
from robottraderslab.backtester.simulator import SimulatedFuturesExchange
from robottraderslab.exchanges import (
    Execution,
    OHLCVProviderProtocol,
    OrderProtocol,
    OrderSide,
    OrderType,
    StopLoss,
    next_client_order_id,
)
from robottraderslab.live.livebot import LiveBot
from robottraderslab.strategies import OHLCVs, TradingMode, TradingSystem
from robottraderslab.strategies.futures import FuturesAccount
from robottraderslab_impulse import ImpulseStrategy
from robottraderslab_impulse.profile_config import ProfileConfig

BTC = Symbol.create("BTC/USDT:USDT")
TIMEFRAME: TimeFrame = "1m"
FIRST_OPEN = datetime(2024, 1, 1, tzinfo=UTC)
CANDLE_COUNT = 13
LAST_CANDLE = CANDLE_COUNT - 1
PRICE = 100.0
HELD_QUANTITY = 40.0
HOLDER = "gamma"
ENTERING = ("alpha", "beta")
PROFILE: dict[str, Any] = {
    "symbol": str(BTC),
    "timeframe": TIMEFRAME,
    "trix_length": 3,
    "signal_length": 3,
    "trend_length": 3,
    "total_balance_ratio": 0.25,
    "leverage": 5.0,
    "margin_mode": "isolated",
    "stop_loss_pct": 0.05,
}
PROFILES = [{**PROFILE, "tag": tag} for tag in (*ENTERING, HOLDER)]
SIGNALS_BY_TAG = {
    "alpha": {LAST_CANDLE: "long_entry"},
    "beta": {LAST_CANDLE: "long_entry"},
    HOLDER: {LAST_CANDLE: "long_exit"},
}


class _ScriptedImpulse(ImpulseStrategy):
    def generate_profile_signals(self, profile: ProfileConfig, ohlcvs: OHLCVs) -> None:
        write_signal_columns(
            ohlcvs,
            profile.symbol,
            profile.timeframe,
            profile.profile_id,
            SIGNALS_BY_TAG[profile.tag],
        )


def _candles() -> pd.DataFrame:
    prices = [PRICE] * CANDLE_COUNT
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": [1000.0] * CANDLE_COUNT,
        },
        index=pd.date_range(FIRST_OPEN, periods=CANDLE_COUNT, freq="1min"),
    )


class _CancelRecordingExchange(SimulatedFuturesExchange):
    cancelled_ids: list[str]
    stop_loss_id: str

    async def cancel_order_by_id(self, symbol: Symbol, order_id: str) -> None:
        self.cancelled_ids.append(order_id)
        await super().cancel_order_by_id(symbol, order_id)


def _play_a_candle(exchange: SimulatedFuturesExchange, opened_at: datetime) -> None:
    exchange.simulation_engine.simulate_on_current_ohlcvs(
        opened_at,
        {
            BTC: OHLCVRow(
                timestamp=opened_at,
                open=PRICE,
                high=PRICE,
                low=PRICE,
                close=PRICE,
                volume=1000.0,
            )
        },
    )


def _resting_stop_losses(exchange: SimulatedFuturesExchange) -> list[OrderProtocol]:
    resting = asyncio.run(exchange.get_open_orders([BTC]))
    return [order for order in resting if order.kind is OrderType.STOP_LOSS]


@pytest.fixture
def exchange() -> _CancelRecordingExchange:
    exchange = _CancelRecordingExchange.create_from_settings(
        initial_balance={"USDT": 10_000.0},
        maker_fee_rate=0.0,
        taker_fee_rate=0.0,
        fee_mode="cost",
    )
    exchange.cancelled_ids = []
    exchange.simulation_engine.enable_execution_recording()
    asyncio.run(
        exchange.place_market_order(
            BTC,
            OrderSide.BUY,
            HELD_QUANTITY,
            stop_loss=StopLoss(trigger_price=PRICE * 0.95),
            client_order_id=next_client_order_id(f"{TIMEFRAME}-{HOLDER}"),
        )
    )
    _play_a_candle(exchange, FIRST_OPEN)
    (stop_loss,) = _resting_stop_losses(exchange)
    exchange.stop_loss_id = stop_loss.order_id
    return exchange


@pytest.fixture
def run_cycle(
    exchange: _CancelRecordingExchange,
) -> Callable[[frozenset[Symbol]], None]:
    def run(exit_only_symbols: frozenset[Symbol]) -> None:
        strategy = _ScriptedImpulse(
            account=FuturesAccount(exchange),
            trading_system=TradingSystem(trading_mode=TradingMode.LIVE),
            config_dir=Path("bot-config-dir"),
            profiles=PROFILES,
        )
        provider = Mock(spec=OHLCVProviderProtocol)
        provider.fetch_ohlcv.return_value = _candles()
        asyncio.run(
            LiveBot(strategy, provider, exit_only_symbols=exit_only_symbols).run()
        )

    return run


def _fills(exchange: SimulatedFuturesExchange) -> list[Execution]:
    return exchange.simulation_engine.get_executions_since(FIRST_OPEN)


class TestTheLastProfileOfAStackedSymbolExits:
    @pytest.fixture
    def exit_only_cycle(self, run_cycle: Callable[[frozenset[Symbol]], None]) -> None:
        run_cycle(frozenset({BTC}))

    @pytest.mark.usefixtures("exit_only_cycle")
    def test_the_symbol_holds_no_position(self, exchange):
        assert exchange.simulation_engine.open_positions == {}

    @pytest.mark.usefixtures("exit_only_cycle")
    def test_the_exit_is_the_only_order_the_cycle_sent(self, exchange):
        assert [(fill.side, fill.quantity) for fill in _fills(exchange)] == [
            (OrderSide.BUY, HELD_QUANTITY),
            (OrderSide.SELL, HELD_QUANTITY),
        ]

    @pytest.mark.usefixtures("exit_only_cycle")
    def test_the_stop_loss_booked_for_cancel_in_the_final_slot_is_sent_and_none_rests(
        self, exchange
    ):
        assert exchange.cancelled_ids == [exchange.stop_loss_id]
        assert _resting_stop_losses(exchange) == []

    def test_the_entries_of_the_other_profiles_are_sent_on_a_symbol_running_as_usual(
        self, exchange, run_cycle
    ):
        run_cycle(frozenset())

        assert [fill.side for fill in _fills(exchange)].count(OrderSide.BUY) == 3
