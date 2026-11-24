import asyncio
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest
from robottraderslab_core.ohlcv import OHLCVRow

from robottraderslab import Symbol, TimeFrame
from robottraderslab.backtester.simulator import SimulatedFuturesExchange
from robottraderslab.exchanges import (
    OHLCVProviderProtocol,
    OrderModifyRequest,
    OrderProtocol,
    OrderSide,
    OrderType,
    PlacedOrder,
    StopLoss,
    TakeProfit,
    next_client_order_id,
)
from robottraderslab.live.livebot import LiveBot
from robottraderslab.strategies import TradingMode, TradingSystem, profile_tag
from robottraderslab.strategies.futures import FuturesAccount
from robottraderslab_envelope import EnvelopeStrategy

BTC = Symbol.create("BTC/USDT:USDT")
TIMEFRAME: TimeFrame = "1m"
FIRST_OPEN = datetime(2024, 1, 1, tzinfo=UTC)
CANDLE_COUNT = 12
PRICE = 100.0
HELD_QUANTITY = 10.0
RUNG_QUANTITY = 5.0
SECOND_BAND_PRICE = 96.0
STALE_STOP_LOSS = 80.0
STALE_TAKE_PROFIT = 150.0
STOP_LOSS_PCT = 0.05
PROFILE: dict[str, Any] = {
    "symbol": str(BTC),
    "timeframe": TIMEFRAME,
    "average_type": "SMA",
    "average_period": 3,
    "long_envelopes": [0.02, 0.04],
    "stop_loss_pct": STOP_LOSS_PCT,
    "total_balance_ratio": 0.3,
    "leverage": 5.0,
    "margin_mode": "isolated",
}


class _RecordingExchange(SimulatedFuturesExchange):
    cancelled_ids: list[str]
    placed_prices: list[float | None]
    modified_ids: list[str]
    second_band_rung_id: str

    async def cancel_order_by_id(self, symbol: Symbol, order_id: str) -> None:
        self.cancelled_ids.append(order_id)
        await super().cancel_order_by_id(symbol, order_id)

    async def modify_orders(
        self, requests: Sequence[OrderModifyRequest]
    ) -> list[PlacedOrder | None]:
        self.modified_ids.extend(request.order_id for request in requests)
        return await super().modify_orders(requests)

    async def place_limit_order(
        self, symbol: Symbol, side: OrderSide, quantity: float, price: float, **kwargs
    ) -> PlacedOrder:
        self.placed_prices.append(price)
        return await super().place_limit_order(symbol, side, quantity, price, **kwargs)

    async def place_market_order(
        self, symbol: Symbol, side: OrderSide, quantity: float, **kwargs
    ) -> PlacedOrder:
        self.placed_prices.append(None)
        return await super().place_market_order(symbol, side, quantity, **kwargs)


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


def _resting(exchange: SimulatedFuturesExchange) -> list[OrderProtocol]:
    return asyncio.run(exchange.get_open_orders([BTC]))


@pytest.fixture
def exchange() -> _RecordingExchange:
    exchange = _RecordingExchange.create_from_settings(
        initial_balance={"USDT": 10_000.0},
        maker_fee_rate=0.0,
        taker_fee_rate=0.0,
        fee_mode="cost",
    )
    exchange.cancelled_ids = []
    exchange.placed_prices = []
    exchange.modified_ids = []
    exchange.simulation_engine.enable_execution_recording()
    asyncio.run(
        exchange.place_market_order(
            BTC,
            OrderSide.BUY,
            HELD_QUANTITY,
            stop_loss=StopLoss(trigger_price=STALE_STOP_LOSS),
            take_profit=TakeProfit(trigger_price=STALE_TAKE_PROFIT),
        )
    )
    asyncio.run(
        exchange.place_limit_order(
            BTC,
            OrderSide.BUY,
            RUNG_QUANTITY,
            SECOND_BAND_PRICE,
            client_order_id=next_client_order_id(profile_tag(TIMEFRAME, "l2")),
        )
    )
    exchange.simulation_engine.simulate_on_current_ohlcvs(
        FIRST_OPEN,
        {
            BTC: OHLCVRow(
                timestamp=FIRST_OPEN,
                open=PRICE,
                high=PRICE,
                low=PRICE,
                close=PRICE,
                volume=1000.0,
            )
        },
    )
    (rung,) = (order for order in _resting(exchange) if order.kind is OrderType.LIMIT)
    exchange.second_band_rung_id = rung.order_id
    exchange.placed_prices.clear()
    return exchange


@pytest.fixture
def run_cycle(exchange: _RecordingExchange) -> Callable[[frozenset[Symbol]], None]:
    def run(exit_only_symbols: frozenset[Symbol]) -> None:
        strategy = EnvelopeStrategy(
            account=FuturesAccount(exchange=exchange),
            trading_system=TradingSystem(trading_mode=TradingMode.BACKTEST),
            config_dir=Path("bot-config-dir"),
            profiles=[PROFILE],
            entry_order="limit",
        )
        provider = Mock(spec=OHLCVProviderProtocol)
        provider.fetch_ohlcv.return_value = _candles()
        asyncio.run(
            LiveBot(strategy, provider, exit_only_symbols=exit_only_symbols).run()
        )

    return run


class TestAnUnfilledRungOnThePositionsSide:
    @pytest.fixture
    def exit_only_cycle(self, run_cycle: Callable[[frozenset[Symbol]], None]) -> None:
        run_cycle(frozenset({BTC}))

    @pytest.mark.usefixtures("exit_only_cycle")
    def test_the_rung_is_cancelled_and_none_rests(self, exchange):
        assert exchange.cancelled_ids == [exchange.second_band_rung_id]
        assert [
            order for order in _resting(exchange) if order.kind is OrderType.LIMIT
        ] == []

    @pytest.mark.usefixtures("exit_only_cycle")
    def test_no_rung_is_placed(self, exchange):
        assert exchange.placed_prices == []

    @pytest.mark.usefixtures("exit_only_cycle")
    def test_no_rung_is_modified(self, exchange):
        assert exchange.modified_ids == []

    @pytest.mark.usefixtures("exit_only_cycle")
    def test_the_stop_loss_and_the_take_profit_stay_and_are_moved(self, exchange):
        protections = sorted(
            (order.kind.value, order.trigger_price) for order in _resting(exchange)
        )

        assert protections == [
            (OrderType.STOP_LOSS.value, PRICE * (1 - STOP_LOSS_PCT)),
            (OrderType.TAKE_PROFIT.value, PRICE),
        ]

    def test_the_rung_is_modified_on_a_symbol_running_as_usual(
        self, exchange, run_cycle
    ):
        run_cycle(frozenset())

        assert exchange.modified_ids == [exchange.second_band_rung_id]
