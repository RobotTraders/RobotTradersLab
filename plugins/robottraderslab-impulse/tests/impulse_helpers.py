import asyncio
import io
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pandas as pd
from robottraderslab_core.ohlcv import OHLCVRow

from robottraderslab import Symbol, TimeFrame
from robottraderslab.backtester.backtester import Backtester
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
from robottraderslab.ohlcv_provider import CSVOHLCVProvider
from robottraderslab.strategies import OHLCVs, TradingMode, TradingSystem
from robottraderslab.strategies.futures import FuturesAccount
from robottraderslab_impulse import ImpulseStrategy
from robottraderslab_impulse.profile_config import ProfileConfig

SIGNAL_NAMES = ("long_entry", "long_exit", "short_entry", "short_exit")
BTC = Symbol.create("BTC/USDT:USDT")
TIMEFRAME: TimeFrame = "1m"
BACKTEST_TIMEFRAME: TimeFrame = "1d"
BACKTEST_START_DATE = "2024-01-01"
FIRST_OPEN = datetime(2024, 1, 1, tzinfo=UTC)
CANDLE_COUNT = 13
LAST_CANDLE = CANDLE_COUNT - 1
PRICE = 100.0
HELD_QUANTITY = 40.0
STOP_LOSS_TRIGGER_PRICE = PRICE * 0.95
HOLDER_TAG = "gamma"
BTC_PROFILE: dict[str, Any] = {
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


def write_signal_columns(
    ohlcvs: OHLCVs,
    symbol: Symbol,
    timeframe: TimeFrame,
    profile_id: str,
    scripted: Mapping[int, str],
) -> None:
    """Write the four signal columns of a profile over every candle of its series.

    `scripted` maps a candle's index to the signal it raises, so a column holds
    1.0 on the candles scripted with its name and 0.0 on every other.
    """
    candles = len(ohlcvs.column(symbol, timeframe, "close"))
    for name in SIGNAL_NAMES:
        ohlcvs.add_column(
            symbol,
            timeframe,
            f"{profile_id}_{name}",
            [float(scripted.get(candle) == name) for candle in range(candles)],
        )


def scripted_impulse(
    signals_by_tag: Mapping[str, Mapping[int, str]],
) -> type[ImpulseStrategy]:
    """Return an Impulse whose profiles raise the signals scripted for their tag.

    Every profile's tag must be a key of `signals_by_tag`.
    """

    class ScriptedImpulse(ImpulseStrategy):
        def generate_profile_signals(
            self, profile: ProfileConfig, ohlcvs: OHLCVs
        ) -> None:
            write_signal_columns(
                ohlcvs,
                profile.symbol,
                profile.timeframe,
                profile.profile_id,
                signals_by_tag[profile.tag],
            )

    return ScriptedImpulse


def flat_candles() -> pd.DataFrame:
    """Return the candles of a cycle, every price the same."""
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


def new_exchange[E: SimulatedFuturesExchange](exchange_class: type[E]) -> E:
    """Return a fresh exchange of the class that holds USDT and charges no fee."""
    return exchange_class.create_from_settings(
        initial_balance={"USDT": 10_000.0},
        maker_fee_rate=0.0,
        taker_fee_rate=0.0,
        fee_mode="cost",
    )


def play_a_candle(
    simulated_exchange: SimulatedFuturesExchange, opened_at: datetime
) -> None:
    """Let the simulator play one flat candle opening at `opened_at`."""
    simulated_exchange.simulation_engine.simulate_on_current_ohlcvs(
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


async def hold_a_long_with_a_stop[E: SimulatedFuturesExchange](
    exchange_class: type[E], profile_tag: str
) -> E:
    """Return a fresh exchange holding a long that `profile_tag` opened and its stop-loss.

    The long is of a fixed quantity, and the stop-loss rests beside it.
    """
    simulated_exchange = new_exchange(exchange_class)
    simulated_exchange.simulation_engine.enable_execution_recording()
    await simulated_exchange.place_market_order(
        BTC,
        OrderSide.BUY,
        HELD_QUANTITY,
        stop_loss=StopLoss(trigger_price=STOP_LOSS_TRIGGER_PRICE),
        client_order_id=next_client_order_id(f"{TIMEFRAME}-{profile_tag}"),
    )
    play_a_candle(simulated_exchange, FIRST_OPEN)
    return simulated_exchange


async def resting_stop_losses(
    simulated_exchange: SimulatedFuturesExchange,
) -> list[OrderProtocol]:
    """Return the stop-loss orders resting on the BTC/USDT perpetual market."""
    resting = await simulated_exchange.get_open_orders([BTC])
    return [
        open_order for open_order in resting if open_order.kind is OrderType.STOP_LOSS
    ]


def executions_of(simulated_exchange: SimulatedFuturesExchange) -> list[Execution]:
    """Return the executions the exchange recorded from the first candle on."""
    return simulated_exchange.simulation_engine.get_executions_since(FIRST_OPEN)


def executed_sides_and_quantities(
    simulated_exchange: SimulatedFuturesExchange,
) -> list[tuple[OrderSide, float]]:
    """Return the side and the quantity of each execution `executions_of` returns."""
    return [
        (execution.side, execution.quantity)
        for execution in executions_of(simulated_exchange)
    ]


async def run_bot(
    impulse_strategy: ImpulseStrategy,
    candles: pd.DataFrame,
    **live_bot_options: Any,
) -> None:
    """Run one live cycle of `impulse_strategy`, every candle read answering `candles`."""
    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.return_value = candles
    await LiveBot(impulse_strategy, ohlcv_provider, **live_bot_options).run()


async def run_one_cycle(
    simulated_exchange: SimulatedFuturesExchange,
    strategy_class: type[ImpulseStrategy],
    profile_dicts: list[dict[str, Any]],
    **live_bot_options: Any,
) -> None:
    """Run one live cycle as its own process would.

    The strategy and the bot are built afresh over the exchange.
    """
    impulse_strategy = strategy_class(
        account=FuturesAccount(simulated_exchange),
        trading_system=TradingSystem(trading_mode=TradingMode.LIVE),
        config_dir=Path("bot-config-dir"),
        profiles=profile_dicts,
    )
    await run_bot(impulse_strategy, flat_candles(), **live_bot_options)


def run_scripted_backtest(
    simulated_exchange: SimulatedFuturesExchange,
    signals_by_tag: Mapping[str, Mapping[int, str]],
    profile_dicts: list[dict[str, Any]],
    daily_candles: io.StringIO,
    end_date: str,
) -> None:
    """Backtest the scripted Impulse over the daily candles of `daily_candles`.

    The backtest runs from the first day of the candles to `end_date`, on the
    simulator of `simulated_exchange`.
    """
    impulse_strategy = scripted_impulse(signals_by_tag)(
        account=FuturesAccount(simulated_exchange),
        trading_system=TradingSystem(trading_mode=TradingMode.BACKTEST),
        config_dir=Path("bot-config-dir"),
        profiles=profile_dicts,
    )
    backtester = Backtester(
        impulse_strategy,
        simulated_exchange.simulation_engine,
        CSVOHLCVProvider(
            file=daily_candles, symbol=str(BTC), timeframe=BACKTEST_TIMEFRAME
        ),
        simulated_exchange.fill_recorder,
        BACKTEST_START_DATE,
        end_date,
    )
    asyncio.run(backtester.run())
