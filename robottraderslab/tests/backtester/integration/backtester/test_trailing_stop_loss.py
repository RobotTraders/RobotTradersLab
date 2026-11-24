import asyncio
import io
from collections.abc import Callable
from datetime import datetime

import pandas as pd
import pytest

from robottraderslab import Symbol, TimeFrame
from robottraderslab.backtester.backtester import Backtester
from robottraderslab.backtester.simulator import SimulatedFuturesExchange
from robottraderslab.exchanges import PositionSide
from robottraderslab.futures import FuturesAccount
from robottraderslab.ohlcv_provider import CSVOHLCVProvider
from robottraderslab.strategies import (
    AccountSnapshots,
    BookKeeper,
    OHLCVs,
    StrategyProtocol,
    StrategyRequirements,
)

BTCUSDT_PERP = Symbol.create("BTC/USDT:USDT")
ENTRY_CLOSE = 114_310.8
HALF_A_PERCENT = 0.005
FIFTEEN_HUNDREDTHS_OF_A_PERCENT = 0.0015
FIRST_CANDLE_AFTER_THE_ENTRY = 1
CANDLE_BEFORE_THE_RISING_ONE = 5
LONG_LEVEL_FROM_THE_BOOKING_CLOSE = 113_890.287
LONG_LEVEL_FROM_THE_FIRST_NEW_HIGH = 113_910.585
LONG_LEVEL_FROM_THE_HIGHEST_HIGH = 114_027.0
LONG_LEVELS_MET = (
    [LONG_LEVEL_FROM_THE_BOOKING_CLOSE] * 4
    + [LONG_LEVEL_FROM_THE_FIRST_NEW_HIGH]
    + [LONG_LEVEL_FROM_THE_HIGHEST_HIGH] * 8
)
SHORT_LEVEL_FROM_THE_FIRST_LOW = 114_653.817
SHORT_LEVELS_MET = [SHORT_LEVEL_FROM_THE_FIRST_LOW] * 13
SHORT_LEVEL_FROM_THE_LOWEST_LOW = 114_157.95
LEVEL_THE_RISING_CANDLE_GIVES = 114_311.2755
SELL_OFF_CANDLE_CLOSE = pd.Timestamp("2025-09-22T06:00:00+00:00")
RISING_THEN_FALLING_CANDLE_CLOSE = pd.Timestamp("2025-09-22T03:45:00+00:00")

BITGET_BTC_SELL_OFF = """\
date,open,high,low,close,volume
2025-09-22T02:00:00+00:00,114215.8,114520.7,114210.3,114310.8,996.5
2025-09-22T02:15:00+00:00,114310.8,114555.0,114300.0,114462.6,705.3
2025-09-22T02:30:00+00:00,114462.6,114462.6,114083.4,114245.7,1123.8
2025-09-22T02:45:00+00:00,114245.7,114370.3,114137.6,114137.6,342.1
2025-09-22T03:00:00+00:00,114137.6,114329.3,114137.6,114256.3,315.5
2025-09-22T03:15:00+00:00,114256.3,114411.7,114256.2,114411.7,322.4
2025-09-22T03:30:00+00:00,114411.7,114483.0,114276.6,114477.6,238.3
2025-09-22T03:45:00+00:00,114477.6,114600.0,114477.6,114599.8,606.0
2025-09-22T04:00:00+00:00,114599.8,114599.9,114386.0,114386.0,215.3
2025-09-22T04:15:00+00:00,114386.0,114496.1,114353.2,114456.1,159.1
2025-09-22T04:30:00+00:00,114456.1,114522.8,114360.0,114361.7,121.0
2025-09-22T04:45:00+00:00,114361.7,114425.5,114343.8,114392.3,148.7
2025-09-22T05:00:00+00:00,114392.3,114392.3,114304.7,114354.2,252.9
2025-09-22T05:15:00+00:00,114354.2,114498.6,114300.0,114480.1,327.9
2025-09-22T05:30:00+00:00,114480.1,114480.2,114271.5,114271.5,238.7
2025-09-22T05:45:00+00:00,114271.5,114340.3,113590.0,113628.2,2434.7
"""


class _TrailingStrategy(StrategyProtocol):
    market_type = "futures"

    def __init__(
        self,
        account: FuturesAccount,
        ohlcv_provider: CSVOHLCVProvider,
        side: PositionSide,
        trail_ratio: float,
        first_trail_candle: int,
    ) -> None:
        self.ohlcv_provider = ohlcv_provider
        self._account = account
        self._side = side
        self._trail_ratio = trail_ratio
        self._first_trail_candle = first_trail_candle
        self._candle = 0
        self._levels_met: list[float] = []
        self._trailing_stop_ids: set[str] = set()

    @property
    def levels_met(self) -> list[float]:
        return self._levels_met

    @property
    def trailing_stop_ids(self) -> set[str]:
        return self._trailing_stop_ids

    def book_trading_actions(
        self,
        ohlcvs: OHLCVs,
        account_snapshots: AccountSnapshots,
        timestamp: datetime,
        bookkeeper: BookKeeper,
        triggered_timeframes: set[TimeFrame],
    ) -> None:
        if self._candle == 0:
            entry = (
                self._account.long_entry(BTCUSDT_PERP, 1.0)
                if self._side is PositionSide.LONG
                else self._account.short_entry(BTCUSDT_PERP, 1.0)
            )
            bookkeeper.add(entry)
        elif self._candle >= self._first_trail_candle:
            self._trail(account_snapshots, bookkeeper)
        self._candle += 1

    def generate_trading_signals(self, ohlcvs: OHLCVs) -> None: ...

    async def setup(self, requirements: StrategyRequirements) -> None:
        requirements.ohlcv.add(BTCUSDT_PERP, "15m")
        requirements.account.add(
            self._account, symbols=[BTCUSDT_PERP], positions=True, open_orders=True
        )

    def _trail(
        self, account_snapshots: AccountSnapshots, bookkeeper: BookKeeper
    ) -> None:
        snapshot = account_snapshots.of(self._account)
        if snapshot.position(BTCUSDT_PERP) is None:
            return
        for order in snapshot.trailing_stop_orders(BTCUSDT_PERP):
            self._levels_met.append(order.trigger_price)
            self._trailing_stop_ids.add(order.order_id)
        bookkeeper.add(
            self._account.trail_stop_loss(
                BTCUSDT_PERP, self._trail_ratio, account_snapshot=snapshot
            )
        )


@pytest.fixture
def make_trailing_backtest() -> Callable[..., tuple[_TrailingStrategy, pd.DataFrame]]:
    def make(
        side: PositionSide, trail_ratio: float, first_trail_candle: int
    ) -> tuple[_TrailingStrategy, pd.DataFrame]:
        exchange = SimulatedFuturesExchange.create_from_settings(
            initial_balance={"USDT": 1_000_000.0},
            maker_fee_rate=0.0,
            taker_fee_rate=0.0,
        )
        ohlcv_provider = CSVOHLCVProvider(
            file=io.StringIO(BITGET_BTC_SELL_OFF),
            symbol=str(BTCUSDT_PERP),
            timeframe="15m",
        )
        strategy = _TrailingStrategy(
            FuturesAccount(exchange),
            ohlcv_provider,
            side,
            trail_ratio,
            first_trail_candle,
        )
        backtester = Backtester(
            strategy,
            exchange.simulation_engine,
            ohlcv_provider,
            exchange.fill_recorder,
            "2025-09-22",
            "2025-09-23",
        )
        fills = asyncio.run(backtester.run()).fills
        return strategy, fills

    return make


def test_the_stop_follows_each_high_and_fires_when_a_low_reaches_it(
    make_trailing_backtest,
):
    strategy, fills = make_trailing_backtest(
        PositionSide.LONG, HALF_A_PERCENT, FIRST_CANDLE_AFTER_THE_ENTRY
    )

    assert strategy.levels_met == pytest.approx(LONG_LEVELS_MET)
    assert list(fills["fill_type"]) == ["enter_long", "exit_long"]
    assert list(fills["reason"])[1] == "trailing-stop"
    assert list(fills.index)[1] == SELL_OFF_CANDLE_CLOSE
    assert list(fills["price"]) == pytest.approx(
        [ENTRY_CLOSE, LONG_LEVEL_FROM_THE_HIGHEST_HIGH]
    )


def test_the_booking_repeated_on_every_candle_rests_one_trailing_stop(
    make_trailing_backtest,
):
    strategy, _ = make_trailing_backtest(
        PositionSide.LONG, HALF_A_PERCENT, FIRST_CANDLE_AFTER_THE_ENTRY
    )

    assert len(strategy.levels_met) > 1
    assert len(strategy.trailing_stop_ids) == 1


def test_a_candle_rising_then_falling_fires_within_that_candle(make_trailing_backtest):
    _, fills = make_trailing_backtest(
        PositionSide.LONG,
        FIFTEEN_HUNDREDTHS_OF_A_PERCENT,
        CANDLE_BEFORE_THE_RISING_ONE,
    )

    assert list(fills["fill_type"]) == ["enter_long", "exit_long"]
    assert list(fills.index)[1] == RISING_THEN_FALLING_CANDLE_CLOSE
    assert list(fills["price"])[1] == pytest.approx(LEVEL_THE_RISING_CANDLE_GIVES)


def test_a_short_follows_each_low_and_fires_when_a_high_reaches_it(
    make_trailing_backtest,
):
    strategy, fills = make_trailing_backtest(
        PositionSide.SHORT, HALF_A_PERCENT, FIRST_CANDLE_AFTER_THE_ENTRY
    )

    assert strategy.levels_met == pytest.approx(SHORT_LEVELS_MET)
    assert len(strategy.trailing_stop_ids) == 1
    assert list(fills["fill_type"]) == ["enter_short", "exit_short"]
    assert list(fills.index)[1] == SELL_OFF_CANDLE_CLOSE
    assert list(fills["price"]) == pytest.approx(
        [ENTRY_CLOSE, SHORT_LEVEL_FROM_THE_LOWEST_LOW]
    )
