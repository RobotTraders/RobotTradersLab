import asyncio
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from robottraderslab import Symbol, TimeFrame
from robottraderslab.backtester.backtester import Backtester
from robottraderslab.backtester.simulator import SimulatedFuturesExchange
from robottraderslab.futures import FuturesAccount
from robottraderslab.strategies import (
    AccountSnapshots,
    BookKeeper,
    OHLCVs,
    StrategyProtocol,
    StrategyRequirements,
)

ETHUSDT_PERP = Symbol.create("ETH/USDT:USDT")
START = pd.Timestamp("2024-01-01", tz="UTC")
HOURS = 720
HOURS_PER_FOUR_HOUR_CANDLE = 4
CUTOFF_HOURS_INTO_ITS_FOUR_HOUR_CANDLE = 2
CUTOFF = START + pd.Timedelta(
    hours=90 * HOURS_PER_FOUR_HOUR_CANDLE + CUTOFF_HOURS_INTO_ITS_FOUR_HOUR_CANDLE
)
RESTING_TAG = "resting"
type MomentValues = list[tuple[datetime, float]]
type RunRecord = tuple[pd.DataFrame, MomentValues, MomentValues]
FILL_COLUMNS = ["fill_type", "side", "price", "gross_quantity", "reason"]


def _price_walk(first_open: float, hours: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    closes = first_open * np.exp(np.cumsum(rng.normal(0.0, 0.01, hours)))
    opens = np.concatenate([[first_open], closes[:-1]])
    wicks = np.abs(rng.normal(0.0, 0.004, (2, hours)))
    return pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes) * (1 + wicks[0]),
            "low": np.minimum(opens, closes) * (1 - wicks[1]),
            "close": closes,
            "volume": 1.0,
        }
    )


def _hourly_candles(perturbed: bool) -> pd.DataFrame:
    candles = _price_walk(100.0, HOURS, seed=7)
    candles.index = pd.date_range(START, periods=HOURS, freq="1h", name="date")
    if perturbed:
        after_cutoff = candles.index >= CUTOFF
        continuation = _price_walk(
            candles.loc[~after_cutoff, "close"].iloc[-1], after_cutoff.sum(), seed=8
        )
        continuation.index = candles.index[after_cutoff]
        candles = pd.concat([candles[~after_cutoff], continuation])
    return candles


def _four_hour_candles(hourly: pd.DataFrame) -> pd.DataFrame:
    return hourly.resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )


class _FramesProvider:
    def __init__(self, frames: dict[TimeFrame, pd.DataFrame]):
        self._frames = frames

    def set_dates(self, start_date: str, end_date: str) -> None: ...

    def fetch_ohlcv(self, symbol: Symbol, timeframe: TimeFrame) -> pd.DataFrame:
        return self._frames[timeframe]

    def get_all_cached_ohlcv_for_symbol(self, symbol: Symbol) -> pd.DataFrame:
        return self._frames["1h"]


class _EveryOrderKindStrategy(StrategyProtocol):
    market_type = "futures"

    def __init__(self, account: FuturesAccount):
        self._account = account
        self._candles_read = 0
        self.stop_moves: MomentValues = []

    async def setup(self, requirements: StrategyRequirements) -> None:
        requirements.ohlcv.add(ETHUSDT_PERP, "1h")
        requirements.ohlcv.add(ETHUSDT_PERP, "4h")
        requirements.account.add(self._account, symbols=[ETHUSDT_PERP], positions=True)

    def generate_trading_signals(self, ohlcvs: OHLCVs) -> None:
        for timeframe in ("1h", "4h"):
            closes = ohlcvs.column(ETHUSDT_PERP, timeframe, "close")
            rose = np.concatenate([[np.nan], closes[1:] > closes[:-1]])
            ohlcvs.add_column(ETHUSDT_PERP, timeframe, "rose", rose)

    def book_trading_actions(
        self,
        ohlcvs: OHLCVs,
        account_snapshots: AccountSnapshots,
        timestamp: datetime,
        bookkeeper: BookKeeper,
        triggered_timeframes: set[TimeFrame],
    ) -> None:
        self._candles_read += 1
        if self._candles_read <= HOURS_PER_FOUR_HOUR_CANDLE:
            return
        close = ohlcvs.current(ETHUSDT_PERP, "1h", "close")
        higher_close = ohlcvs.current(ETHUSDT_PERP, "4h", "close")
        rose = ohlcvs.signal(ETHUSDT_PERP, "1h", "rose")
        higher_rose = ohlcvs.signal(ETHUSDT_PERP, "4h", "rose")
        position = account_snapshots.of(self._account).position(ETHUSDT_PERP)
        booked = f" @ {timestamp.isoformat()}"
        bookkeeper.add(self._account.cancel_orders(ETHUSDT_PERP, tag=RESTING_TAG))
        if position is None:
            self._book_entry(close, higher_close, rose, higher_rose, booked, bookkeeper)
        else:
            self._manage_position(close, rose, timestamp, booked, bookkeeper)

    def _manage_position(
        self,
        close: float,
        rose: bool,
        timestamp: datetime,
        booked: str,
        bookkeeper: BookKeeper,
    ) -> None:
        if rose:
            level = close * 0.985
            self.stop_moves.append((timestamp, level))
            bookkeeper.add(self._account.move_stop_loss(ETHUSDT_PERP, level))
        else:
            bookkeeper.add(
                self._account.close_position(ETHUSDT_PERP)
                .reason("market exit" + booked)
                .build()
            )

    def _book_entry(
        self,
        close: float,
        higher_close: float,
        rose: bool,
        higher_rose: bool,
        booked: str,
        bookkeeper: BookKeeper,
    ) -> None:
        entry = self._account.long_entry(ETHUSDT_PERP, 1.0)
        if rose and higher_rose:
            entry = (
                entry.stop_loss(
                    min(close, higher_close) * 0.98, reason="stop-loss" + booked
                )
                .take_profit(
                    max(close, higher_close) * 1.03, reason="take-profit" + booked
                )
                .reason("market entry" + booked)
            )
        elif higher_rose:
            level = max(close, higher_close) * 1.002
            entry = (
                entry.trigger(level)
                .stop_loss(level * 0.98, reason="stop-loss" + booked)
                .reason("trigger entry" + booked)
                .tag(RESTING_TAG)
            )
        else:
            price = min(close, higher_close) * 0.998
            entry = (
                entry.limit(price)
                .stop_loss(price * 0.98, reason="stop-loss" + booked)
                .reason("limit entry" + booked)
                .tag(RESTING_TAG)
            )
        bookkeeper.add(entry.build())


def _run(perturbed: bool) -> RunRecord:
    hourly = _hourly_candles(perturbed)
    four_hour = _four_hour_candles(hourly)
    hourly = hourly.loc[: four_hour.index[-1]]
    exchange = SimulatedFuturesExchange.create_from_settings(
        initial_balance={"USDT": 100_000.0}, maker_fee_rate=0.0, taker_fee_rate=0.0
    )
    strategy = _EveryOrderKindStrategy(FuturesAccount(exchange))
    backtester = Backtester(
        strategy,
        exchange.simulation_engine,
        _FramesProvider({"1h": hourly, "4h": four_hour}),
        exchange.fill_recorder,
        START.date().isoformat(),
        (START + pd.Timedelta(hours=HOURS)).date().isoformat(),
    )
    outputs = asyncio.run(backtester.run())
    equity = [
        (snapshot.timestamp, snapshot.get_equity("USDT"))
        for snapshot in outputs.daily_equity_snapshots + outputs.trade_equity_snapshots
    ]
    return outputs.fills[FILL_COLUMNS], equity, strategy.stop_moves


@pytest.fixture(scope="module")
def generated_run() -> RunRecord:
    return _run(perturbed=False)


@pytest.fixture(scope="module")
def perturbed_run() -> RunRecord:
    return _run(perturbed=True)


def test_candles_after_a_moment_change_nothing_up_to_it(generated_run, perturbed_run):
    fills, equity, _ = generated_run
    perturbed_fills, perturbed_equity, _ = perturbed_run

    pd.testing.assert_frame_equal(
        fills[fills.index <= CUTOFF], perturbed_fills[perturbed_fills.index <= CUTOFF]
    )
    assert [point for point in equity if point[0] <= CUTOFF] == [
        point for point in perturbed_equity if point[0] <= CUTOFF
    ]
    assert not fills[fills.index > CUTOFF].equals(
        perturbed_fills[perturbed_fills.index > CUTOFF]
    )


def test_an_order_fills_on_prices_after_it_was_booked(generated_run):
    fills, _, _ = generated_run
    booking = fills["reason"].str.extract(r"^(?P<kind>.+) @ (?P<moment>.+)$")
    filled_at = fills.index.to_numpy()
    booked_at = pd.to_datetime(booking["moment"]).to_numpy()
    at_market = booking["kind"].str.startswith("market").to_numpy()

    early = (filled_at < booked_at) | (~at_market & (filled_at == booked_at))

    assert fills[early].empty


def test_a_moved_stop_fills_on_prices_after_its_move(generated_run):
    fills, _, stop_moves = generated_run
    moved_at = {level: moment for moment, level in stop_moves}
    stop_fills = fills[fills["reason"].str.startswith("stop-loss")]
    moved_stop_fills = [
        (filled_at, moved_at[price])
        for filled_at, price in zip(stop_fills.index, stop_fills["price"])
        if price in moved_at
    ]

    assert moved_stop_fills
    assert [fill for fill in moved_stop_fills if fill[0] <= fill[1]] == []
