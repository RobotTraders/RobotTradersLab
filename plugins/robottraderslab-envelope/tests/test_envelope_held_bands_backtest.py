import asyncio
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import StringIO
from typing import Any

import pytest
from simulated_backtest import create_backtester, create_envelope, create_exchange

from robottraderslab import Symbol
from robottraderslab.backtester.simulator import SimulatedFuturesExchange
from robottraderslab.ohlcv_provider import CSVOHLCVProvider
from robottraderslab.strategies import (
    Execution,
    OrderSide,
    OrderType,
    PositionSide,
    tag_of,
)
from robottraderslab_envelope import EnvelopeStrategy
from robottraderslab_envelope.band_tracking import held_bands

BTC = Symbol.create("BTC/USDT:USDT")
SERIES_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
WARM_UP_CANDLES = 10
EARLIEST_EXECUTION = datetime.min.replace(tzinfo=timezone.utc)
FLAT_TOLERANCE = 1e-9

_RUNG_TAG = re.compile(r"^1h-([ls])(\d+)$")
_NO_BANDS: frozenset[int] = frozenset()

type Candle = tuple[float, float, float, float]
type BandsBySide = tuple[frozenset[int], frozenset[int]]
type Observation = tuple[datetime, frozenset[int], frozenset[int]]

_WARM_UP: list[Candle] = [(100.0, 100.5, 99.5, 100.0)] * WARM_UP_CANDLES
_QUIET: Candle = (90.0, 90.5, 89.5, 90.0)
_CANDLES: list[Candle] = (
    _WARM_UP
    + [
        (100.0, 100.0, 82.0, 82.5),
        (82.5, 84.0, 82.0, 83.0),
        (83.0, 86.0, 82.5, 85.0),
        (85.0, 91.0, 84.5, 90.0),
    ]
    + [_QUIET] * 6
    + [
        (90.0, 92.0, 89.5, 91.0),
        (91.0, 94.5, 90.5, 94.0),
        (94.0, 94.0, 88.5, 89.0),
    ]
    + [_QUIET] * 6
    + [
        (90.0, 90.0, 88.0, 89.5),
        (89.5, 92.0, 89.0, 91.0),
    ]
    + [_QUIET] * 6
    + [
        (90.0, 90.0, 88.0, 89.0),
        (89.0, 89.0, 78.0, 79.0),
        (79.0, 91.0, 79.0, 90.5),
    ]
    + [_QUIET] * 6
    + [
        (90.0, 90.0, 88.0, 89.5),
        (89.5, 92.0, 89.0, 91.0),
    ]
    + [_QUIET] * 6
    + [(90.0, 92.5, 88.0, 90.0)]
    + [_QUIET] * 4
)


@dataclass(frozen=True)
class BacktestRun:
    observations: list[Observation]
    fills: list[Execution]

    @property
    def expected(self) -> list[BandsBySide]:
        """The Envelope's exits are the position's stop-loss and take-profit,
        each of which closes the whole position, so a band is never released
        alone and a flat position holds none.
        """
        expected: list[BandsBySide] = []
        net = 0.0
        bands: dict[str, set[int]] = {"l": set(), "s": set()}
        pending = iter(self.fills)
        upcoming = next(pending, None)
        for moment, _, _ in self.observations:
            while upcoming is not None and upcoming.timestamp <= moment:
                net, bands = _after_fill(net, bands, upcoming)
                upcoming = next(pending, None)
            expected.append((frozenset(bands["l"]), frozenset(bands["s"])))
        return expected

    @property
    def held(self) -> list[BandsBySide]:
        return [(long, short) for _, long, short in self.observations]


def _after_fill(
    net: float, bands: dict[str, set[int]], fill: Execution
) -> tuple[float, dict[str, set[int]]]:
    before = net
    net += _signed_quantity(fill)
    if abs(net) < FLAT_TOLERANCE:
        return 0.0, {"l": set(), "s": set()}
    if abs(before) >= FLAT_TOLERANCE and (before > 0) != (net > 0):
        bands = {"l": set(), "s": set()}
    rung = _RUNG_TAG.match(tag_of(fill.client_order_id) or "")
    if rung is not None:
        bands[rung.group(1)].add(int(rung.group(2)))
    return net, bands


def _signed_quantity(fill: Execution) -> float:
    return fill.quantity if fill.side is OrderSide.BUY else -fill.quantity


def _exits_leaving_a_position(
    fills: Iterable[Execution],
) -> list[tuple[datetime, float, float]]:
    net = 0.0
    exits = []
    for fill in fills:
        before = net
        net += _signed_quantity(fill)
        is_exit = fill.kind in (OrderType.STOP_LOSS, OrderType.TAKE_PROFIT)
        if is_exit and abs(net) > FLAT_TOLERANCE:
            exits.append((fill.timestamp, before, net))
    return exits


def _sides_filled_at(fills: Iterable[Execution]) -> dict[datetime, set[str]]:
    sides: dict[datetime, set[str]] = {}
    for fill in fills:
        rung = _RUNG_TAG.match(tag_of(fill.client_order_id) or "")
        if rung is not None:
            sides.setdefault(fill.timestamp, set()).add(rung.group(1))
    return sides


def _csv_of(candles: Iterable[Candle]) -> StringIO:
    rows = ["date,open,high,low,close,volume"]
    for hour, (open_, high, low, close) in enumerate(candles):
        moment = SERIES_START + timedelta(hours=hour)
        rows.append(
            f"{moment:%Y-%m-%d %H:%M:%S}+0000,{open_},{high},{low},{close},1000"
        )
    return StringIO("\n".join(rows) + "\n")


def _recording_envelope(observations: list[Observation]) -> type[EnvelopeStrategy]:
    class RecordingEnvelope(EnvelopeStrategy):
        def book_profile_actions(
            self, profile, ohlcvs, account_snapshots, timestamp, bookkeeper
        ):
            observations.append(
                (
                    timestamp,
                    held_bands(profile, PositionSide.LONG, self._position_tracker),
                    held_bands(profile, PositionSide.SHORT, self._position_tracker),
                )
            )
            super().book_profile_actions(
                profile, ohlcvs, account_snapshots, timestamp, bookkeeper
            )

    return RecordingEnvelope


@pytest.fixture
def run(
    entry_order: str, make_profile_settings: Callable[..., dict[str, Any]]
) -> BacktestRun:
    observations: list[Observation] = []
    exchange = create_exchange(SimulatedFuturesExchange)
    strategy = create_envelope(
        _recording_envelope(observations),
        exchange,
        make_profile_settings(
            str(BTC),
            "1h",
            long_envelopes=[0.02, 0.04, 0.06],
            short_envelopes=[0.02, 0.04],
            average_period=5,
            stop_loss_pct=0.10,
            total_balance_ratio=0.3,
            leverage=5.0,
        ),
        entry_order,
    )
    start = SERIES_START + timedelta(hours=WARM_UP_CANDLES)
    end = SERIES_START + timedelta(hours=len(_CANDLES))
    backtester = create_backtester(
        strategy,
        exchange,
        CSVOHLCVProvider(file=_csv_of(_CANDLES), symbol=BTC, timeframe="1h"),
        f"{start:%Y-%m-%d %H:%M:%S}",
        f"{end:%Y-%m-%d %H:%M:%S}",
    )

    asyncio.run(backtester.run())

    return BacktestRun(
        observations=observations,
        fills=exchange.simulation_engine.get_executions_since(EARLIEST_EXECUTION),
    )


class TestHeldBandsAcrossABacktest:
    def test_the_held_bands_equal_the_bands_opened_since_flat_on_every_candle(
        self, run
    ):
        disagreeing = [
            (moment, expected, held)
            for (moment, _, _), expected, held in zip(
                run.observations, run.expected, run.held, strict=True
            )
            if held != expected
        ]

        assert disagreeing == []

    def test_every_exit_the_series_produces_closes_the_whole_position(self, run):
        exits_that_left_a_position = _exits_leaving_a_position(run.fills)

        assert exits_that_left_a_position == []


class TestTheSeriesProduces:
    def test_a_band_filled_without_the_bands_above_it(self, run):
        held_sets = [bands for state in run.expected for bands in state if bands]

        assert any(1 not in bands for bands in held_sets)

    def test_entries_after_flat_again_and_again(self, run):
        flat = (_NO_BANDS, _NO_BANDS)
        entries = [
            current
            for previous, current in zip(run.expected, run.expected[1:])
            if previous == flat and current != flat
        ]

        assert len(entries) >= 3

    def test_a_stop_loss_fill(self, run):
        kinds = {fill.kind for fill in run.fills}

        assert OrderType.STOP_LOSS in kinds

    def test_a_candle_filling_a_band_on_each_side(self, run):
        sides_filled_at = _sides_filled_at(run.fills)

        assert {"l", "s"} in sides_filled_at.values()

    def test_a_short_position_of_two_bands(self, run):
        short_sets = [short for _, short in run.expected]

        assert frozenset({1, 2}) in short_sets
