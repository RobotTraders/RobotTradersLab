import logging
import random
from datetime import datetime, timedelta, timezone

import pytest

from robottraderslab_core import Symbol
from robottraderslab_core._internal.account_snapshot import AccountSnapshot
from robottraderslab_core._internal.actions.members.tags import profile_tag
from robottraderslab_core._internal.exchange_position_tracker import (
    ExchangePositionTracker,
)
from robottraderslab_core._internal.position_tracker import TrackingId
from robottraderslab_core.exchanges import (
    Execution,
    OrderSide,
    OrderType,
    PositionSide,
    PositionSnapshot,
    client_order_id_carrying,
)

BTC = Symbol.create("BTC/USDT:USDT")
ETH = Symbol.create("ETH/USDT:USDT")
SOL = Symbol.create("SOL/USDT:USDT")
BTC_ALPHA = TrackingId("BTC/USDT:USDT@1d#alpha")
BTC_BETA = TrackingId("BTC/USDT:USDT@1d#beta")
ETH_ALPHA = TrackingId("ETH/USDT:USDT@1d#alpha")
DECLARED_IDS = (BTC_ALPHA, BTC_BETA, ETH_ALPHA)

FIRST_CANDLE_OPEN = datetime(2026, 1, 1, tzinfo=timezone.utc)
CANDLE_LENGTH = timedelta(days=1)
CANDLES = 14
WINDOW_IN_CANDLES = 3
SEEDS = 60
FILLS_PER_CANDLE = (0, 0, 1, 1, 2, 3)
MINUTES_IN_A_CANDLE = range(6)
OFFSETS = (0, 0, 0, 2.0)
QUANTITIES = (1.0, 2.0, 3.0)
TAGS = ("1h-a", "1h-b", "9z-foreign", "sl", "tp", None)
KINDS = (None, None, None, OrderType.STOP_LOSS, OrderType.TAKE_PROFIT)
SYMBOLS = (BTC, ETH, SOL)
LATE_AT_LATEST_TIMESTAMP = 0.3
DRIFT_SHARE = 0.4
DRIFT_DELTAS = (1.0, -2.0)
DRIFT_START_CANDLES = range(1, CANDLES - 4)
DRIFT_LENGTHS_IN_CANDLES = range(1, 4)

type Drift = tuple[int, int, float]


@pytest.mark.parametrize("seed", range(SEEDS))
def test_every_candle_matches_a_recount_of_every_fill_from_scratch(seed, caplog):
    rng = random.Random(seed)
    caplog.set_level(logging.INFO)
    tracker = _tracker()
    base_offsets = {symbol: rng.choice(OFFSETS) for symbol in SYMBOLS}
    drifts = {symbol: _drift(rng) for symbol in SYMBOLS}
    fills_by_candle: list[list[Execution]] = []
    counters = {symbol: 0 for symbol in SYMBOLS}
    latest: dict[Symbol, datetime] = {}

    for candle in range(CANDLES):
        opened_at = FIRST_CANDLE_OPEN + candle * CANDLE_LENGTH
        closed_at = opened_at + CANDLE_LENGTH
        fills_by_candle.append(_fills_of_candle(rng, opened_at, counters, latest))
        every_fill = [fill for fills in fills_by_candle for fill in fills]
        window = [
            fill for fills in fills_by_candle[-WINDOW_IN_CANDLES:] for fill in fills
        ]
        rng.shuffle(window)
        offsets = _offsets(base_offsets, drifts, candle)
        positions = _venue_positions(every_fill, offsets)
        recounted_read = list(every_fill)
        rng.shuffle(recounted_read)

        caplog.clear()
        asking = tracker.refresh(_snapshot(window, positions), opened_at, closed_at)
        reports = sorted(caplog.messages)
        caplog.clear()
        recount = _tracker()
        recount_asking = recount.refresh(
            _snapshot(recounted_read, positions), opened_at, closed_at
        )
        recount_reports = sorted(caplog.messages)

        context = f"seed {seed}, candle {candle}"
        assert _held(tracker) == _held(recount), context
        assert tracker.unanchored_symbols() == recount.unanchored_symbols(), context
        assert _external_shares(tracker) == _external_shares(recount), context
        assert asking == recount_asking, context
        assert reports == recount_reports, context


def _tracker() -> ExchangePositionTracker:
    return ExchangePositionTracker.create(
        {BTC: (BTC_ALPHA, BTC_BETA), ETH: (ETH_ALPHA,)},
        {
            BTC_ALPHA: profile_tag("1h", "a"),
            BTC_BETA: profile_tag("1h", "b"),
            ETH_ALPHA: profile_tag("1h", "a"),
        },
    )


def _held(tracker):
    return {tracking_id: tracker.get(tracking_id) for tracking_id in DECLARED_IDS}


def _external_shares(tracker):
    """No accessor answers the external share of an unanchored symbol, whose
    report `refresh` skips, so the cursor's own fields are read.
    """
    return {
        symbol: (cursor.external_steps, cursor.external_opened_at)
        for symbol, cursor in tracker._cursors.items()
    }


def _drift(rng: random.Random) -> Drift | None:
    if rng.random() >= DRIFT_SHARE:
        return None
    first = rng.choice(DRIFT_START_CANDLES)
    return first, first + rng.choice(DRIFT_LENGTHS_IN_CANDLES), rng.choice(DRIFT_DELTAS)


def _offsets(
    base_offsets: dict[Symbol, float], drifts: dict[Symbol, Drift | None], candle: int
) -> dict[Symbol, float]:
    """A drifting venue position moves away from the fills with no fill, then
    comes back.
    """
    return {
        symbol: base
        + (
            drift[2]
            if (drift := drifts[symbol]) is not None and drift[0] <= candle < drift[1]
            else 0.0
        )
        for symbol, base in base_offsets.items()
    }


def _fills_of_candle(
    rng: random.Random,
    opened_at: datetime,
    counters: dict[Symbol, int],
    latest: dict[Symbol, datetime],
) -> list[Execution]:
    """Each symbol's fills arrive in the order of their timestamp then id, as
    a venue books them, with ties on a timestamp, and the first of a candle
    sometimes lands at the latest timestamp the symbol already folded.
    """
    fills = []
    for symbol in SYMBOLS:
        timestamps = sorted(
            opened_at + timedelta(minutes=rng.choice(MINUTES_IN_A_CANDLE))
            for _ in range(rng.choice(FILLS_PER_CANDLE))
        )
        if timestamps and symbol in latest and rng.random() < LATE_AT_LATEST_TIMESTAMP:
            timestamps[0] = latest[symbol]
        for timestamp in timestamps:
            counters[symbol] += 1
            tag = rng.choice(TAGS)
            fills.append(
                Execution(
                    execution_id=f"{symbol}-{counters[symbol]:04d}",
                    order_id=f"order-{symbol}-{counters[symbol]:04d}",
                    symbol=symbol,
                    side=rng.choice((OrderSide.BUY, OrderSide.SELL)),
                    price=100.0,
                    quantity=rng.choice(QUANTITIES),
                    timestamp=timestamp,
                    effect="open",
                    kind=rng.choice(KINDS),
                    client_order_id=(
                        None if tag is None else client_order_id_carrying(tag)
                    ),
                )
            )
        if timestamps:
            latest[symbol] = timestamps[-1]
    return fills


def _venue_positions(
    fills: list[Execution], offsets: dict[Symbol, float]
) -> dict[Symbol, PositionSnapshot]:
    nets = dict(offsets)
    for fill in fills:
        nets[fill.symbol] += (
            fill.quantity if fill.side == OrderSide.BUY else -fill.quantity
        )
    return {
        symbol: PositionSnapshot(
            symbol=symbol,
            side=PositionSide.LONG if net > 0 else PositionSide.SHORT,
            quantity=abs(net),
            average_entry_price=100.0,
            entry_time=FIRST_CANDLE_OPEN,
            leverage=1.0,
            liquidation_price=0.0,
            stop_loss_price=None,
            take_profit_price=None,
        )
        for symbol, net in nets.items()
        if net != 0
    }


def _snapshot(
    executions: list[Execution], positions: dict[Symbol, PositionSnapshot]
) -> AccountSnapshot:
    return AccountSnapshot(
        account_name="test",
        positions=positions,
        executions=executions,
        executions_declared=True,
    )
