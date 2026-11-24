import numpy as np

from robottraderslab.indicators import MAType
from robottraderslab_impulse.impulse_indicator import ImpulseSignals, compute_signals

TREND_LENGTH = 10

FLAT = [100.0] * 40
SURGE = [100.0] * 30 + [100.0 + 3.0 * (candle + 1) for candle in range(30)]
SLIDE = [100.0] * 30 + [100.0 - 3.0 * (candle + 1) for candle in range(30)]
ROUND_TRIP = [100.0 + 3.0 * candle for candle in range(30)] + [
    187.0 - 3.0 * (candle + 1) for candle in range(30)
]
CYCLES = [
    100.0 + 15.0 * float(np.sin(2.0 * np.pi * candle / 20.0)) for candle in range(80)
]


def _signals(close: list[float]) -> ImpulseSignals:
    return compute_signals(
        np.array(close),
        trix_length=5,
        signal_length=3,
        signal_type=MAType.SMA,
        trend_length=TREND_LENGTH,
        trend_type=MAType.SMA,
    )


def _first(flags: np.ndarray) -> int:
    return int(np.argmax(flags))


class TestComputeSignals:
    def test_a_flag_per_candle_per_side(self):
        signals = _signals(SURGE)

        assert len(signals.long_entry) == len(SURGE)
        assert len(signals.long_exit) == len(SURGE)
        assert len(signals.short_entry) == len(SURGE)
        assert len(signals.short_exit) == len(SURGE)

    def test_flat_prices_never_trade(self):
        signals = _signals(FLAT)

        assert not signals.long_entry.any()
        assert not signals.long_exit.any()
        assert not signals.short_entry.any()
        assert not signals.short_exit.any()

    def test_a_surge_after_flat_prices_enters_long(self):
        signals = _signals(SURGE)

        assert not signals.long_entry[:30].any()
        assert signals.long_entry.any()
        assert not signals.short_entry.any()

    def test_a_slide_after_flat_prices_enters_short(self):
        signals = _signals(SLIDE)

        assert not signals.short_entry[:30].any()
        assert signals.short_entry.any()
        assert not signals.long_entry.any()

    def test_a_collapse_after_a_climb_exits_the_long(self):
        signals = _signals(ROUND_TRIP)

        assert signals.long_entry.any()
        assert signals.long_exit.any()
        assert _first(signals.long_exit) > _first(signals.long_entry)

    def test_a_collapse_after_a_climb_opens_the_short_side(self):
        signals = _signals(ROUND_TRIP)

        assert signals.short_entry.any()
        assert _first(signals.short_entry) > _first(signals.long_exit)

    def test_the_moving_average_warmup_stays_silent(self):
        signals = _signals(ROUND_TRIP)

        assert not signals.long_entry[: TREND_LENGTH - 1].any()
        assert not signals.short_entry[: TREND_LENGTH - 1].any()

    def test_a_run_trades_once_before_it_ends(self):
        signals = _signals(CYCLES)

        assert signals.long_entry.sum() >= 2
        holding = False
        for entry, leave in zip(signals.long_entry, signals.long_exit, strict=True):
            if entry:
                assert not holding
                holding = True
            if leave:
                assert holding
                holding = False

    def test_a_candle_never_signals_both_sides(self):
        signals = _signals(CYCLES)

        assert not (signals.long_entry & signals.short_entry).any()
        assert not (signals.long_exit & signals.short_exit).any()
