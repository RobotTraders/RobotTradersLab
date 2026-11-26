import numpy as np

from robottraderslab.indicators import MAType, trix
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


class TestComputeSignals:
    def test_a_flag_per_candle_per_side(self):
        signals = _signals(SURGE)

        assert len(signals.long_entry) == len(SURGE)
        assert len(signals.long_exit) == len(SURGE)
        assert len(signals.short_entry) == len(SURGE)
        assert len(signals.short_exit) == len(SURGE)

    def test_flat_prices_never_enter(self):
        signals = _signals(FLAT)

        assert not signals.long_entry.any()
        assert not signals.short_entry.any()

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

    def test_the_exit_stays_raised_while_the_side_stays_closed(self):
        signals = _signals(ROUND_TRIP)

        assert _stays_raised(signals.long_exit[_first_raised(signals.long_entry) :])

    def test_the_short_side_stays_closed_while_prices_climb(self):
        signals = _signals(SURGE)

        assert _stays_raised(signals.short_exit)

    def test_an_entry_candle_does_not_exit_its_own_side(self):
        signals = _signals(CYCLES)

        assert not (signals.long_entry & signals.long_exit).any()
        assert not (signals.short_entry & signals.short_exit).any()

    def test_a_collapse_after_a_climb_opens_the_short_side(self):
        signals = _signals(ROUND_TRIP)

        long_entry = _first_raised(signals.long_entry)
        first_long_exit = long_entry + _first_raised(signals.long_exit[long_entry:])
        assert _first_raised(signals.short_entry) > first_long_exit

    def test_the_moving_average_warmup_stays_silent(self):
        signals = _signals(ROUND_TRIP)

        assert not signals.long_entry[: TREND_LENGTH - 1].any()
        assert not signals.short_entry[: TREND_LENGTH - 1].any()
        assert not signals.long_exit[: TREND_LENGTH - 1].any()
        assert not signals.short_exit[: TREND_LENGTH - 1].any()

    def test_a_candle_the_trix_cannot_decide_raises_no_exit(self):
        close = np.array(ROUND_TRIP)
        undecided = np.isnan(trix(close, 5, 3, signal_type=MAType.SMA).histogram)

        signals = _signals(ROUND_TRIP, trend_length=2)

        assert undecided.any()
        assert not signals.long_exit[undecided].any()
        assert not signals.short_exit[undecided].any()

    def test_a_run_enters_once_and_its_side_closes_before_the_next_entry(self):
        signals = _signals(CYCLES)

        entries = np.flatnonzero(signals.long_entry)
        exits_so_far = np.cumsum(signals.long_exit)
        assert len(entries) >= 2
        assert (np.diff(exits_so_far[entries]) > 0).all()

    def test_a_candle_never_enters_both_sides(self):
        signals = _signals(CYCLES)

        assert not (signals.long_entry & signals.short_entry).any()


def _signals(close: list[float], trend_length: int = TREND_LENGTH) -> ImpulseSignals:
    return compute_signals(
        np.array(close),
        trix_length=5,
        signal_length=3,
        signal_type=MAType.SMA,
        trend_length=trend_length,
        trend_type=MAType.SMA,
    )


def _stays_raised(flags: np.ndarray) -> bool:
    return bool(flags[_first_raised(flags) :].all())


def _first_raised(flags: np.ndarray) -> int:
    assert flags.any()

    return int(np.argmax(flags))
