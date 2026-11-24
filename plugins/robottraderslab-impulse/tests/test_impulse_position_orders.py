from robottraderslab.strategies import (
    PositionSide,
    TrackedPosition,
)
from robottraderslab.strategies.futures import TotalBalanceRatio
from robottraderslab_impulse.position_orders import wanted_entry_side, wants_exit
from robottraderslab_impulse.profile_config import ProfileConfig
from robottraderslab_impulse.profile_snapshot import ProfileSnapshot


def _profile(**overrides) -> ProfileConfig:
    return ProfileConfig(
        sizing=TotalBalanceRatio(1.0),
        symbol="BTC/USDT:USDT",
        timeframe="1d",
        trix_length=9,
        signal_length=21,
        trend_length=200,
        **overrides,
    )


def _snapshot(**signals) -> ProfileSnapshot:
    return ProfileSnapshot(
        price=100.0,
        long_entry=signals.get("long_entry", False),
        long_exit=signals.get("long_exit", False),
        short_entry=signals.get("short_entry", False),
        short_exit=signals.get("short_exit", False),
        position=signals.get("position"),
    )


class TestWantedEntrySide:
    def test_a_long_signal(self):
        wanted = wanted_entry_side(_profile(), _snapshot(long_entry=True))

        assert wanted == PositionSide.LONG

    def test_a_short_signal(self):
        wanted = wanted_entry_side(_profile(), _snapshot(short_entry=True))

        assert wanted == PositionSide.SHORT

    def test_no_signal(self):
        assert wanted_entry_side(_profile(), _snapshot()) is None

    def test_short_only_refuses_a_long(self):
        wanted = wanted_entry_side(
            _profile(short_only=True), _snapshot(long_entry=True)
        )

        assert wanted is None

    def test_long_only_refuses_a_short(self):
        wanted = wanted_entry_side(
            _profile(long_only=True), _snapshot(short_entry=True)
        )

        assert wanted is None

    def test_a_short_only_profile_hearing_both_signals(self):
        wanted = wanted_entry_side(
            _profile(short_only=True), _snapshot(long_entry=True, short_entry=True)
        )

        assert wanted == PositionSide.SHORT


class TestWantsExit:
    def test_a_long_told_to_leave(self):
        position = TrackedPosition(side=PositionSide.LONG, quantity=1.0)

        assert wants_exit(position, _snapshot(long_exit=True))

    def test_a_short_told_to_leave(self):
        position = TrackedPosition(side=PositionSide.SHORT, quantity=1.0)

        assert wants_exit(position, _snapshot(short_exit=True))

    def test_a_long_hearing_only_the_short_exit(self):
        position = TrackedPosition(side=PositionSide.LONG, quantity=1.0)

        assert not wants_exit(position, _snapshot(short_exit=True))

    def test_a_short_hearing_only_the_long_exit(self):
        position = TrackedPosition(side=PositionSide.SHORT, quantity=1.0)

        assert not wants_exit(position, _snapshot(long_exit=True))
