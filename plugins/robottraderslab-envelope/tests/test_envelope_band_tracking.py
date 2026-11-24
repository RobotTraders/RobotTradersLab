import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from robottraderslab import Symbol
from robottraderslab.strategies import (
    PositionSide,
    StrategyRequirements,
    profile_tag,
)
from robottraderslab_envelope import EnvelopeStrategy
from robottraderslab_envelope.band_tracking import (
    held_bands,
    tags_by_tracking_id,
    tracked_symbols,
)

BTC = Symbol.create("BTC/USDT:USDT")
ETH = Symbol.create("ETH/USDT:USDT")


class TestBandTracking:
    def test_every_id_carries_the_profile_tag_of_its_band_and_side(self, profiles):
        tags = tags_by_tracking_id(profiles)

        assert {
            tracking_id: tag
            for tracking_id, tag in tags.items()
            if tracked_symbols(profiles)[tracking_id] == BTC
        } == {
            "BTC/USDT:USDT#1d-l1": profile_tag("1d", "l1"),
            "BTC/USDT:USDT#1d-l2": profile_tag("1d", "l2"),
            "BTC/USDT:USDT#1d-l3": profile_tag("1d", "l3"),
            "BTC/USDT:USDT#1d-s1": profile_tag("1d", "s1"),
        }


class TestTrackerDeclaration:
    def test_declares_one_tracker_over_every_band_of_every_symbol(
        self, band_strategy, requirements
    ):
        asyncio.run(band_strategy.setup(requirements))

        (declared,) = requirements.tracker._get_all()
        assert declared.tracker is band_strategy._position_tracker
        assert declared.account is band_strategy.account
        assert declared.ids_by_symbol == {
            BTC: (
                "BTC/USDT:USDT#1d-l1",
                "BTC/USDT:USDT#1d-l2",
                "BTC/USDT:USDT#1d-l3",
                "BTC/USDT:USDT#1d-s1",
            ),
            ETH: (
                "ETH/USDT:USDT#1h-l1",
                "ETH/USDT:USDT#1h-l2",
                "ETH/USDT:USDT#1h-s1",
                "ETH/USDT:USDT#1h-s2",
            ),
        }


class TestHeldBands:
    @pytest.fixture
    def strategy(
        self, band_strategy: EnvelopeStrategy, requirements: StrategyRequirements
    ) -> EnvelopeStrategy:
        asyncio.run(band_strategy.setup(requirements))
        return band_strategy

    def test_the_bands_whose_rungs_filled_on_the_held_side(self, strategy, hold_bands):
        hold_bands(strategy, PositionSide.LONG, 1, 3)

        held = held_bands(
            strategy.profiles[0], PositionSide.LONG, strategy._position_tracker
        )

        assert held == {1, 3}

    def test_a_band_held_on_the_other_side_than_the_position(
        self, strategy, fold_fills, make_position, make_band_fill
    ):
        short_band_fill = make_band_fill(strategy.profiles[0], PositionSide.SHORT, 1)
        fold_fills([short_band_fill], make_position(PositionSide.LONG))

        held_on_the_position_side = held_bands(
            strategy.profiles[0], PositionSide.LONG, strategy._position_tracker
        )
        held_on_its_own_side = held_bands(
            strategy.profiles[0], PositionSide.SHORT, strategy._position_tracker
        )

        assert held_on_the_position_side == frozenset()
        assert held_on_its_own_side == {1}


class TestBandsOfOneNettedPosition:
    @pytest.fixture
    def strategy(
        self,
        make_strategy: Callable[..., EnvelopeStrategy],
        make_profile_settings: Callable[..., dict[str, Any]],
        requirements: StrategyRequirements,
    ) -> EnvelopeStrategy:
        strategy = make_strategy(
            [make_profile_settings("BTC/USDT:USDT", "1d", envelopes=[0.05, 0.10, 0.15])]
        )
        asyncio.run(strategy.setup(requirements))
        return strategy

    @pytest.fixture
    def held_by_side(
        self, strategy: EnvelopeStrategy
    ) -> Callable[[], dict[PositionSide, frozenset[int]]]:
        def _held() -> dict[PositionSide, frozenset[int]]:
            return {
                side: held_bands(strategy.profiles[0], side, strategy._position_tracker)
                for side in PositionSide
            }

        return _held

    def test_a_long_band_then_two_short_bands_leave_both_short_bands_held(
        self, strategy, fold_fills, held_by_side, make_band_fill, make_position
    ):
        profile = strategy.profiles[0]
        fills = [
            make_band_fill(profile, PositionSide.LONG, 1, quantity=1.0, minute=1),
            make_band_fill(profile, PositionSide.SHORT, 1, quantity=2.0, minute=2),
            make_band_fill(profile, PositionSide.SHORT, 2, quantity=2.0, minute=3),
        ]

        fold_fills(fills, make_position(PositionSide.SHORT, 3.0))

        assert held_by_side()[PositionSide.SHORT] == {1, 2}

    def test_two_short_bands_then_a_long_band_never_reach_flat_so_all_three_stay_held(
        self, strategy, fold_fills, held_by_side, make_band_fill, make_position
    ):
        profile = strategy.profiles[0]
        fills = [
            make_band_fill(profile, PositionSide.SHORT, 1, minute=1),
            make_band_fill(profile, PositionSide.SHORT, 2, minute=2),
            make_band_fill(profile, PositionSide.LONG, 1, minute=3),
        ]

        fold_fills(fills, make_position(PositionSide.SHORT, 1.0))

        assert held_by_side() == {
            PositionSide.LONG: {1},
            PositionSide.SHORT: {1, 2},
        }

    def test_a_long_and_a_short_band_netting_to_flat_leave_the_next_long_band_alone(
        self, strategy, fold_fills, held_by_side, make_band_fill, make_position
    ):
        profile = strategy.profiles[0]
        fills = [
            make_band_fill(profile, PositionSide.LONG, 1, minute=1),
            make_band_fill(profile, PositionSide.SHORT, 1, minute=2),
            make_band_fill(profile, PositionSide.LONG, 2, minute=3),
        ]

        fold_fills(fills, make_position(PositionSide.LONG, 1.0))

        assert held_by_side() == {
            PositionSide.LONG: {2},
            PositionSide.SHORT: frozenset(),
        }

    def test_a_long_band_and_a_short_band_of_another_index_netting_to_flat_leave_nothing_held(
        self, strategy, fold_fills, held_by_side, make_band_fill
    ):
        profile = strategy.profiles[0]
        fills = [
            make_band_fill(profile, PositionSide.LONG, 1, minute=1),
            make_band_fill(profile, PositionSide.SHORT, 2, minute=2),
        ]

        fold_fills(fills, None)

        assert held_by_side() == {
            PositionSide.LONG: frozenset(),
            PositionSide.SHORT: frozenset(),
        }
