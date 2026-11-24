import asyncio
from collections.abc import Callable

import pytest

from robottraderslab import Symbol
from robottraderslab.exceptions import BotCriticalError
from robottraderslab.strategies import StrategyRequirements
from robottraderslab.strategies.futures import MarginMode, MarginSettings
from robottraderslab_impulse import ImpulseStrategy


def _profile(tag: str, stop_loss_pct: float) -> dict:
    return {
        "symbol": "BTC/USDT:USDT",
        "timeframe": "1d",
        "total_balance_ratio": 1.0,
        "trix_length": 9,
        "signal_length": 21,
        "trend_length": 200,
        "tag": tag,
        "stop_loss_pct": stop_loss_pct,
    }


class TestSetup:
    @pytest.fixture
    def single_profile_strategy(
        self, make_strategy: Callable[..., ImpulseStrategy]
    ) -> ImpulseStrategy:
        return make_strategy(
            [
                {
                    "symbol": "BTC/USDT:USDT",
                    "timeframe": "1d",
                    "total_balance_ratio": 1.0,
                    "trix_length": 9,
                    "signal_length": 21,
                    "trend_length": 200,
                }
            ]
        )

    @pytest.fixture
    def multi_profile_strategy(
        self, make_strategy: Callable[..., ImpulseStrategy]
    ) -> ImpulseStrategy:
        return make_strategy(
            [
                {
                    "symbol": "BTC/USDT:USDT",
                    "timeframe": "1d",
                    "total_balance_ratio": 1.0,
                    "trix_length": 9,
                    "signal_length": 21,
                    "trend_length": 200,
                },
                {
                    "symbol": "ETH/USDT:USDT",
                    "timeframe": "1h",
                    "total_balance_ratio": 1.0,
                    "trix_length": 12,
                    "signal_length": 26,
                    "trend_length": 50,
                },
            ]
        )

    def test_single_profile_registers_requirement(self, single_profile_strategy):
        requirements = StrategyRequirements()

        asyncio.run(single_profile_strategy.setup(requirements))

        registered = requirements.ohlcv._get_all()
        assert len(registered) == 1
        assert registered[0].symbol == Symbol.create("BTC/USDT:USDT")
        assert registered[0].timeframe == "1d"
        assert registered[0].lookback == 200

    def test_multiple_profiles_register_each_symbol_timeframe(
        self, multi_profile_strategy
    ):
        requirements = StrategyRequirements()

        asyncio.run(multi_profile_strategy.setup(requirements))

        registered = requirements.ohlcv._get_all()
        assert len(registered) == 2

    @pytest.mark.parametrize(
        ("trix_length", "signal_length", "trend_length", "expected_lookback"),
        [
            (9, 21, 200, 200),
            (300, 21, 200, 300),
            (9, 500, 200, 500),
        ],
        ids=["trend_largest", "trix_largest", "signal_largest"],
    )
    def test_lookback_uses_max_indicator_parameter(
        self, make_strategy, trix_length, signal_length, trend_length, expected_lookback
    ):
        strategy = make_strategy(
            [
                {
                    "symbol": "BTC/USDT:USDT",
                    "timeframe": "15m",
                    "total_balance_ratio": 1.0,
                    "trix_length": trix_length,
                    "signal_length": signal_length,
                    "trend_length": trend_length,
                }
            ]
        )
        requirements = StrategyRequirements()

        asyncio.run(strategy.setup(requirements))

        registered = requirements.ohlcv._get_all()
        assert registered[0].lookback == expected_lookback

    def test_same_timeframe_uses_max_lookback_across_profiles(self, make_strategy):
        strategy = make_strategy(
            [
                {
                    "symbol": "BTC/USDT:USDT",
                    "timeframe": "15m",
                    "total_balance_ratio": 1.0,
                    "trix_length": 9,
                    "signal_length": 21,
                    "trend_length": 100,
                },
                {
                    "symbol": "ETH/USDT:USDT",
                    "timeframe": "15m",
                    "total_balance_ratio": 1.0,
                    "trix_length": 12,
                    "signal_length": 26,
                    "trend_length": 200,
                },
            ]
        )
        requirements = StrategyRequirements()

        asyncio.run(strategy.setup(requirements))

        registered = requirements.ohlcv._get_all()
        lookbacks = [r.lookback for r in registered]
        assert max(lookbacks) == 200


class TestStopLossValidation:
    def test_conflicting_stop_loss_pct_on_same_symbol_rejected(self, make_strategy):
        strategy = make_strategy(
            [_profile("a", stop_loss_pct=0.05), _profile("b", stop_loss_pct=0.10)]
        )

        with pytest.raises(BotCriticalError, match="stop_loss_pct"):
            asyncio.run(strategy.setup(StrategyRequirements()))

    def test_matching_stop_loss_pct_on_same_symbol_allowed(self, make_strategy):
        strategy = make_strategy(
            [_profile("a", stop_loss_pct=0.05), _profile("b", stop_loss_pct=0.05)]
        )
        requirements = StrategyRequirements()

        asyncio.run(strategy.setup(requirements))

        assert requirements.ohlcv._get_all()

    def test_mixing_set_and_unset_on_same_symbol_rejected(self, make_strategy):
        strategy = make_strategy(
            [_profile("a", stop_loss_pct=0.05), _profile("b", stop_loss_pct=None)]
        )

        with pytest.raises(BotCriticalError, match="stop_loss_pct"):
            asyncio.run(strategy.setup(StrategyRequirements()))


class TestMarginTargets:
    def test_each_symbol_declares_its_profiles_settings(self, make_strategy):
        strategy = make_strategy(
            [{**_profile("a", 0.05), "leverage": 3.0, "margin_mode": "cross"}]
        )
        requirements = StrategyRequirements()

        asyncio.run(strategy.setup(requirements))

        assert requirements.account._get_all()[0].margin_targets == {
            Symbol.create("BTC/USDT:USDT"): MarginSettings(
                leverage=3.0, margin_mode=MarginMode.CROSS
            )
        }

    @pytest.mark.parametrize(
        ("key", "first", "second"),
        [("leverage", 2.0, 3.0), ("margin_mode", "isolated", "cross")],
    )
    def test_stacked_profiles_disagreeing_stop_the_run(
        self, make_strategy, key, first, second
    ):
        strategy = make_strategy(
            [
                {**_profile("a", 0.05), key: first},
                {**_profile("b", 0.05), key: second},
            ]
        )

        with pytest.raises(BotCriticalError, match="BTC/USDT:USDT"):
            asyncio.run(strategy.setup(StrategyRequirements()))


def _timeframe_profile(timeframe: str, tag: str) -> dict:
    return {
        "symbol": "BTC/USDT:USDT",
        "timeframe": timeframe,
        "total_balance_ratio": 1.0,
        "trix_length": 9,
        "signal_length": 21,
        "trend_length": 200,
        "tag": tag,
    }


class TestTagCollisionAcrossTimeframes:
    def test_same_tag_on_different_timeframes_is_not_a_collision(self, make_strategy):
        strategy = make_strategy(
            [
                _timeframe_profile("1h", "alpha"),
                _timeframe_profile("2h", "alpha"),
            ]
        )

        asyncio.run(strategy.setup(StrategyRequirements()))


def _sized_by(key: str, value: object) -> dict:
    return {
        "symbol": "BTC/USDT:USDT",
        "timeframe": "1d",
        key: value,
        "trix_length": 9,
        "signal_length": 21,
    }


class TestSizingReads:
    def test_a_rule_reading_equity_declares_it(self, make_strategy):
        strategy = make_strategy([_sized_by("risk_ratio", 0.02)])
        requirements = StrategyRequirements()

        asyncio.run(strategy.setup(requirements))

        (account_requirement,) = requirements.account._get_all()
        assert account_requirement.equity == ("USDT",)

    def test_a_rule_reading_the_balance_declares_no_equity(self, make_strategy):
        strategy = make_strategy([_sized_by("total_balance_ratio", 0.5)])
        requirements = StrategyRequirements()

        asyncio.run(strategy.setup(requirements))

        (account_requirement,) = requirements.account._get_all()
        assert account_requirement.equity == ()
