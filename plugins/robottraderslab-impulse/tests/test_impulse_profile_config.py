import pytest

from robottraderslab.exceptions import BotCriticalError, ProfileConfigError
from robottraderslab.indicators import MAType
from robottraderslab.strategies.futures import (
    MarginMode,
    TotalBalanceRatio,
)
from robottraderslab_impulse.profile_config import ProfileConfig


class TestDirectionFlags:
    def test_both_flags_false(self):
        profile = ProfileConfig(
            sizing=TotalBalanceRatio(1.0),
            symbol="BTC/USDT:USDT",
            timeframe="15m",
            trix_length=9,
            signal_length=21,
            long_only=False,
            short_only=False,
        )

        assert profile.long_only is False
        assert profile.short_only is False

    def test_long_only(self):
        profile = ProfileConfig(
            sizing=TotalBalanceRatio(1.0),
            symbol="BTC/USDT:USDT",
            timeframe="15m",
            trix_length=9,
            signal_length=21,
            long_only=True,
            short_only=False,
        )

        assert profile.long_only is True
        assert profile.short_only is False

    def test_short_only(self):
        profile = ProfileConfig(
            sizing=TotalBalanceRatio(1.0),
            symbol="BTC/USDT:USDT",
            timeframe="15m",
            trix_length=9,
            signal_length=21,
            long_only=False,
            short_only=True,
        )

        assert profile.long_only is False
        assert profile.short_only is True

    def test_contradictory_flags(self):
        with pytest.raises(ProfileConfigError, match="cannot both be true"):
            ProfileConfig(
                sizing=TotalBalanceRatio(1.0),
                symbol="BTC/USDT:USDT",
                timeframe="15m",
                trix_length=9,
                signal_length=21,
                long_only=True,
                short_only=True,
            )


class TestMATypeParameters:
    def test_built_from_the_strings_a_section_spells(self, make_strategy):
        strategy = make_strategy(
            [
                {
                    "symbol": "BTC/USDT:USDT",
                    "timeframe": "15m",
                    "total_balance_ratio": 1.0,
                    "trix_length": 9,
                    "signal_length": 21,
                    "signal_type": "WMA",
                    "trend_type": "EMA",
                    "margin_mode": "cross",
                }
            ]
        )

        (profile,) = strategy.profiles
        assert profile.signal_type is MAType.WMA
        assert profile.trend_type is MAType.EMA
        assert profile.margin_mode is MarginMode.CROSS

    def test_defaults_to_sma(self):
        profile = ProfileConfig(
            sizing=TotalBalanceRatio(1.0),
            symbol="BTC/USDT:USDT",
            timeframe="15m",
            trix_length=9,
            signal_length=21,
        )

        assert profile.signal_type is MAType.SMA
        assert profile.trend_type is MAType.SMA

    def test_with_unknown_signal_type(self, make_strategy):
        with pytest.raises(BotCriticalError, match="signal_type: Input should be"):
            make_strategy(
                [
                    {
                        "symbol": "BTC/USDT:USDT",
                        "timeframe": "15m",
                        "total_balance_ratio": 1.0,
                        "trix_length": 9,
                        "signal_length": 21,
                        "signal_type": "DMA",
                    }
                ]
            )


class TestTag:
    def test_accepts_plain_word(self):
        profile = ProfileConfig(
            sizing=TotalBalanceRatio(1.0),
            symbol="BTC/USDT:USDT",
            timeframe="15m",
            trix_length=9,
            signal_length=21,
            tag="alpha",
        )

        assert profile.tag == "alpha"


class TestStopLossPct:
    def test_defaults_to_none(self):
        profile = ProfileConfig(
            sizing=TotalBalanceRatio(1.0),
            symbol="BTC/USDT:USDT",
            timeframe="15m",
            trix_length=9,
            signal_length=21,
        )

        assert profile.stop_loss_pct is None

    def test_accepts_fraction(self):
        profile = ProfileConfig(
            sizing=TotalBalanceRatio(1.0),
            symbol="BTC/USDT:USDT",
            timeframe="15m",
            trix_length=9,
            signal_length=21,
            stop_loss_pct=0.1,
        )

        assert profile.stop_loss_pct == 0.1

    @pytest.mark.parametrize("stop_loss_pct", [-0.1, 0.0, 1.0, 1.5])
    def test_rejects_out_of_range(self, stop_loss_pct):
        with pytest.raises(ProfileConfigError, match="stop_loss_pct"):
            ProfileConfig(
                sizing=TotalBalanceRatio(1.0),
                symbol="BTC/USDT:USDT",
                timeframe="15m",
                trix_length=9,
                signal_length=21,
                stop_loss_pct=stop_loss_pct,
            )


class TestOrderTag:
    def test_composes_the_timeframe_with_the_tag(self):
        profile = ProfileConfig(
            sizing=TotalBalanceRatio(1.0),
            symbol="BTC/USDT:USDT",
            timeframe="1h",
            trix_length=9,
            signal_length=21,
            tag="alpha",
        )

        assert profile.order_tag == "1h-alpha"

    def test_empty_tag_leaves_just_the_timeframe(self):
        profile = ProfileConfig(
            sizing=TotalBalanceRatio(1.0),
            symbol="BTC/USDT:USDT",
            timeframe="1h",
            trix_length=9,
            signal_length=21,
        )

        assert profile.order_tag == "1h"


class TestSignalColumns:
    def test_each_column_carries_the_profile_id(self):
        profile = ProfileConfig(
            sizing=TotalBalanceRatio(1.0),
            symbol="BTC/USDT:USDT",
            timeframe="1d",
            trix_length=9,
            signal_length=21,
        )

        assert profile.long_entry_column == f"{profile.profile_id}_long_entry"
        assert profile.long_exit_column == f"{profile.profile_id}_long_exit"
        assert profile.short_entry_column == f"{profile.profile_id}_short_entry"
        assert profile.short_exit_column == f"{profile.profile_id}_short_exit"

    def test_stacked_profiles_do_not_share_a_column(self):
        first = ProfileConfig(
            sizing=TotalBalanceRatio(1.0),
            symbol="BTC/USDT:USDT",
            timeframe="1d",
            trix_length=9,
            signal_length=21,
            tag="fast",
        )
        second = ProfileConfig(
            sizing=TotalBalanceRatio(1.0),
            symbol="BTC/USDT:USDT",
            timeframe="1d",
            trix_length=30,
            signal_length=60,
            tag="slow",
        )

        assert first.long_entry_column != second.long_entry_column
