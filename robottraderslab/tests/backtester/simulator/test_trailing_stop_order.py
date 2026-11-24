import math

import pytest

from robottraderslab import Symbol
from robottraderslab.backtester.simulator.order_models import TrailingStopOrder
from robottraderslab.exchanges import OrderSide, PositionSide

TRAILING_SHARE = 0.1


def _trailing_stop(position_side: PositionSide, best_price: float) -> TrailingStopOrder:
    return TrailingStopOrder(
        position_side=position_side,
        symbol=Symbol.create("BTC/USDT:USDT"),
        quantity=None,
        best_price=best_price,
        trailing_share=TRAILING_SHARE,
        reason=None,
    )


@pytest.fixture
def long_stop() -> TrailingStopOrder:
    return _trailing_stop(PositionSide.LONG, best_price=100.0)


@pytest.fixture
def short_stop() -> TrailingStopOrder:
    return _trailing_stop(PositionSide.SHORT, best_price=100.0)


class TestLongTrailingStop:
    def test_starts_a_share_below_the_price_it_is_placed_at(self, long_stop):
        assert long_stop.trigger_price == pytest.approx(90.0)

    def test_closes_on_the_sell_side(self, long_stop):
        assert long_stop.side is OrderSide.SELL

    def test_the_high_of_a_candle_moves_the_level_up(self, long_stop):
        long_stop.trail(high=120.0, low=100.0)

        assert long_stop.trigger_price == pytest.approx(108.0)
        assert long_stop.best_price == 120.0

    def test_a_lower_high_never_loosens_the_level(self, long_stop):
        long_stop.trail(high=120.0, low=100.0)

        long_stop.trail(high=115.0, low=50.0)

        assert long_stop.trigger_price == pytest.approx(108.0)
        assert long_stop.best_price == 120.0

    def test_a_candle_without_a_price_leaves_the_level(self, long_stop):
        long_stop.trail(high=math.nan, low=math.nan)

        assert long_stop.trigger_price == pytest.approx(90.0)


class TestShortTrailingStop:
    def test_starts_a_share_above_the_price_it_is_placed_at(self, short_stop):
        assert short_stop.trigger_price == pytest.approx(110.0)

    def test_closes_on_the_buy_side(self, short_stop):
        assert short_stop.side is OrderSide.BUY

    def test_the_low_of_a_candle_moves_the_level_down(self, short_stop):
        short_stop.trail(high=95.0, low=80.0)

        assert short_stop.trigger_price == pytest.approx(88.0)
        assert short_stop.best_price == 80.0

    def test_a_higher_low_never_loosens_the_level(self, short_stop):
        short_stop.trail(high=95.0, low=80.0)

        short_stop.trail(high=200.0, low=85.0)

        assert short_stop.trigger_price == pytest.approx(88.0)
        assert short_stop.best_price == 80.0
