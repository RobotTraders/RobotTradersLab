import pytest

from robottraderslab import Symbol
from robottraderslab.exceptions import StrategyCriticalError
from robottraderslab.exchanges import OrderSide, StopLoss, TakeProfit

ENTRY_PRICE = 100.0


class TestTpSlValidation:
    @pytest.mark.parametrize(
        ("side", "sl_offset", "tp_offset", "expected_match"),
        [
            (OrderSide.BUY, 0, None, r"Invalid SL .* for long entry"),
            (OrderSide.BUY, 10, None, r"Invalid SL .* for long entry"),
            (OrderSide.BUY, None, 0, r"Invalid TP .* for long entry"),
            (OrderSide.BUY, None, -10, r"Invalid TP .* for long entry"),
            (OrderSide.SELL, 0, None, r"Invalid SL .* for short entry"),
            (OrderSide.SELL, -10, None, r"Invalid SL .* for short entry"),
            (OrderSide.SELL, None, 0, r"Invalid TP .* for short entry"),
            (OrderSide.SELL, None, 10, r"Invalid TP .* for short entry"),
        ],
    )
    async def test_invalid_tp_sl(
        self,
        sim,
        btc_usdt_perp: Symbol,
        side,
        sl_offset,
        tp_offset,
        expected_match,
    ):
        sl = (
            StopLoss(trigger_price=ENTRY_PRICE + sl_offset)
            if sl_offset is not None
            else None
        )
        tp = (
            TakeProfit(trigger_price=ENTRY_PRICE + tp_offset)
            if tp_offset is not None
            else None
        )

        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=side,
            quantity=1.0,
            stop_loss=sl,
            take_profit=tp,
        )

        with pytest.raises(StrategyCriticalError, match=expected_match):
            sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=ENTRY_PRICE)

        assert len(sim.open_orders) == 0

    @pytest.mark.parametrize(
        ("side", "sl_offset", "tp_offset", "expected_order_type"),
        [
            (OrderSide.BUY, -10, None, "stop-loss"),
            (OrderSide.BUY, None, 10, "take-profit"),
            (OrderSide.SELL, 10, None, "stop-loss"),
            (OrderSide.SELL, None, -10, "take-profit"),
        ],
    )
    async def test_valid_single_tp_or_sl(
        self,
        sim,
        btc_usdt_perp: Symbol,
        side,
        sl_offset,
        tp_offset,
        expected_order_type,
    ):
        sl = (
            StopLoss(trigger_price=ENTRY_PRICE + sl_offset)
            if sl_offset is not None
            else None
        )
        tp = (
            TakeProfit(trigger_price=ENTRY_PRICE + tp_offset)
            if tp_offset is not None
            else None
        )

        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=side,
            quantity=1.0,
            stop_loss=sl,
            take_profit=tp,
        )
        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp,
            open=ENTRY_PRICE,
            high=ENTRY_PRICE + 5,
            low=ENTRY_PRICE - 5,
            close=ENTRY_PRICE,
        )

        assert len(sim.open_positions) == 1
        assert len(sim.open_orders) == 1
        assert sim.open_orders[0].kind == expected_order_type

    @pytest.mark.parametrize(
        ("side", "sl_offset", "tp_offset"),
        [
            (OrderSide.BUY, -10, 10),
            (OrderSide.SELL, 10, -10),
        ],
    )
    async def test_valid_tp_and_sl(
        self, sim, btc_usdt_perp: Symbol, side, sl_offset, tp_offset
    ):
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=side,
            quantity=1.0,
            stop_loss=StopLoss(trigger_price=ENTRY_PRICE + sl_offset),
            take_profit=TakeProfit(trigger_price=ENTRY_PRICE + tp_offset),
        )
        sim.simulate_on_current_ohlcvs(
            btc_usdt_perp,
            open=ENTRY_PRICE,
            high=ENTRY_PRICE + 5,
            low=ENTRY_PRICE - 5,
            close=ENTRY_PRICE,
        )

        assert len(sim.open_positions) == 1
        assert len(sim.open_orders) == 2
