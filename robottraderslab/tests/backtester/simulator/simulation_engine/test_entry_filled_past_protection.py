import pytest
from gapped_entry_helpers import (
    LEVEL,
    MAKER,
    QUANTITY,
    TAKER,
    last_execution,
    place_limit_entry,
    place_trigger_entry,
)
from simulation_fixture import SimulationFixture

from robottraderslab import Symbol
from robottraderslab.backtester.simulator import FeeRates
from robottraderslab.exceptions import BotCriticalError
from robottraderslab.exchanges import OrderSide, StopLoss, TimeInForce

QUANTITY_AFTER_ENTRY_FEE = QUANTITY * (1 - TAKER)


@pytest.fixture
def fee_rates() -> FeeRates:
    return FeeRates(maker=MAKER, taker=TAKER)


def _gap_to(sim: SimulationFixture, symbol: Symbol, open_price: float) -> None:
    sim.simulate_on_current_ohlcvs(
        symbol,
        open=open_price,
        high=open_price + 1.0,
        low=open_price - 1.0,
        close=open_price,
    )


def _assert_entered_and_exited_at(
    sim: SimulationFixture, fill_price: float, exit_reason: str
) -> None:
    fills = sim.fill_recorder.get_fills()
    assert list(fills["price"]) == [fill_price, fill_price]
    assert fills.index.nunique() == 1
    assert fills["reason"].iloc[-1] == exit_reason
    assert sim.open_positions == {}
    assert sim.open_orders == []


class TestLimitEntryFilledPastItsStopLoss:
    @pytest.mark.parametrize(
        ("side", "stop_loss", "open_price", "fill_types"),
        [
            (OrderSide.BUY, 99.68, 99.0, ["enter_long", "exit_long"]),
            (OrderSide.SELL, 100.32, 101.0, ["enter_short", "exit_short"]),
        ],
    )
    async def test_limit_entry_gapped_through_its_stop_loss(
        self, sim, btc_usdt_perp, side, stop_loss, open_price, fill_types
    ):
        await place_limit_entry(sim, btc_usdt_perp, side, stop_loss=stop_loss)

        _gap_to(sim, btc_usdt_perp, open_price)

        _assert_entered_and_exited_at(sim, open_price, "stop-loss")
        assert list(sim.fill_recorder.get_fills()["fill_type"]) == fill_types

    async def test_ioc_limit_entry_placed_past_its_stop_loss(self, sim, btc_usdt_perp):
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=99.0)

        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=QUANTITY,
            price=LEVEL,
            time_in_force=TimeInForce.IOC,
            stop_loss=StopLoss(trigger_price=99.5),
        )

        _assert_entered_and_exited_at(sim, 99.0, "stop-loss")

    async def test_the_take_profit_resting_beside_it_is_cancelled(
        self, sim, btc_usdt_perp
    ):
        await place_limit_entry(
            sim, btc_usdt_perp, OrderSide.BUY, stop_loss=99.68, take_profit=110.0
        )

        _gap_to(sim, btc_usdt_perp, 99.0)

        assert sim.open_orders == []

    async def test_candle_after_the_stop_out_adds_no_fill(self, sim, btc_usdt_perp):
        await place_limit_entry(sim, btc_usdt_perp, OrderSide.BUY, stop_loss=99.68)
        _gap_to(sim, btc_usdt_perp, 99.0)

        _gap_to(sim, btc_usdt_perp, 99.5)

        _assert_entered_and_exited_at(sim, 99.0, "stop-loss")

    async def test_flipped_remainder_closes_at_the_fill_price(self, sim, btc_usdt_perp):
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=LEVEL)
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp, side=OrderSide.SELL, quantity=1.0
        )
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=LEVEL)
        await place_limit_entry(sim, btc_usdt_perp, OrderSide.BUY, stop_loss=99.68)

        _gap_to(sim, btc_usdt_perp, 99.0)

        fills = sim.fill_recorder.get_fills()
        assert list(fills["fill_type"]) == [
            "enter_short",
            "exit_short",
            "enter_long",
            "exit_long",
        ]
        assert list(fills["price"]) == [LEVEL, 99.0, 99.0, 99.0]
        assert sim.open_positions == {}

    async def test_stop_limit_entry_gapped_through_its_stop_loss(
        self, sim, btc_usdt_perp
    ):
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=105.0)
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=QUANTITY,
            price=LEVEL,
            trigger_price=98.0,
            stop_loss=StopLoss(trigger_price=99.0),
        )

        _gap_to(sim, btc_usdt_perp, 97.0)

        _assert_entered_and_exited_at(sim, 97.0, "stop-loss")

    @pytest.mark.parametrize(
        ("side", "stop_loss", "open_price"),
        [
            (OrderSide.BUY, 99.68, 99.68),
            (OrderSide.SELL, 100.32, 100.32),
        ],
    )
    async def test_limit_entry_filled_exactly_at_its_stop_loss(
        self, sim, btc_usdt_perp, side, stop_loss, open_price
    ):
        await place_limit_entry(sim, btc_usdt_perp, side, stop_loss=stop_loss)

        _gap_to(sim, btc_usdt_perp, open_price)

        _assert_entered_and_exited_at(sim, open_price, "stop-loss")

    async def test_stop_loss_exit_pays_the_taker_fee(self, sim, btc_usdt_perp):
        await place_limit_entry(sim, btc_usdt_perp, OrderSide.BUY, stop_loss=99.68)

        _gap_to(sim, btc_usdt_perp, 99.0)

        assert (await last_execution(sim)).fee == pytest.approx(
            QUANTITY_AFTER_ENTRY_FEE * 99.0 * TAKER
        )


class TestLimitEntryAddedToAGuardedPosition:
    async def test_add_on_gapped_through_its_own_stop_loss_closes_only_its_quantity(
        self, sim, btc_usdt_perp
    ):
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=LEVEL)
        await sim.exchange.place_market_order(
            symbol=btc_usdt_perp,
            side=OrderSide.BUY,
            quantity=QUANTITY,
            stop_loss=StopLoss(trigger_price=95.0),
        )
        await place_limit_entry(sim, btc_usdt_perp, OrderSide.BUY, stop_loss=99.68)

        _gap_to(sim, btc_usdt_perp, 99.0)

        fills = sim.fill_recorder.get_fills()
        resting_guard_levels = [order.trigger_price for order in sim.open_orders]
        assert list(fills["fill_type"]) == ["enter_long", "add_to_long", "reduce_long"]
        assert list(fills["price"]) == [LEVEL, 99.0, 99.0]
        assert fills["reason"].iloc[-1] == "stop-loss"
        assert sim.open_positions[btc_usdt_perp].quantity == pytest.approx(
            QUANTITY_AFTER_ENTRY_FEE
        )
        assert resting_guard_levels == [95.0]


class TestTriggerEntryFilledPastItsProtection:
    @pytest.mark.parametrize(
        ("side", "placement_close", "stop_loss", "open_price"),
        [
            (OrderSide.BUY, 105.0, 98.0, 97.0),
            (OrderSide.SELL, 95.0, 102.0, 103.0),
        ],
    )
    async def test_stop_loss_closes_at_the_fill_price_on_the_same_candle(
        self, sim, btc_usdt_perp, side, placement_close, stop_loss, open_price
    ):
        await place_trigger_entry(
            sim, btc_usdt_perp, side, placement_close, stop_loss=stop_loss
        )

        _gap_to(sim, btc_usdt_perp, open_price)

        _assert_entered_and_exited_at(sim, open_price, "stop-loss")

    @pytest.mark.parametrize(
        ("side", "placement_close", "take_profit", "open_price"),
        [
            (OrderSide.BUY, 95.0, 102.0, 103.0),
            (OrderSide.SELL, 105.0, 98.0, 97.0),
        ],
    )
    async def test_take_profit_closes_at_the_fill_price_on_the_same_candle(
        self, sim, btc_usdt_perp, side, placement_close, take_profit, open_price
    ):
        await place_trigger_entry(
            sim, btc_usdt_perp, side, placement_close, take_profit=take_profit
        )

        _gap_to(sim, btc_usdt_perp, open_price)

        _assert_entered_and_exited_at(sim, open_price, "take-profit")

    @pytest.mark.parametrize(
        ("side", "placement_close", "take_profit", "open_price"),
        [
            (OrderSide.BUY, 95.0, 102.0, 102.0),
            (OrderSide.SELL, 105.0, 98.0, 98.0),
        ],
    )
    async def test_trigger_entry_filled_exactly_at_its_take_profit(
        self, sim, btc_usdt_perp, side, placement_close, take_profit, open_price
    ):
        await place_trigger_entry(
            sim, btc_usdt_perp, side, placement_close, take_profit=take_profit
        )

        _gap_to(sim, btc_usdt_perp, open_price)

        _assert_entered_and_exited_at(sim, open_price, "take-profit")


class TestProtectionOnTheWrongSideOfTheBookedPrice:
    @pytest.mark.parametrize(
        ("side", "stop_loss", "take_profit", "open_price", "expected_match"),
        [
            (
                OrderSide.BUY,
                101.0,
                None,
                99.0,
                r"Invalid SL 101.0 for long entry .* at 100.0",
            ),
            (
                OrderSide.BUY,
                None,
                99.5,
                99.0,
                r"Invalid TP 99.5 for long entry .* at 100.0",
            ),
            (
                OrderSide.SELL,
                99.0,
                None,
                101.0,
                r"Invalid SL 99.0 for short entry .* at 100.0",
            ),
            (
                OrderSide.SELL,
                None,
                100.5,
                101.0,
                r"Invalid TP 100.5 for short entry .* at 100.0",
            ),
        ],
    )
    async def test_limit_entry_with_protection_on_the_wrong_side_of_its_limit_price(
        self,
        sim,
        btc_usdt_perp,
        side,
        stop_loss,
        take_profit,
        open_price,
        expected_match,
    ):
        await place_limit_entry(
            sim, btc_usdt_perp, side, stop_loss=stop_loss, take_profit=take_profit
        )

        with pytest.raises(BotCriticalError, match=expected_match):
            _gap_to(sim, btc_usdt_perp, open_price)

    @pytest.mark.parametrize(
        (
            "side",
            "placement_close",
            "stop_loss",
            "take_profit",
            "open_price",
            "expected_match",
        ),
        [
            (
                OrderSide.BUY,
                95.0,
                100.5,
                None,
                103.0,
                r"Invalid SL 100.5 for long entry .* at 100.0",
            ),
            (
                OrderSide.BUY,
                105.0,
                None,
                99.0,
                97.0,
                r"Invalid TP 99.0 for long entry .* at 100.0",
            ),
            (
                OrderSide.SELL,
                105.0,
                99.5,
                None,
                97.0,
                r"Invalid SL 99.5 for short entry .* at 100.0",
            ),
            (
                OrderSide.SELL,
                95.0,
                None,
                101.0,
                103.0,
                r"Invalid TP 101.0 for short entry .* at 100.0",
            ),
        ],
    )
    async def test_trigger_entry_with_protection_on_the_wrong_side_of_its_trigger_level(
        self,
        sim,
        btc_usdt_perp,
        side,
        placement_close,
        stop_loss,
        take_profit,
        open_price,
        expected_match,
    ):
        await place_trigger_entry(
            sim,
            btc_usdt_perp,
            side,
            placement_close,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        with pytest.raises(BotCriticalError, match=expected_match):
            _gap_to(sim, btc_usdt_perp, open_price)
