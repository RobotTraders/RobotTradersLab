import pytest

from robottraderslab._core import PlacementReserve
from robottraderslab.backtester.simulator import fee_model_for

NOTIONAL = 100_000.0
LEVERAGE = 10.0
TAKER_FEE_RATE = 0.001
MARGIN = NOTIONAL / LEVERAGE
FEE = NOTIONAL * TAKER_FEE_RATE
ORDER_FEE = 30.0
CLOSE_STEPS = 10
REQUESTED_STEPS = 15
BETWEEN_TWO_STEPS = 0.123456787
REPORTED_QUANTITY = 0.12345679
_NO_RESERVE = PlacementReserve(margin_markup=0.0, fee_markup=0.0, notional_reserve=0.0)


class TestPlacementRequirement:
    """Each share of the reserve marks up one term and one alone, so a share
    landing on the wrong term shifts the requirement by an amount the combined
    boundary would hide.
    """

    def test_cost_holds_nothing_back(self):
        model = fee_model_for(
            "cost",
            PlacementReserve(margin_markup=0.5, fee_markup=0.0, notional_reserve=0.0),
        )

        assert model.placement_requirement(NOTIONAL, LEVERAGE, TAKER_FEE_RATE) == 0.0

    def test_exchange_with_no_markup_holds_the_margin_and_the_fee(self):
        model = fee_model_for("exchange", _NO_RESERVE)

        required = model.placement_requirement(NOTIONAL, LEVERAGE, TAKER_FEE_RATE)

        assert required == pytest.approx(MARGIN + FEE)

    def test_the_margin_markup_scales_the_margin_alone(self):
        model = fee_model_for(
            "exchange",
            PlacementReserve(margin_markup=0.5, fee_markup=0.0, notional_reserve=0.0),
        )

        required = model.placement_requirement(NOTIONAL, LEVERAGE, TAKER_FEE_RATE)

        assert required == pytest.approx(MARGIN * 1.5 + FEE)

    def test_the_fee_markup_scales_the_fee_alone(self):
        model = fee_model_for(
            "exchange",
            PlacementReserve(margin_markup=0.0, fee_markup=0.5, notional_reserve=0.0),
        )

        required = model.placement_requirement(NOTIONAL, LEVERAGE, TAKER_FEE_RATE)

        assert required == pytest.approx(MARGIN + FEE * 1.5)

    def test_the_notional_reserve_is_a_flat_share_of_the_notional(self):
        model = fee_model_for(
            "exchange",
            PlacementReserve(margin_markup=0.0, fee_markup=0.0, notional_reserve=0.02),
        )

        required = model.placement_requirement(NOTIONAL, LEVERAGE, TAKER_FEE_RATE)

        assert required == pytest.approx(MARGIN + FEE + NOTIONAL * 0.02)


class TestPlacementReserveRate:
    """A sizing sets aside what a whole-balance entry needs beyond its margin
    at leverage 1, so an entry sized from a whole balance at any leverage is
    one the venue accepts.
    """

    def test_cost_sets_nothing_aside(self):
        model = fee_model_for("cost", PlacementReserve())

        assert model.placement_reserve_rate(TAKER_FEE_RATE) == 0.0

    def test_exchange_sets_aside_the_leverage_one_requirement_beyond_the_margin(
        self,
    ):
        model = fee_model_for("exchange", PlacementReserve())

        rate = model.placement_reserve_rate(TAKER_FEE_RATE)

        assert rate == pytest.approx(
            model.placement_requirement(1.0, 1.0, TAKER_FEE_RATE) - 1.0
        )

    def test_exchange_with_no_markup_sets_aside_the_taker_fee(self):
        model = fee_model_for("exchange", _NO_RESERVE)

        assert model.placement_reserve_rate(TAKER_FEE_RATE) == TAKER_FEE_RATE


class TestPlacementRequirementRate:
    def test_cost_locks_the_margin_alone(self):
        model = fee_model_for("cost", PlacementReserve())

        assert model.placement_requirement_rate(LEVERAGE, TAKER_FEE_RATE) == (
            1 / LEVERAGE
        )

    def test_exchange_locks_the_requirement_per_unit_of_value(self):
        model = fee_model_for("exchange", PlacementReserve())

        rate = model.placement_requirement_rate(LEVERAGE, TAKER_FEE_RATE)

        assert rate == pytest.approx(
            model.placement_requirement(1.0, LEVERAGE, TAKER_FEE_RATE)
        )


class TestClosingLegFee:
    def test_cost_charges_the_whole_order_on_the_closing_leg(self):
        model = fee_model_for("cost", _NO_RESERVE)

        assert model.closing_leg_fee(ORDER_FEE, CLOSE_STEPS, REQUESTED_STEPS) == 30.0

    def test_exchange_splits_the_fee_in_the_ratio_of_the_steps(self):
        model = fee_model_for("exchange", _NO_RESERVE)

        assert model.closing_leg_fee(ORDER_FEE, CLOSE_STEPS, REQUESTED_STEPS) == 20.0


class TestOrderSteps:
    """A quantity a strategy read back from a position is a whole number of
    steps already, so cutting down never takes a step off it.
    """

    def test_cost_rounds_to_the_nearest_step(self):
        model = fee_model_for("cost", _NO_RESERVE)

        assert model.order_steps(BETWEEN_TWO_STEPS) == 12345679

    def test_exchange_cuts_down_to_the_step(self):
        model = fee_model_for("exchange", _NO_RESERVE)

        assert model.order_steps(BETWEEN_TWO_STEPS) == 12345678

    def test_exchange_keeps_a_reported_quantity_whole(self):
        model = fee_model_for("exchange", _NO_RESERVE)

        assert model.order_steps(REPORTED_QUANTITY) == 12345679
