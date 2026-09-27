from dataclasses import dataclass

from robottraderslab._core import FeeMode, PlacementReserve, StepCount, to_step_count

from .quantities import to_floored_step_count

type FeeModel = CostFeeModel | ExchangeFeeModel


def fee_model_for(fee_mode: FeeMode, placement_reserve: PlacementReserve) -> FeeModel:
    """The one place a mode name becomes a convention, so a run and a test
    building the same mode charge the same way.
    """
    if fee_mode == "exchange":
        return ExchangeFeeModel(placement_reserve=placement_reserve)
    return CostFeeModel()


@dataclass(frozen=True, slots=True)
class FeeRates:
    maker: float
    taker: float


@dataclass(frozen=True, slots=True)
class CostFeeModel:
    """Committing a whole balance buys a position worth that balance."""

    def closing_leg_fee(
        self, order_fee: float, close_steps: StepCount, requested_steps: StepCount
    ) -> float:
        """The leg that flips the position opens free of any further charge."""
        return order_fee

    def entry_locked_delta(self, required_margin: float, fee: float) -> float:
        """The fee is paid out of the margin being locked, so the available
        balance falls by the requested margin alone.
        """
        return required_margin - fee

    def entry_steps(self, requested_steps: StepCount, fee_rate: float) -> StepCount:
        return round(requested_steps * (1 - fee_rate))

    def order_steps(self, quantity: float) -> StepCount:
        return to_step_count(quantity)

    def placement_requirement(
        self, notional: float, leverage: float, taker_fee_rate: float
    ) -> float:
        """The fee comes out of the position, so nothing is held back beyond
        the margin the fill locks.
        """
        return 0.0

    def placement_requirement_rate(
        self, leverage: float, taker_fee_rate: float
    ) -> float:
        """The fee comes out of the position, so the venue locks the margin alone."""
        return 1 / leverage

    def placement_reserve_rate(self, taker_fee_rate: float) -> float:
        """The fee comes out of the position, so an order sized from a whole
        balance is one that balance can carry.
        """
        return 0.0


@dataclass(frozen=True, slots=True)
class ExchangeFeeModel:
    """Committing a whole balance leaves the fee unpaid, so a sizing under this
    convention holds it back.
    """

    placement_reserve: PlacementReserve

    def closing_leg_fee(
        self, order_fee: float, close_steps: StepCount, requested_steps: StepCount
    ) -> float:
        return order_fee * close_steps / requested_steps

    def entry_locked_delta(self, required_margin: float, fee: float) -> float:
        return required_margin

    def entry_steps(self, requested_steps: StepCount, fee_rate: float) -> StepCount:
        return requested_steps

    def order_steps(self, quantity: float) -> StepCount:
        """An order sized to the last unit of a balance fits only if nothing
        rounds it up.
        """
        return to_floored_step_count(quantity)

    def placement_requirement(
        self, notional: float, leverage: float, taker_fee_rate: float
    ) -> float:
        return notional * self.placement_reserve.requirement_rate(
            leverage, taker_fee_rate
        )

    def placement_requirement_rate(
        self, leverage: float, taker_fee_rate: float
    ) -> float:
        return self.placement_reserve.requirement_rate(leverage, taker_fee_rate)

    def placement_reserve_rate(self, taker_fee_rate: float) -> float:
        return self.placement_reserve.rate(taker_fee_rate)
