from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from robottraderslab._core import (
    StepCount,
    Symbol,
    calculate_pnl_long,
    calculate_pnl_short,
    to_quantity,
)
from robottraderslab.exchanges import PositionSide, PositionSnapshot


@dataclass(slots=True)
class SimulatedPosition:
    """Fills from several profiles accumulate and cancel exactly, leaving no residue."""

    symbol: Symbol
    side: PositionSide
    leverage: float
    taker_fee_rate: float
    entry_time: datetime
    quantity_steps: StepCount = field(default=0, init=False)
    average_entry_price: float = field(default=0.0, init=False)
    liquidation_price: float = field(init=False)
    locked_margin: float = field(default=0.0, init=False)
    group_id: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._update_liquidation_price()

    def add_to_position(self, steps: StepCount, price: float) -> None:
        """`steps` is at least one: opening a position with none divides zero by
        zero in the weighted average.
        """
        self.average_entry_price = _calculate_average_entry_price(
            current_steps=self.quantity_steps,
            current_avg_price=self.average_entry_price,
            added_steps=steps,
            new_price=price,
        )

        self.quantity_steps += steps
        self._update_liquidation_price()

    def get_realised_PnL(self, quantity: float, exit_price: float) -> float:
        """Profit in the symbol's quote currency, which the caller converts."""
        return _CALCULATE_PNL[self.side](self.average_entry_price, exit_price, quantity)

    def get_unrealised_PnL(self, current_price: float) -> float:
        if self.quantity_steps == 0:
            return 0.0

        return self.get_realised_PnL(to_quantity(self.quantity_steps), current_price)

    def lock_margin(self, amount: float) -> None:
        """Add to the margin backing this position, in margin currency."""
        self.locked_margin += amount

    def margin_backing(self, steps: StepCount) -> float:
        """The share of the locked margin a close of `steps` releases."""
        return self.locked_margin * (steps / self.quantity_steps)

    def reduce_position(self, steps: StepCount) -> None:
        self.quantity_steps -= steps

    def release_margin(self, steps: StepCount) -> float:
        """Must be called before `reduce_position`, so the proportion is
        computed against the quantity still open.
        """
        released = self.margin_backing(steps)
        self.locked_margin -= released
        return released

    def snapshot(self) -> PositionSnapshot:
        return PositionSnapshot(
            symbol=self.symbol,
            side=self.side,
            quantity=to_quantity(self.quantity_steps),
            average_entry_price=self.average_entry_price,
            entry_time=self.entry_time,
            leverage=self.leverage,
            liquidation_price=self.liquidation_price,
        )

    def _update_liquidation_price(self) -> None:
        self.liquidation_price = _CALCULATE_LIQUIDATION_PRICE[self.side](
            self.average_entry_price, self.leverage
        )


def _calculate_average_entry_price(
    current_steps: StepCount,
    current_avg_price: float,
    added_steps: StepCount,
    new_price: float,
) -> float:
    """A weighted average is unchanged when both weights scale by the same
    factor, so the step counts serve as the weights.
    """
    return (current_steps * current_avg_price + added_steps * new_price) / (
        current_steps + added_steps
    )


def _calculate_long_liquidation_price(
    entry_price: float, leverage: float, realised_pnl: float = 0
) -> float:
    """Returns 0.0 for spot trading, where leverage is always 1.0."""
    if leverage == 1.0:
        return 0.0
    return entry_price * (1 - 1 / leverage) - realised_pnl / (leverage - 1)


def _calculate_short_liquidation_price(
    entry_price: float, leverage: float, realised_pnl: float = 0
) -> float:
    """Returns twice the entry price for spot trading, where leverage is
    always 1.0.
    """
    if leverage == 1.0:
        return entry_price * 2
    return entry_price * (1 + 1 / leverage) + realised_pnl / (leverage - 1)


_CALCULATE_LIQUIDATION_PRICE: dict[PositionSide, Callable[[float, float], float]] = {
    PositionSide.LONG: _calculate_long_liquidation_price,
    PositionSide.SHORT: _calculate_short_liquidation_price,
}

_CALCULATE_PNL: dict[PositionSide, Callable[[float, float, float], float]] = {
    PositionSide.LONG: calculate_pnl_long,
    PositionSide.SHORT: calculate_pnl_short,
}
