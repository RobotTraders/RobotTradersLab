from dataclasses import dataclass
from typing import Literal

type FeeMode = Literal["cost", "exchange"]

DEFAULT_FEE_MODE: FeeMode = "exchange"


@dataclass(frozen=True, slots=True)
class PlacementReserve:
    """What a venue holds back at placement beyond the position's margin.

    The defaults are Bitget's: the two markups it publishes, identical on
    every contract of its three futures listings, and the share of a market
    order's notional it was measured to hold back beyond them. A venue
    documenting no buffer sets each share to zero.
    """

    margin_markup: float = 0.01
    fee_markup: float = 0.005
    notional_reserve: float = 0.0025

    def rate(self, taker_fee_rate: float) -> float:
        """The share of an order's value that a whole-balance entry at leverage
        1 needs beyond its margin, which is the most an entry at any leverage
        needs.
        """
        return self.margin_markup + self._fee_and_notional(taker_fee_rate)

    def requirement_rate(self, leverage: float, taker_fee_rate: float) -> float:
        """What the venue holds per unit of an order's value at placement; it
        values the fee of any order at its taker rate.
        """
        return (1 + self.margin_markup) / leverage + self._fee_and_notional(
            taker_fee_rate
        )

    def _fee_and_notional(self, taker_fee_rate: float) -> float:
        return taker_fee_rate * (1 + self.fee_markup) + self.notional_reserve
