from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FuturesCapabilities:
    """Traits a venue declares about itself, read from an account's
    `capabilities`; each defaults to the common case and a venue overrides
    what differs.

    Attributes:
        reserves_margin_on_pending_orders: Whether the venue locks margin for
            an order while it rests, so a strategy keeping many orders on the
            book fills only the ones the free margin covers.
        modifies_resting_orders_in_place: Whether the venue moves a resting
            limit order to a new price and size under the same order id, with
            its attached brackets and label kept; a venue without it serves
            the move as a cancel and a placement.
    """

    reserves_margin_on_pending_orders: bool = False
    modifies_resting_orders_in_place: bool = True
