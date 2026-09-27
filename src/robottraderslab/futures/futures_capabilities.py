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
    """

    reserves_margin_on_pending_orders: bool = False
