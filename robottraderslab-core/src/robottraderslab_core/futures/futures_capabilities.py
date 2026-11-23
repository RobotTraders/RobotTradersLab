from dataclasses import dataclass

from robottraderslab_core._internal import TriggerType


@dataclass(frozen=True, slots=True)
class FuturesCapabilities:
    """Traits a venue declares about itself, read from an account's
    `capabilities`; each defaults to the common case and a venue overrides
    what differs.

    A venue without `labels_resting_triggers` lists every resting trigger of a
    symbol as the bot's own, since the engine tells a trigger this bot placed
    from another's among the open orders by the client order id the caller set
    on it.

    Attributes:
        reserves_margin_on_pending_orders: Whether the venue locks margin for
            an order while it rests, so a strategy keeping many orders on the
            book fills only the ones the free margin covers.
        modifies_resting_orders_in_place: Whether the venue moves a resting
            limit order to a new price and size under the same order id, with
            its attached brackets and label kept; a venue without it serves
            the move as a cancel and a placement.
        labels_trigger_fills: Whether the fill of an order a trigger produced
            carries the client order id the caller set on the trigger; a
            strategy reading its fills back by that id needs it.
        places_trailing_stop_orders: Whether the venue places a trailing stop on a
            position and trails it behind the price on its own; a venue
            without it refuses `place_trailing_stop_order`,
            `update_position_trailing_stop` and
            `update_position_trailing_stop_ratio` before any request.
        min_trailing_share: The smallest share of the price the venue holds a
            trailing stop at; a connector refuses a share below it before any
            request.
        max_trailing_share: The largest share of the price the venue holds a
            trailing stop at, held to the same rule.
        trigger_types: The prices a trigger level may watch on the venue,
            `DEFAULT` among them; an order carrying another type is refused
            before it is sent.
        reserved_order_fields: The venue fields the connector sets from an
            order itself, which a strategy's or a configuration's params may
            not set; a configuration setting one stops the bot as its
            account loads. On a live bot with a working copy, an edited file
            setting one is not applied and the bot runs the copy.
    """

    reserves_margin_on_pending_orders: bool = False
    modifies_resting_orders_in_place: bool = True
    labels_trigger_fills: bool = True
    labels_resting_triggers: bool = True
    places_trailing_stop_orders: bool = False
    min_trailing_share: float = 0.0
    max_trailing_share: float = 1.0
    trigger_types: frozenset[TriggerType] = frozenset({TriggerType.DEFAULT})
    reserved_order_fields: frozenset[str] = frozenset()

    def holds_trailing_share(self, share: float) -> bool:
        """Report whether the venue holds a trailing stop at a share of the price.

        Args:
            share: The share of the price the trail stays behind the best price
                reached, 0.02 for 2%.
        """
        return (
            self.places_trailing_stop_orders
            and self.min_trailing_share <= share <= self.max_trailing_share
        )
