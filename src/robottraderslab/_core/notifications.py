import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterable
from typing import Protocol, runtime_checkable

from .exceptions import ExchangeCriticalError, StrategyCriticalError
from .order import OrderFill, OrderPlacement

logger = logging.getLogger(__name__)

type OnOrderFilled = Callable[[OrderFill], Awaitable[None]]
type OnOrderPlaced = Callable[[OrderPlacement], Awaitable[None]]


@runtime_checkable
class PlacementNotifier(Protocol):
    """Subscribing to placements requires implementing this, so a notifier
    that only speaks about fills is subscribed to fills alone.
    """

    async def notify_placement(self, placement: OrderPlacement) -> None:
        """Report an order the venue accepted."""
        ...


class Notifications:
    """Collects what a cycle did and notifies its subscribers once it has run.

    A fill reaches its subscribers whatever told the engine about it, so how
    one was discovered stays behind this. Holding events until the cycle ends
    keeps one part of a cycle from notifying before a later part has booked.
    """

    def __init__(
        self,
        on_fill: Iterable[OnOrderFilled] = (),
        on_placement: Iterable[OnOrderPlaced] = (),
    ) -> None:
        """Take the subscribers of each occasion.

        Args:
            on_fill: Notified once for each fill of the cycle.
            on_placement: Notified once for each order the venue accepted.
        """
        self._subscribers = list(on_fill)
        self._placement_subscribers = list(on_placement)
        self._fills: list[OrderFill] = []
        self._placements: list[OrderPlacement] = []

    async def record_fill(self, order: OrderFill) -> None:
        """Hold a fill until the cycle ends."""
        self._fills.append(order)

    async def record_placement(self, placement: OrderPlacement) -> None:
        """Hold an accepted order until the cycle ends."""
        self._placements.append(placement)

    async def flush(self) -> None:
        """Report everything recorded since construction to every subscriber.

        Each subscriber receives its events in the order they were recorded,
        and waits on no other subscriber to receive them. Trading never
        depends on being able to report what it did, so whatever a subscriber
        raises is logged and the events after it still reach that subscriber.
        """
        streams = [
            _forward_in_order(subscriber, self._fills)
            for subscriber in self._subscribers
        ] + [
            _forward_in_order(subscriber, self._placements)
            for subscriber in self._placement_subscribers
        ]
        if not streams:
            return

        await asyncio.gather(*streams)


async def _forward_in_order[T](
    subscriber: Callable[[T], Awaitable[None]], events: list[T]
) -> None:
    for event in events:
        try:
            await subscriber(event)
        except (Exception, ExchangeCriticalError, StrategyCriticalError) as e:
            logger.error("Notifier failed to report %s: %s", event, e)
