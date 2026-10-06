import asyncio
import logging
from datetime import datetime

from robottraderslab._core import (
    AccountRequirement,
    AccountRequirements,
    Execution,
    OrderFill,
    OrderPlacement,
    OrderType,
    Symbol,
    settle_booked_fill,
)
from robottraderslab.exceptions import ExchangeRecoverableError

logger = logging.getLogger(__name__)


class BookedFills:
    """The fills a cycle's own actions produced, held until the account's
    executions say what each did to the position.

    A market order fills as the venue accepts it, so the read of the account's
    positions and executions starts at that acceptance; a fill of any other
    kind starts it when the fill is reported.
    """

    def __init__(
        self,
        accounts: AccountRequirements,
        since: datetime,
        *,
        settling: bool,
        max_attempts: int,
        base_delay: float,
    ) -> None:
        """Initialise the held fills.

        Args:
            settling: Whether a fill subscriber is configured; with none, no
                read is made.
            max_attempts: Total retry attempts for transient errors, per read.
            base_delay: Base retry delay in seconds.
        """
        self._accounts = accounts
        self._since = since
        self._settling = settling
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._fills: list[OrderFill] = []
        self._reads: dict[str, asyncio.Task[list[Execution]]] = {}

    def cancel(self) -> None:
        """Stop the reads still running, for a cycle that ended before it
        settled its fills.
        """
        for read in self._reads.values():
            read.cancel()

    async def record_fill(self, order_fill: OrderFill) -> None:
        """Hold a fill the executor reported until the cycle's actions ran."""
        self._fills.append(order_fill)
        self._start_reads(order_fill.symbol)

    async def record_placement(self, placement: OrderPlacement) -> None:
        """Start the account's read only for an untriggered market order, the
        one placement that fills as the venue accepts it.
        """
        if placement.kind == OrderType.MARKET and placement.trigger_price is None:
            self._start_reads(placement.symbol)

    async def settled(self) -> list[OrderFill]:
        """A fill whose account could not be read keeps the effect it had."""
        reads = await asyncio.gather(*self._reads.values())
        by_order: dict[str, list[Execution]] = {}
        for read in reads:
            for execution in read:
                by_order.setdefault(execution.order_id, []).append(execution)
        return [
            settle_booked_fill(fill, by_order.get(fill.order_id, ()))
            for fill in self._fills
        ]

    def _start_reads(self, symbol: Symbol) -> None:
        if not self._settling:
            return
        for requirement in self._accounts._get_all():
            name = requirement.account.name
            if symbol in requirement.symbols and name not in self._reads:
                self._reads[name] = asyncio.create_task(self._read(requirement))

    async def _read(self, requirement: AccountRequirement) -> list[Execution]:
        try:
            return await requirement.account._attributed_executions(
                self._since,
                requirement.symbols,
                max_attempts=self._max_attempts,
                base_delay=self._base_delay,
            )
        except ExchangeRecoverableError as e:
            logger.warning(
                "Booked fill settlement of account '%s' failed: %s",
                requirement.account.name,
                e,
            )
            return []
        except Exception as e:
            logger.error(
                "Internal error settling booked fills of account '%s': %s",
                requirement.account.name,
                e,
            )
            return []
