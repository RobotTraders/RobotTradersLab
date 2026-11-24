from abc import ABC, abstractmethod
from enum import StrEnum

from .trade_record import TradeRecord


class AccountingMethodChoice(StrEnum):
    """Which open entry an exit closes first.

    AVERAGE names the average-cost method still to be written; the factory
    refuses it until an implementation exists.
    """

    FIFO = "fifo"
    LIFO = "lifo"
    AVERAGE = "average"


class AccountingMethod(ABC):
    """An exit consumes a position's open entries in the order the method dictates."""

    @abstractmethod
    def handle_entry(
        self, trades: list[TradeRecord], trade: TradeRecord
    ) -> list[TradeRecord]:
        """Place a new entry among the position's open entries.

        Args:
            trades: The position's open entries, in fill order.
            trade: The entry just filled.

        Returns:
            The open entries with the new one placed where the method reads it.
        """

    @abstractmethod
    def order_trades_for_exit(self, trades: list[TradeRecord]) -> list[TradeRecord]:
        """Order the open entries for an exit to consume.

        Args:
            trades: The position's open entries, in fill order.

        Returns:
            The same entries, in the order the exit closes them.
        """


class FIFOMethod(AccountingMethod):
    """The oldest entry closes first."""

    def handle_entry(
        self, trades: list[TradeRecord], trade: TradeRecord
    ) -> list[TradeRecord]:
        trades.append(trade)
        return trades

    def order_trades_for_exit(self, trades: list[TradeRecord]) -> list[TradeRecord]:
        return trades


class LIFOMethod(AccountingMethod):
    """The newest entry closes first."""

    def handle_entry(
        self, trades: list[TradeRecord], trade: TradeRecord
    ) -> list[TradeRecord]:
        trades.append(trade)
        return trades

    def order_trades_for_exit(self, trades: list[TradeRecord]) -> list[TradeRecord]:
        return list(reversed(trades))


def get_accounting_method(method_choice: AccountingMethodChoice) -> AccountingMethod:
    """AVERAGE has no implementation yet, so choosing it is refused.

    Raises:
        ValueError: If the choice names a method with no implementation.
    """
    method_map: dict[AccountingMethodChoice, type[AccountingMethod]] = {
        AccountingMethodChoice.FIFO: FIFOMethod,
        AccountingMethodChoice.LIFO: LIFOMethod,
    }

    method_class = method_map.get(method_choice)
    if not method_class:
        raise ValueError(f"Unknown accounting method: {method_choice}")

    return method_class()
