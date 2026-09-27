import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from pydantic_core import core_schema

from .account_snapshot import AccountSnapshot
from .currency import Currency
from .exceptions import StrategyCriticalError
from .symbol import Symbol

type RiskReference = Literal["equity", "total_balance", "available_balance"]

_RISK_REFERENCES: tuple[RiskReference, ...] = (
    "equity",
    "total_balance",
    "available_balance",
)


class SizingRule:
    """How an order is sized, chosen by the configuration key named after it.

    A rule states what the position is worth, which the leverage leaves
    unchanged, or what the venue locks for it, as `margin` does. A rule of
    the strategy's own subclasses this one, implements `quantity`, and joins
    the engine's rules through the strategy's `sizing_rules`, where a
    configuration names it exactly as it names an engine rule: a scalar
    builds the rule from its one argument, a table from its keywords.
    """

    __slots__ = ()

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: Any
    ) -> core_schema.CoreSchema:
        """A `sizing` field holds the rule the engine built from the key named
        after it.
        """
        return core_schema.is_instance_schema(cls)

    @classmethod
    def from_settings(
        cls,
        settings: Mapping[str, Any],
        rules: Mapping[str, type["SizingRule"]] = MappingProxyType({}),
    ) -> "SizingRule":
        """Build the rule the settings name among their keys.

        Args:
            settings: Configuration keys, among which exactly one names a rule;
                the others are left alone.
            rules: The strategy's own rules by the key naming each, beside the
                engine's.

        Raises:
            StrategyCriticalError: If no key or several keys name a rule, or
                the rule refuses the value under its key.
            TypeError: If one of `rules` takes the key of an engine rule.
        """
        registry = sizing_rules(rules)
        named = [key for key in settings if key in registry]
        if len(named) != 1:
            raise StrategyCriticalError(_naming_fault(named, registry))
        (key,) = named
        value = settings[key]
        build: Callable[..., SizingRule] = registry[key]
        try:
            if isinstance(value, Mapping):
                return build(**value)
            return build(value)
        except (TypeError, ValueError) as error:
            raise StrategyCriticalError(f"{key}: {error}") from error

    @property
    def reads_balances(self) -> bool:
        """True makes `requirements.account.add` declare the balances for the
        rules it is given as `sizing`.
        """
        return False

    @property
    def reads_equity(self) -> bool:
        """True makes `requirements.account.add` declare equity for the rules
        it is given as `sizing`.
        """
        return False

    @property
    def reads_margin_settings(self) -> bool:
        """True makes `requirements.account.add` declare the margin settings for
        the rules it is given as `sizing`.
        """
        return False

    def quantity(
        self,
        symbol: Symbol,
        price: float,
        account_snapshot: AccountSnapshot,
        *,
        placement_reserve_rate: float,
        placement_requirement_rate: Callable[[float], float],
        stop_loss_price: float | None,
    ) -> float:
        """Return the quantity to trade, in the base currency.

        Args:
            price: Price the order is expected to fill at.
            account_snapshot: The account's snapshot for this candle.
            placement_reserve_rate: Share of an order's value the venue holds
                back at placement beyond the margin, which a rule sizing from
                a balance sets aside so the venue accepts the order.
            placement_requirement_rate: Given a leverage, returns the share
                of an order's value the venue locks at placement, the margin
                included, which a rule sizing from what the trader puts in
                divides its amount by.
            stop_loss_price: The stop-loss attached to the order, None when
                none is.

        Raises:
            StrategyCriticalError: If the rule cannot size the order from what
                it is given.
        """
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class AvailableBalanceRatio(SizingRule):
    """The position is worth `ratio` of the available balance, less the
    placement reserve.

    Attributes:
        ratio: Share of the available balance, within (0, 1].
    """

    ratio: float

    def __post_init__(self) -> None:
        """
        Raises:
            ValueError: If the ratio is outside (0, 1].
        """
        validate_share(self.ratio)

    @property
    def reads_balances(self) -> bool:
        return True

    def quantity(
        self,
        symbol: Symbol,
        price: float,
        account_snapshot: AccountSnapshot,
        *,
        placement_reserve_rate: float,
        placement_requirement_rate: Callable[[float], float],
        stop_loss_price: float | None,
    ) -> float:
        return quantity_from_available_balance(
            self.ratio, price, account_snapshot, symbol, placement_reserve_rate
        )


@dataclass(frozen=True, slots=True)
class EquityRatio(SizingRule):
    """The position is worth `ratio` of equity, capped by the available
    balance, less the placement reserve.

    Attributes:
        ratio: Share of equity, within (0, 1].
    """

    ratio: float

    def __post_init__(self) -> None:
        """
        Raises:
            ValueError: If the ratio is outside (0, 1].
        """
        validate_share(self.ratio)

    @property
    def reads_balances(self) -> bool:
        return True

    @property
    def reads_equity(self) -> bool:
        return True

    def quantity(
        self,
        symbol: Symbol,
        price: float,
        account_snapshot: AccountSnapshot,
        *,
        placement_reserve_rate: float,
        placement_requirement_rate: Callable[[float], float],
        stop_loss_price: float | None,
    ) -> float:
        return quantity_from_equity(
            self.ratio, price, account_snapshot, symbol, placement_reserve_rate
        )


@dataclass(frozen=True, slots=True)
class Margin(SizingRule):
    """`amount` of the margin currency is what the venue locks for the
    position: its margin at the symbol's leverage in the snapshot, plus the
    placement reserve. A leverage change booked on a candle sizes the next
    candle's orders.

    Attributes:
        amount: What the venue locks, in the symbol's margin currency,
            above zero.
    """

    amount: float

    def __post_init__(self) -> None:
        """
        Raises:
            ValueError: If the amount is not positive.
        """
        validate_positive("margin", self.amount)

    @property
    def reads_balances(self) -> bool:
        return True

    @property
    def reads_margin_settings(self) -> bool:
        return True

    def quantity(
        self,
        symbol: Symbol,
        price: float,
        account_snapshot: AccountSnapshot,
        *,
        placement_reserve_rate: float,
        placement_requirement_rate: Callable[[float], float],
        stop_loss_price: float | None,
    ) -> float:
        """
        Raises:
            StrategyCriticalError: If the venue reports no leverage on the
                symbol.
        """
        settings = account_snapshot.margin_settings(symbol)
        if settings.leverage is None:
            raise StrategyCriticalError(
                f"margin sizes from the leverage set on {symbol}, and the venue "
                f"reports none in {settings.margin_mode} margin mode: set a "
                "leverage on the symbol before sizing by margin."
            )
        return _to_quantity(
            self.amount / placement_requirement_rate(settings.leverage),
            price,
            account_snapshot,
            symbol,
        )


@dataclass(frozen=True, slots=True)
class Notional(SizingRule):
    """The position is worth `amount` of the margin currency, less the
    placement reserve.

    Attributes:
        amount: The position's worth, in the symbol's margin currency, above
            zero.
    """

    amount: float

    def __post_init__(self) -> None:
        """
        Raises:
            ValueError: If the amount is not positive.
        """
        validate_positive("notional", self.amount)

    @property
    def reads_balances(self) -> bool:
        return True

    def quantity(
        self,
        symbol: Symbol,
        price: float,
        account_snapshot: AccountSnapshot,
        *,
        placement_reserve_rate: float,
        placement_requirement_rate: Callable[[float], float],
        stop_loss_price: float | None,
    ) -> float:
        return _to_quantity(
            _after_placement_reserve(self.amount, placement_reserve_rate),
            price,
            account_snapshot,
            symbol,
        )


@dataclass(frozen=True, slots=True)
class Quantity(SizingRule):
    """The order trades `amount` of the base currency.

    Attributes:
        amount: The quantity, in the base currency, above zero.
    """

    amount: float

    def __post_init__(self) -> None:
        """
        Raises:
            ValueError: If the amount is not positive.
        """
        validate_positive("quantity", self.amount)

    def quantity(
        self,
        symbol: Symbol,
        price: float,
        account_snapshot: AccountSnapshot,
        *,
        placement_reserve_rate: float,
        placement_requirement_rate: Callable[[float], float],
        stop_loss_price: float | None,
    ) -> float:
        return self.amount


@dataclass(frozen=True, slots=True)
class RiskRatio(SizingRule):
    """The stop-loss attached to the order costs `ratio` of the reference `of`
    names, which sizes the order from the distance between the price and the
    stop.

    Attributes:
        ratio: Share of the reference the stop-loss may cost, within (0, 1].
        of: The reference: `"equity"`, `"total_balance"` or
            `"available_balance"`.
    """

    ratio: float
    of: RiskReference = "equity"

    def __post_init__(self) -> None:
        """
        Raises:
            ValueError: If the ratio is outside (0, 1] or `of` names no reference.
        """
        validate_share(self.ratio)
        if self.of not in _RISK_REFERENCES:
            raise ValueError(
                f"`of` must be one of {', '.join(_RISK_REFERENCES)}; received {self.of!r}"
            )

    @property
    def reads_balances(self) -> bool:
        return True

    @property
    def reads_equity(self) -> bool:
        return self.of == "equity"

    def quantity(
        self,
        symbol: Symbol,
        price: float,
        account_snapshot: AccountSnapshot,
        *,
        placement_reserve_rate: float,
        placement_requirement_rate: Callable[[float], float],
        stop_loss_price: float | None,
    ) -> float:
        """
        Raises:
            StrategyCriticalError: If no stop-loss is attached to the order.
        """
        if stop_loss_price is None:
            raise StrategyCriticalError(
                f"risk_ratio sizes from a stop-loss, and none is given on {symbol}: "
                "an entry takes one with stop_loss() before it is built, and a "
                "position target has no stop-loss to size from."
            )
        return quantity_at_risk(
            self.ratio,
            price,
            stop_loss_price,
            _risk_reference(self.of, account_snapshot, margin_currency(symbol)),
            account_snapshot,
            symbol,
        )


@dataclass(frozen=True, slots=True)
class TotalBalanceRatio(SizingRule):
    """The position is worth `ratio` of the total balance, less the placement
    reserve.

    Attributes:
        ratio: Share of the total balance, within (0, 1].
    """

    ratio: float

    def __post_init__(self) -> None:
        """
        Raises:
            ValueError: If the ratio is outside (0, 1].
        """
        validate_share(self.ratio)

    @property
    def reads_balances(self) -> bool:
        return True

    def quantity(
        self,
        symbol: Symbol,
        price: float,
        account_snapshot: AccountSnapshot,
        *,
        placement_reserve_rate: float,
        placement_requirement_rate: Callable[[float], float],
        stop_loss_price: float | None,
    ) -> float:
        return quantity_from_total_balance(
            self.ratio, price, account_snapshot, symbol, placement_reserve_rate
        )


_ENGINE_SIZING_RULES: Mapping[str, type[SizingRule]] = MappingProxyType(
    {
        "available_balance_ratio": AvailableBalanceRatio,
        "equity_ratio": EquityRatio,
        "margin": Margin,
        "notional": Notional,
        "quantity": Quantity,
        "risk_ratio": RiskRatio,
        "total_balance_ratio": TotalBalanceRatio,
    }
)


def sizing_rules(
    rules: Mapping[str, type[SizingRule]],
) -> Mapping[str, type[SizingRule]]:
    """
    Raises:
        TypeError: If one of `rules` takes the key of an engine rule.
    """
    taken = sorted(set(rules) & set(_ENGINE_SIZING_RULES))
    if taken:
        raise TypeError(
            f"sizing rule {', '.join(taken)} is the engine's; register yours "
            "under a name of its own"
        )
    return {**_ENGINE_SIZING_RULES, **rules}


def margin_currency(symbol: Symbol) -> Currency:
    """
    Raises:
        ValueError: If the symbol carries no margin currency.
    """
    if symbol.margin is None:
        raise ValueError(
            f"A sizing rule sizes from the margin currency of a futures symbol, "
            f"and '{symbol}' carries none"
        )
    return symbol.margin


def quantity_at_risk(
    risk_ratio: float,
    entry_price: float,
    stop_loss_price: float,
    reference: float,
    account_snapshot: AccountSnapshot,
    symbol: Symbol,
) -> float:
    """The stop-loss of the returned quantity costs `risk_ratio` of the
    reference.
    """
    return _to_quantity(
        reference * risk_ratio,
        abs(entry_price - stop_loss_price),
        account_snapshot,
        symbol,
    )


def quantity_from_available_balance(
    available_balance_ratio: float,
    current_price: float,
    account_snapshot: AccountSnapshot,
    symbol: Symbol,
    placement_reserve_rate: float,
) -> float:
    """An entry and a target sized from the same share come out the same
    quantity.
    """
    balance = account_snapshot.balance(margin_currency(symbol))
    return _to_quantity(
        _after_placement_reserve(
            balance.available * available_balance_ratio, placement_reserve_rate
        ),
        current_price,
        account_snapshot,
        symbol,
    )


def quantity_from_equity(
    equity_ratio: float,
    current_price: float,
    account_snapshot: AccountSnapshot,
    symbol: Symbol,
    placement_reserve_rate: float,
) -> float:
    """An entry and a target sized from the same share come out the same
    quantity.
    """
    currency = margin_currency(symbol)
    balance = account_snapshot.balance(currency)
    requested = min(account_snapshot.equity(currency) * equity_ratio, balance.available)
    return _to_quantity(
        _after_placement_reserve(requested, placement_reserve_rate),
        current_price,
        account_snapshot,
        symbol,
    )


def quantity_from_total_balance(
    total_balance_ratio: float,
    current_price: float,
    account_snapshot: AccountSnapshot,
    symbol: Symbol,
    placement_reserve_rate: float,
) -> float:
    """An entry and a target sized from the same share come out the same
    quantity.
    """
    balance = account_snapshot.balance(margin_currency(symbol))
    return _to_quantity(
        _after_placement_reserve(
            balance.total * total_balance_ratio, placement_reserve_rate
        ),
        current_price,
        account_snapshot,
        symbol,
    )


def validate_positive(name: str, value: float) -> None:
    """
    Raises:
        ValueError: If the value is not positive or is NaN.
    """
    if value <= 0 or math.isnan(value):
        raise ValueError(f"`{name}` must be greater than 0; received {value}")


def validate_ratio(name: str, value: float) -> None:
    """A share of a balance or an equity is a number within [0, 1].

    Raises:
        ValueError: If the value is outside [0, 1] or is NaN.
    """
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"`{name}` must be between 0.0 and 1.0; received {value}")


def validate_share(value: float) -> None:
    """A rule sizing from a share sizes something, and never more than the
    whole.

    Raises:
        ValueError: If the value is outside (0, 1] or is NaN.
    """
    if not (0.0 < value <= 1.0):
        raise ValueError(f"`ratio` must be above 0.0 and at most 1.0; received {value}")


def _naming_fault(named: list[str], registry: Mapping[str, type[SizingRule]]) -> str:
    if named:
        return f"names several sizing rules, {', '.join(named)}; keep one"
    return f"names no sizing rule; add one of {', '.join(sorted(registry))}"


def _to_quantity(
    amount: float,
    denominator: float,
    account_snapshot: AccountSnapshot,
    symbol: Symbol,
) -> float:
    return amount / (denominator * account_snapshot.quote_conversion_rate(symbol))


def _after_placement_reserve(amount: float, placement_reserve_rate: float) -> float:
    return amount / (1 + placement_reserve_rate)


def _risk_reference(
    of: RiskReference, account_snapshot: AccountSnapshot, currency: Currency
) -> float:
    if of == "equity":
        return account_snapshot.equity(currency)
    balance = account_snapshot.balance(currency)
    return balance.total if of == "total_balance" else balance.available
