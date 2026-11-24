import re
from typing import Any, SupportsIndex

from pydantic_core import core_schema

from .currency import Currency

symbol_regex = re.compile(r"^(?P<base>\w+)/(?P<quote>\w+)(:(?P<settlement>\w+))?$")

SymbolStr = str


class Symbol(str):
    """A trading pair, like "BTC/USDT", parsed into its currencies.

    An optional third part names the margin currency of a derivative, like
    "BTC/USDT:USDT"; without it the symbol is spot. A string is parsed with
    `create`; the constructor takes the currencies already apart.
    """

    __slots__ = ("_base", "_quote", "_margin")

    _base: Currency
    _quote: Currency
    _margin: Currency | None

    @classmethod
    def create(cls, symbol: "str | Symbol") -> "Symbol":
        """Parse a symbol string, returning an already-parsed one unchanged.

        Args:
            symbol: "BASE/QUOTE" for spot, "BASE/QUOTE:MARGIN" for a
                derivative (e.g. "BTC/USDT", "BTC/USDT:USDT").

        Raises:
            ValueError: If the string matches neither form.
        """
        if isinstance(symbol, Symbol):
            return symbol

        matched = symbol_regex.match(symbol)
        if not matched:
            raise ValueError(
                f"Invalid symbol format: {symbol}. E.g. `BTC/USDT` or `BTC/USDT:USDT`"
            )

        base, quote, settlement = matched.group("base", "quote", "settlement")
        return cls(base=base, quote=quote, margin=settlement)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: Any
    ) -> core_schema.CoreSchema:
        """A string is parsed with `create` wherever a `Symbol` field is built."""
        return core_schema.no_info_after_validator_function(
            cls.create, core_schema.str_schema()
        )

    def __new__(
        cls, base: Currency, quote: Currency, margin: Currency | None = None
    ) -> "Symbol":
        value = f"{base}/{quote}" if margin is None else f"{base}/{quote}:{margin}"
        obj = str.__new__(cls, value)
        obj._base = base
        obj._quote = quote
        obj._margin = margin
        return obj

    def __str__(self) -> str:
        return super().__str__()

    def __repr__(self) -> str:
        return f"Symbol('{str(self)}')"

    def __reduce_ex__(self, protocol: SupportsIndex) -> str | tuple[Any, ...]:
        return (Symbol.create, (str(self),))

    def __reduce__(self) -> str | tuple[Any, ...]:
        return (Symbol.create, (str(self),))

    @property
    def base(self) -> Currency:
        """The currency being traded."""
        return self._base

    @property
    def quote(self) -> Currency:
        """The currency the base is priced in."""
        return self._quote

    @property
    def margin(self) -> Currency | None:
        """The margin currency of a derivative, None on spot."""
        return self._margin

    @property
    def is_spot(self) -> bool:
        """Whether the symbol carries no margin currency."""
        return self._margin is None

    @property
    def settlement(self) -> Currency:
        return self._margin or self._quote
