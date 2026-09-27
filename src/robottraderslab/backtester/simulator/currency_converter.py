from robottraderslab._core import Symbol
from robottraderslab.exchanges import Currency

type Prices = dict[Symbol, float]


class MissingConversionRateError(RuntimeError):
    """Raised when no market price is available to convert a currency."""


class CurrencyConverter:
    """Resolves conversion rates from any currency into the account currency.

    Rates come from the latest observed market prices. A currency can be
    registered explicitly against a pair whose price (inverted if flagged)
    converts it; an unregistered currency resolves once against a direct or
    inverted pair found in the prices and keeps that resolution.
    """

    def __init__(self, account_currency: Currency):
        """
        Args:
            account_currency: Currency every amount is converted into.
        """
        self._account_currency = account_currency
        self._pair_by_currency: dict[Currency, tuple[Symbol, bool]] = {}

    @property
    def account_currency(self) -> Currency:
        return self._account_currency

    def conversion_candidates(self, currency: Currency) -> list[tuple[Symbol, bool]]:
        """Return the pairs able to convert the currency, with their inversion flag."""
        return [
            (Symbol(currency, self._account_currency), False),
            (Symbol(self._account_currency, currency), True),
        ]

    def rate(self, currency: Currency, prices: Prices) -> float:
        """Return the multiplier converting the currency into the account currency.

        Args:
            currency: Currency of the amount to convert.
            prices: Latest known price per symbol.

        Raises:
            MissingConversionRateError: If no usable price is available.
        """
        if currency == self._account_currency:
            return 1.0

        registered = self._pair_by_currency.get(currency)
        if registered is not None:
            via, inverted = registered
            price = prices.get(via)
            if price is None:
                raise MissingConversionRateError(
                    f"No {via} price available yet to convert "
                    f"{currency} into {self._account_currency}"
                )
            return 1.0 / price if inverted else price

        for via, inverted in self.conversion_candidates(currency):
            price = prices.get(via)
            if price is not None:
                self._pair_by_currency[currency] = (via, inverted)
                return 1.0 / price if inverted else price

        raise MissingConversionRateError(
            f"No price available to convert {currency} into {self._account_currency}"
        )

    def register(self, currency: Currency, *, via: Symbol, inverted: bool) -> None:
        """Convert the currency using the pair's price.

        Args:
            currency: Currency the registration applies to.
            via: Pair whose price provides the rate.
            inverted: Whether the rate is the reciprocal of the pair's price.
        """
        self._pair_by_currency[currency] = (via, inverted)

    def registered_pair(self, currency: Currency) -> tuple[Symbol, bool] | None:
        """Return the registered (pair, inverted) for the currency, if any."""
        return self._pair_by_currency.get(currency)
