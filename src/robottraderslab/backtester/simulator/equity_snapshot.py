import copy
from dataclasses import dataclass
from datetime import datetime

from robottraderslab._core import Symbol
from robottraderslab.exchanges import Balance, Currency, MarketType

from .currency_converter import CurrencyConverter, Prices
from .simulated_position import SimulatedPosition


@dataclass(frozen=True, slots=True)
class EquitySnapshot:
    """Single point-in-time snapshot of account state."""

    timestamp: datetime
    market_type: MarketType
    balances: dict[Currency, Balance]
    positions: dict[Symbol, SimulatedPosition]
    prices: Prices
    converter: CurrencyConverter | None = None

    def frozen_copy(self) -> "EquitySnapshot":
        """Return a defensively-copied snapshot safe for long-term storage."""
        return EquitySnapshot(
            timestamp=self.timestamp,
            market_type=self.market_type,
            balances=self.balances.copy(),
            positions=copy.deepcopy(self.positions),
            prices=self.prices.copy(),
            converter=self.converter,
        )

    def get_equity(self, currency: Currency) -> float:
        if self.market_type == "futures":
            return self._calculate_futures_equity_curve(currency)
        else:
            return self._calculate_spot_equity_curve(currency)

    def _calculate_futures_equity_curve(self, margin_currency: Currency) -> float:
        converter = self._converter_for(margin_currency)
        default = Balance(locked=0.0, total=0.0)
        equity = self.balances.get(margin_currency, default).total
        for symbol, position in self.positions.items():
            current_price = self.prices[symbol]
            unrealised_pnl = position.get_unrealised_PnL(current_price)
            equity += unrealised_pnl * converter.rate(symbol.quote, self.prices)
        return equity

    def _calculate_spot_equity_curve(self, quote_currency: Currency) -> float:
        converter = self._converter_for(quote_currency)
        default = Balance(locked=0.0, total=0.0)
        equity = self.balances.get(quote_currency, default).total
        for currency, balance in self.balances.items():
            if currency != quote_currency:
                equity += balance.total * converter.rate(currency, self.prices)
        return equity

    def _converter_for(self, reference: Currency) -> CurrencyConverter:
        if self.converter is not None and self.converter.account_currency == reference:
            return self.converter
        return CurrencyConverter(reference)
