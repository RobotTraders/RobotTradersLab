import pytest

from robottraderslab import Symbol
from robottraderslab.backtester.simulator import (
    CurrencyConverter,
    MissingConversionRateError,
)

CAD_USD = Symbol("CAD", "USD")
USD_CAD = Symbol("USD", "CAD")
BTC_ETH_PERP = Symbol.create("BTC/ETH:BTC")


@pytest.fixture
def converter() -> CurrencyConverter:
    return CurrencyConverter("USD")


class TestRate:
    def test_account_currency(self, converter):
        assert converter.rate("USD", {}) == 1.0

    def test_direct_pair_in_prices(self, converter):
        assert converter.rate("CAD", {CAD_USD: 0.8}) == 0.8

    def test_inverted_pair_in_prices(self, converter):
        assert converter.rate("CAD", {USD_CAD: 1.25}) == 0.8

    def test_registered_pair_overrides_price_fallback(self, converter):
        converter.register("CAD", via=USD_CAD, inverted=True)

        rate = converter.rate("CAD", {USD_CAD: 1.25, CAD_USD: 999.0})

        assert rate == 0.8

    def test_registered_via_base_margined_pair(self):
        btc_converter = CurrencyConverter("BTC")
        btc_converter.register("ETH", via=BTC_ETH_PERP, inverted=True)

        assert btc_converter.rate("ETH", {BTC_ETH_PERP: 20.0}) == 0.05

    def test_no_price_available(self, converter):
        with pytest.raises(MissingConversionRateError, match="convert CAD into USD"):
            converter.rate("CAD", {})

    def test_registered_pair_without_price(self, converter):
        converter.register("CAD", via=USD_CAD, inverted=True)

        with pytest.raises(MissingConversionRateError, match="USD/CAD"):
            converter.rate("CAD", {CAD_USD: 0.8})


class TestRegistration:
    def test_unregistered_currency(self, converter):
        assert converter.registered_pair("CAD") is None

    def test_registered_currency(self, converter):
        converter.register("CAD", via=USD_CAD, inverted=True)

        assert converter.registered_pair("CAD") == (USD_CAD, True)

    def test_conversion_candidates(self, converter):
        assert converter.conversion_candidates("CAD") == [
            (CAD_USD, False),
            (USD_CAD, True),
        ]
