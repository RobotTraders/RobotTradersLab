import pytest

from robottraderslab import Symbol


def test_wrong_symbol():
    with pytest.raises(ValueError, match="Invalid symbol format: BTC"):
        Symbol.create("BTC")


def test_futures_symbol():
    symbol = Symbol.create("BTC/USDT:USDT")

    assert not symbol.is_spot
    assert symbol.base == "BTC"
    assert symbol.quote == "USDT"
    assert symbol.margin == "USDT"
    assert str(symbol) == "BTC/USDT:USDT"


def test_spot_symbol():
    symbol = Symbol.create("BTC/USDT")

    assert symbol.is_spot
    assert symbol.base == "BTC"
    assert symbol.quote == "USDT"
    assert symbol.margin is None
    assert str(symbol) == "BTC/USDT"


def test_already_parsed_symbol():
    parsed = Symbol.create("BTC/USDT")
    reparsed = Symbol.create(parsed)

    assert reparsed is parsed


@pytest.mark.parametrize(
    ("symbol", "settlement"),
    [
        ("BTC/USDT:USDT", "USDT"),
        ("BTC/USD:BTC", "BTC"),
        ("ETH/BTC", "BTC"),
    ],
    ids=[
        "derivative_settles_in_its_margin",
        "inverse_derivative_settles_in_the_coin_it_is_margined_in",
        "spot_settles_in_its_quote",
    ],
)
def test_settlement_currency(symbol, settlement):
    assert Symbol.create(symbol).settlement == settlement
