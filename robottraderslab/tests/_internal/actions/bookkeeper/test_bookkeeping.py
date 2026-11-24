import pytest

from robottraderslab import Symbol
from robottraderslab.strategies import BookKeeper

BTC_USDT = Symbol.create("BTC/USDT:USDT")


@pytest.fixture
def bookkeeper() -> BookKeeper:
    """A fresh BookKeeper."""
    return BookKeeper()


def test_add_books_the_order_a_builder_builds(bookkeeper, account):
    booked = bookkeeper.add(account.long_entry(BTC_USDT, 1.5).limit(99.0))

    assert bookkeeper.list_actions() == [booked]
    assert booked.quantity == 1.5
    assert booked.price == 99.0


def test_unsized_builder_handed_to_the_bookkeeper(bookkeeper, account):
    with pytest.raises(ValueError, match="The order has no size"):
        bookkeeper.add(account.long_entry(BTC_USDT))

    assert bookkeeper.list_actions() == []
