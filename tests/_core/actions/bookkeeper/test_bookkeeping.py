from dataclasses import dataclass

import pytest

from robottraderslab import Symbol
from robottraderslab._core import BaseExchangeAction
from robottraderslab._core.actions.exceptions import (
    ActionAlreadyExistsError,
    ActionNotFoundError,
)
from robottraderslab.exceptions import StrategyCriticalError
from robottraderslab.strategies import BookKeeper

BTC_USDT = Symbol.create("BTC/USDT:USDT")


@dataclass
class MockExchangeAction(BaseExchangeAction):
    """Minimal exchange action for exercising the BookKeeper."""

    def __init__(self, custom_id: int | None = None):
        """Initialize with optional custom ID for testing duplicate IDs."""
        super().__init__(symbol="BTC/USDT")
        if custom_id:
            self.id = custom_id

    def execute(self) -> None:
        """Mock execute method for testing."""
        pass


@pytest.fixture
def bookkeeper() -> BookKeeper:
    """Fixture providing a fresh BookKeeper instance"""
    return BookKeeper()


@pytest.fixture
def action() -> MockExchangeAction:
    """Fixture providing a test action"""
    return MockExchangeAction()


def test_new_bookkeeper_is_empty():
    keeper = BookKeeper()
    assert keeper.list_actions() == []


def test_add_stores_and_returns_action(bookkeeper, action):
    stored_action = bookkeeper.add(action)
    assert stored_action is action
    assert action in bookkeeper.list_actions()


@pytest.mark.parametrize(
    "action_factory",
    [
        lambda original: original,  # Same action
        lambda original: MockExchangeAction(
            custom_id=original.id
        ),  # Different action, same ID
    ],
)
def test_add_rejects_duplicates(bookkeeper, action, action_factory):
    bookkeeper.add(action)
    duplicate_action = action_factory(action)
    with pytest.raises(ActionAlreadyExistsError):
        bookkeeper.add(duplicate_action)


def test_add_books_the_order_a_builder_builds(bookkeeper, account):
    booked = bookkeeper.add(account.long_entry(BTC_USDT, 1.5).limit(99.0))

    assert bookkeeper.list_actions() == [booked]
    assert booked.quantity == 1.5
    assert booked.price == 99.0


def test_unsized_builder_handed_to_the_bookkeeper(bookkeeper, account):
    with pytest.raises(StrategyCriticalError, match="The order has no size"):
        bookkeeper.add(account.long_entry(BTC_USDT))

    assert bookkeeper.list_actions() == []


def test_add_records_what_an_action_follows(bookkeeper, action):
    follower = MockExchangeAction()
    bookkeeper.add(action)
    bookkeeper.add(follower, after=action)
    assert bookkeeper.declared_waits() == {follower.id: (action.id,)}


def test_add_rejects_following_an_unbooked_action(bookkeeper, action):
    with pytest.raises(ActionNotFoundError) as exc:
        bookkeeper.add(MockExchangeAction(), after=action)
    assert str(action.id) in str(exc.value)


def test_list_actions_returns_all_stored_actions(bookkeeper, action):
    action2 = MockExchangeAction()
    bookkeeper.add(action)
    bookkeeper.add(action2)
    actions = bookkeeper.list_actions()
    assert len(actions) == 2
    assert action in actions
    assert action2 in actions
