from dataclasses import dataclass

import pytest

from robottraderslab_core._internal.actions import BookKeeper
from robottraderslab_core._internal.actions.exceptions import (
    ActionAlreadyExistsError,
    ActionNotFoundError,
)
from robottraderslab_core.exchanges import BaseExchangeAction


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
    """A fresh BookKeeper."""
    return BookKeeper()


@pytest.fixture
def action() -> MockExchangeAction:
    """An action to book."""
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
        pytest.param(lambda original: original, id="same-action"),
        pytest.param(
            lambda original: MockExchangeAction(custom_id=original.id),
            id="different-action-same-id",
        ),
    ],
)
def test_add_rejects_duplicates(bookkeeper, action, action_factory):
    bookkeeper.add(action)
    duplicate_action = action_factory(action)

    with pytest.raises(ActionAlreadyExistsError):
        bookkeeper.add(duplicate_action)


def test_add_records_what_an_action_follows(bookkeeper, action):
    follower = MockExchangeAction()
    bookkeeper.add(action)

    bookkeeper.add(follower, after=action)

    assert bookkeeper.declared_waits() == {follower.id: (action.id,)}


def test_add_rejects_following_an_unbooked_action(bookkeeper, action):
    with pytest.raises(ActionNotFoundError) as refusal:
        bookkeeper.add(MockExchangeAction(), after=action)

    assert str(action.id) in str(refusal.value)


def test_list_actions_returns_all_stored_actions(bookkeeper, action):
    second_action = MockExchangeAction()
    bookkeeper.add(action)
    bookkeeper.add(second_action)

    listed_actions = bookkeeper.list_actions()

    assert len(listed_actions) == 2
    assert action in listed_actions
    assert second_action in listed_actions


class TestReplacingTheActions:
    def test_the_actions_held_become_the_ones_given_in_their_sequence(self, bookkeeper):
        first, second, third = (MockExchangeAction() for _ in range(3))
        for booked in (first, second, third):
            bookkeeper.add(booked)

        bookkeeper._replace_actions([third, first])

        assert bookkeeper.list_actions() == [third, first]

    def test_a_wait_on_an_action_left_out_is_dropped(self, bookkeeper):
        first, second, third = (MockExchangeAction() for _ in range(3))
        bookkeeper.add(first)
        bookkeeper.add(second)
        bookkeeper.add(third, after=[first, second])

        bookkeeper._replace_actions([first, third])

        assert bookkeeper.declared_waits() == {third.id: (first.id,)}

    def test_an_action_that_waited_on_nothing_but_an_action_left_out_waits_for_nothing(
        self, bookkeeper
    ):
        dropped, kept = MockExchangeAction(), MockExchangeAction()
        bookkeeper.add(dropped)
        bookkeeper.add(kept, after=dropped)

        bookkeeper._replace_actions([kept])

        assert bookkeeper.declared_waits() == {kept.id: ()}

    def test_the_wait_of_an_action_left_out_is_forgotten(self, bookkeeper):
        first, dropped = MockExchangeAction(), MockExchangeAction()
        bookkeeper.add(first)
        bookkeeper.add(dropped, after=first)

        bookkeeper._replace_actions([first])

        assert bookkeeper.declared_waits() == {}


class TestFollowersOfAnAction:
    def test_a_follower_of_an_action_booked_to_wait_on_nothing_waits_on_that_action(
        self, bookkeeper
    ):
        leader = MockExchangeAction()
        follower = MockExchangeAction()
        bookkeeper.add(leader, after=[])

        bookkeeper._replace_with_followers([leader, follower], {follower.id: leader.id})

        assert bookkeeper.declared_waits() == {
            leader.id: (),
            follower.id: (leader.id,),
        }

    def test_a_follower_takes_the_waits_its_leader_was_booked_with(self, bookkeeper):
        before, leader, follower = (MockExchangeAction() for _ in range(3))
        bookkeeper.add(before)
        bookkeeper.add(leader, after=before)

        bookkeeper._replace_with_followers(
            [before, leader, follower], {follower.id: leader.id}
        )

        assert bookkeeper.declared_waits()[follower.id] == (before.id, leader.id)

    def test_a_follower_waits_for_the_followers_of_what_its_leader_waits_on(
        self, bookkeeper
    ):
        first, first_follower, second, second_follower = (
            MockExchangeAction() for _ in range(4)
        )
        bookkeeper.add(first)
        bookkeeper.add(second, after=first)

        bookkeeper._replace_with_followers(
            [first, first_follower, second, second_follower],
            {first_follower.id: first.id, second_follower.id: second.id},
        )

        assert bookkeeper.declared_waits()[second_follower.id] == (
            first.id,
            first_follower.id,
            second.id,
        )

    def test_a_follower_of_an_action_booked_with_no_waits_declares_none(
        self, bookkeeper
    ):
        leader, follower = MockExchangeAction(), MockExchangeAction()
        bookkeeper.add(leader)

        bookkeeper._replace_with_followers([leader, follower], {follower.id: leader.id})

        assert bookkeeper.declared_waits() == {}

    def test_several_followers_of_one_leader_are_all_waited_for(self, bookkeeper):
        leader, first, second, waiting = (MockExchangeAction() for _ in range(4))
        bookkeeper.add(leader)
        bookkeeper.add(waiting, after=leader)

        bookkeeper._replace_with_followers(
            [leader, first, second, waiting],
            {first.id: leader.id, second.id: leader.id},
        )

        assert bookkeeper.declared_waits() == {
            waiting.id: (leader.id, first.id, second.id)
        }
