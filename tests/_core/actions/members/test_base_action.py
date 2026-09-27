import pytest

from robottraderslab._core.actions.members.base_action import BaseExchangeAction
from robottraderslab._core.actions.members.tags import tag_of


class MockAction(BaseExchangeAction):
    """Mock implementation of BaseExchangeAction for testing."""

    def execute(self) -> None:
        """Mock execute method for testing."""
        pass


@pytest.mark.parametrize("num_actions", [1, 10, 100])
def test_unique_ids(num_actions):
    """Test that each action gets a unique ID."""
    actions = [MockAction(symbol="BTC/USDT") for _ in range(num_actions)]
    ids = {action.id for action in actions}
    assert len(ids) == len(actions)  # All IDs should be unique


def test_id_is_generated() -> None:
    """Test that an ID is automatically generated for each action."""
    action = MockAction(symbol="BTC/USDT")
    assert action.id is not None


class TestClientOrderId:
    def test_untagged_ids_are_unique(self):
        client_order_ids = {
            MockAction(symbol="BTC/USDT").client_order_id for _ in range(10)
        }

        assert len(client_order_ids) == 10

    def test_tagged_id_stays_unique(self):
        client_order_ids = {
            MockAction(symbol="BTC/USDT", tag="r2").client_order_id for _ in range(10)
        }

        assert len(client_order_ids) == 10

    def test_every_segment_is_separated(self):
        action = MockAction(symbol="BTC/USDT", tag="r2")

        process_tag, action_id, tag = action.client_order_id.split("-")

        assert len(process_tag) == 12
        assert action_id == str(action.id)
        assert tag == "r2"

    def test_untagged_id_stops_after_the_action_id(self):
        action = MockAction(symbol="BTC/USDT")

        assert action.client_order_id.endswith(f"-{action.id}")

    def test_tag_survives_the_round_trip(self):
        action = MockAction(symbol="BTC/USDT", tag="r2")

        assert tag_of(action.client_order_id) == "r2"

    def test_untagged_action_carries_no_tag(self):
        action = MockAction(symbol="BTC/USDT")

        assert tag_of(action.client_order_id) is None


class TestTagOf:
    @pytest.mark.parametrize(
        ("client_order_id", "expected_tag"),
        [
            ("t1785908301620", None),
            (None, None),
            ("123972f3dd28-7-2", "2"),
            ("--r2", "r2"),
        ],
        ids=[
            "id_from_another_program",
            "missing_id",
            "id_whose_tag_is_numeric",
            "id_whose_leading_segments_are_empty",
        ],
    )
    def test_tag_read_off_an_id(self, client_order_id, expected_tag):
        assert tag_of(client_order_id) == expected_tag

    def test_tag_keeps_its_own_separators(self):
        action = MockAction(symbol="BTC/USDT", tag="grid-3-of-5")

        assert tag_of(action.client_order_id) == "grid-3-of-5"
