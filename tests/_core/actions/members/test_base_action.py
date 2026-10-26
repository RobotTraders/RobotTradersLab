import re
from datetime import UTC, datetime, timedelta

import pytest

from robottraderslab._core import (
    BaseExchangeAction,
    bot_leading_bytes,
    name_process,
    tag_of,
)
from robottraderslab._core.actions.members import carries_this_bot
from robottraderslab.exchanges import next_client_order_id, process_bot_bytes


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

        bot_and_launch, action_id, tag = action.client_order_id.split("-")

        assert re.fullmatch("[0-9a-z]{12}", bot_and_launch)
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


LAUNCH = datetime(2026, 10, 22, 8, 0, tzinfo=UTC)


def _first_segment(bot_name: str, launched_at: datetime) -> str:
    name_process(bot_name, launched_at)
    return MockAction(symbol="BTC/USDT").client_order_id.split("-")[0]


class TestProcessIdentity:
    def test_every_order_of_a_process_opens_alike(self):
        name_process("impulse-kraken", LAUNCH)

        first = MockAction(symbol="BTC/USDT", tag="4h-alpha").client_order_id
        second = MockAction(symbol="ETH/USDT").client_order_id

        assert first.split("-")[0] == second.split("-")[0]

    def test_two_processes_of_one_bot_share_the_bot(self):
        earlier = _first_segment("impulse-kraken", LAUNCH)
        later = _first_segment("impulse-kraken", LAUNCH + timedelta(days=2))

        assert earlier[:6] == later[:6]

    def test_two_bots_carry_different_bots(self):
        impulse = _first_segment("impulse-kraken", LAUNCH)
        envelope = _first_segment("envelope-kraken", LAUNCH)

        assert impulse[:6] != envelope[:6]

    def test_a_later_process_carries_a_greater_launch(self):
        earlier = _first_segment("impulse-kraken", LAUNCH)
        later = _first_segment("impulse-kraken", LAUNCH + timedelta(seconds=1))

        assert later[6:] > earlier[6:]

    def test_a_launch_a_minute_into_2026(self):
        launch = _first_segment(
            "impulse-kraken", datetime(2026, 1, 1, 0, 1, tzinfo=UTC)
        )

        assert launch[6:] == "00001o"


class TestBotLeadingBytes:
    def test_the_bytes_are_the_leading_two_of_the_names_digest(self):
        assert bot_leading_bytes("impulse-kraken") == bytes((0xF6, 0x9D))


class TestProcessBotBytes:
    def test_the_bytes_are_the_leading_two_of_the_bot_digest(self):
        name_process("impulse-kraken", LAUNCH)

        assert process_bot_bytes() == bytes((0xF6, 0x9D))

    def test_the_bytes_reach_past_the_range_of_the_segments_value(self):
        name_process("pilot", LAUNCH)

        assert process_bot_bytes() > bytes((0x81, 0xBF))


class TestNextClientOrderId:
    def test_a_connectors_id_opens_like_a_strategys_order(self):
        name_process("impulse-kraken", LAUNCH)

        strategys = MockAction(symbol="BTC/USDT", tag="4h-alpha").client_order_id
        connectors = next_client_order_id("tp-reached")

        assert connectors.split("-")[0] == strategys.split("-")[0]

    def test_a_connectors_id_carries_the_tag_it_was_given(self):
        assert tag_of(next_client_order_id("tp-reached")) == "tp-reached"

    def test_two_ids_never_share_a_number(self):
        name_process("impulse-kraken", LAUNCH)

        drawn = [next_client_order_id("sl-reached") for _ in range(3)]
        booked = MockAction(symbol="BTC/USDT")

        assert len({*drawn, booked.client_order_id}) == 4


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


class TestCarriesThisBot:
    def test_an_id_of_this_process_is_this_bots(self):
        name_process("impulse-kraken", LAUNCH)

        assert carries_this_bot(MockAction(symbol="BTC/USDT").client_order_id)

    def test_an_id_of_an_earlier_process_of_this_bot_is_this_bots(self):
        name_process("impulse-kraken", LAUNCH)
        earlier = MockAction(symbol="BTC/USDT").client_order_id
        name_process("impulse-kraken", LAUNCH + timedelta(days=2))

        assert carries_this_bot(earlier)

    def test_an_id_of_another_bot_is_not(self):
        name_process("envelope-bitget", LAUNCH)
        other = MockAction(symbol="BTC/USDT").client_order_id
        name_process("impulse-kraken", LAUNCH)

        assert not carries_this_bot(other)

    def test_a_hyperliquid_readback_of_this_bot_is_this_bots(self):
        name_process("impulse-kraken", LAUNCH)
        readback = f"{bot_leading_bytes('impulse-kraken').hex()}-4h-alpha"

        assert carries_this_bot(readback)

    def test_an_id_placed_before_the_bot_was_named_is_not(self):
        name_process("impulse-kraken", LAUNCH)

        assert not carries_this_bot("12f89ea14c0e-1-4h-alpha")

    def test_no_id_is_not(self):
        assert not carries_this_bot(None)
