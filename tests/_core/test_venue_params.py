import pytest

from robottraderslab.exceptions import StrategyCriticalError
from robottraderslab.exchanges import TriggerType, venue_trigger_type, with_venue_params

NOTHING_RESERVED: frozenset[str] = frozenset()


def test_a_strategy_field_joins_the_body():
    merged = with_venue_params(
        {"size": "1"},
        {"bitget": {"venueOnly": "99"}},
        venue="bitget",
        reserved=NOTHING_RESERVED,
    )

    assert merged == {"size": "1", "venueOnly": "99"}


def test_a_field_addressed_to_another_venue_stays_out():
    merged = with_venue_params(
        {"size": "1"},
        {"kraken": {"triggerSignal": "mark"}},
        venue="bitget",
        reserved=NOTHING_RESERVED,
    )

    assert merged == {"size": "1"}


def test_a_strategy_field_naming_a_field_set_from_the_order():
    with pytest.raises(
        StrategyCriticalError, match="bitget params set clientOid, size"
    ):
        with_venue_params(
            {"size": "1", "clientOid": "c1"},
            {"bitget": {"size": "9", "clientOid": "other"}},
            venue="bitget",
            reserved=NOTHING_RESERVED,
        )


def test_a_strategy_field_naming_a_reserved_field_the_order_left_out():
    with pytest.raises(StrategyCriticalError, match="stopLoss"):
        with_venue_params(
            {"size": "1"},
            {"bitget": {"stopLoss": "90"}},
            venue="bitget",
            reserved=frozenset({"stopLoss"}),
        )


def test_the_default_trigger_type_is_the_venues_own():
    name = venue_trigger_type(
        TriggerType.DEFAULT,
        {TriggerType.MARK_PRICE: "mark"},
        default="last",
        venue="Kraken",
    )

    assert name == "last"


def test_a_trigger_type_in_the_venues_words():
    name = venue_trigger_type(
        TriggerType.MARK_PRICE,
        {TriggerType.MARK_PRICE: "mark"},
        default="last",
        venue="Kraken",
    )

    assert name == "mark"


def test_a_trigger_type_the_venue_does_not_offer():
    with pytest.raises(StrategyCriticalError, match="Kraken offers no index_price"):
        venue_trigger_type(
            TriggerType.INDEX_PRICE,
            {TriggerType.MARK_PRICE: "mark"},
            default="last",
            venue="Kraken",
        )
