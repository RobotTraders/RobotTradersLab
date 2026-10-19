from robottraderslab._core import OrderSettings, TimeInForce


def test_a_configured_field_takes_the_place_of_the_strategys():
    settings = OrderSettings(params={"bitget": {"stpMode": "cancel_maker"}})

    merged = settings.params_over(
        {"bitget": {"stpMode": "cancel_taker", "venueOnly": "99"}}
    )

    assert merged == {"bitget": {"stpMode": "cancel_maker", "venueOnly": "99"}}


def test_configured_fields_join_the_strategys_other_venues():
    settings = OrderSettings(params={"kraken": {"triggerSignal": "mark"}})

    merged = settings.params_over({"bitget": {"venueOnly": "99"}})

    assert merged == {
        "bitget": {"venueOnly": "99"},
        "kraken": {"triggerSignal": "mark"},
    }


def test_a_configured_time_in_force_wins():
    settings = OrderSettings(time_in_force=TimeInForce.IOC)

    assert settings.time_in_force_over(TimeInForce.GTC) is TimeInForce.IOC


def test_the_strategys_time_in_force_stands_when_none_is_configured():
    assert OrderSettings().time_in_force_over(TimeInForce.POST_ONLY) is (
        TimeInForce.POST_ONLY
    )
