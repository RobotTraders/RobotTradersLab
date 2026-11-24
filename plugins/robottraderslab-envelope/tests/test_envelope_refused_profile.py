import logging
from collections.abc import Callable
from typing import Any

import pytest

from robottraderslab import Symbol
from robottraderslab.strategies import TradingMode
from robottraderslab_envelope import EnvelopeStrategy

ETH = Symbol.create("ETH/USDT:USDT")


def _section(symbol: str, **overrides: Any) -> dict[str, Any]:
    section: dict[str, Any] = {
        "symbol": symbol,
        "timeframe": "1h",
        "total_balance_ratio": 1.0,
        "average_type": "SMA",
        "average_period": 100,
        "envelopes": [0.05],
        "stop_loss_pct": 0.15,
    }
    section.update(overrides)
    return section


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"envelopes": None}, "profile must declare envelopes"),
        (
            {"envelopes": [1.2]},
            "envelope offsets must be in (0, 1), got [1.2] on the long side",
        ),
        ({"leverage": 0.0}, "leverage must be greater than 0, got 0.0"),
        (
            {"stop_loss_pct": 1.5},
            "stop_loss_pct must be in (0, 1), got 1.5",
        ),
        (
            {"envelopes": [0.05, 0.3], "stop_loss_pct": 0.1},
            "stop_loss_pct 0.1 on BTC/USDT:USDT can sit inside the long ladder",
        ),
        ({"envelopes": []}, "envelopes must be a non-empty list"),
        (
            {"long_envelopes": [0.05]},
            "envelopes cannot be combined with long_envelopes or short_envelopes",
        ),
    ],
    ids=[
        "no_envelopes",
        "offset_out_of_range",
        "leverage_not_positive",
        "stop_loss_out_of_range",
        "stop_loss_inside_ladder",
        "empty_list",
        "envelopes_with_directional_list",
    ],
)
def test_a_section_the_class_refuses_sits_out_while_the_other_symbol_trades(
    make_strategy: Callable[..., EnvelopeStrategy], caplog, overrides, reason
):
    with caplog.at_level(logging.ERROR):
        strategy = make_strategy(
            [_section("ETH/USDT:USDT"), _section("BTC/USDT:USDT", **overrides)],
            TradingMode.LIVE,
        )

    assert [profile.symbol for profile in strategy.profiles] == [ETH]
    (refusal,) = caplog.records
    assert refusal.levelno == logging.ERROR
    assert refusal.getMessage().startswith(
        f"EnvelopeStrategy: profile 2 (BTC/USDT:USDT 1h): {reason}"
    )
