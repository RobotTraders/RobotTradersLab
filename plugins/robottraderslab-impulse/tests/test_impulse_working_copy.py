import json
import logging
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import time_machine
from impulse_helpers import write_signal_columns

from robottraderslab import Symbol
from robottraderslab.exchanges import OrderSide, PlacedOrder
from robottraderslab.live import main
from robottraderslab.strategies import OHLCVs
from robottraderslab_impulse import ImpulseStrategy
from robottraderslab_impulse.profile_config import ProfileConfig

BTC = Symbol.create("BTC/USDT:USDT")
ETH = Symbol.create("ETH/USDT:USDT")
PROFILE: dict[str, Any] = {
    "timeframe": "1d",
    "trix_length": 3,
    "signal_length": 3,
    "trend_length": 3,
    "total_balance_ratio": 0.1,
    "leverage": 2.0,
    "margin_mode": "isolated",
    "stop_loss_pct": 0.05,
}


class EnteringImpulse(ImpulseStrategy):
    """Impulse whose every profile signals a long entry on the newest candle."""

    def generate_profile_signals(self, profile: ProfileConfig, ohlcvs: OHLCVs) -> None:
        newest = len(ohlcvs.timestamps(profile.symbol, profile.timeframe)) - 1
        write_signal_columns(
            ohlcvs,
            profile.symbol,
            profile.timeframe,
            profile.profile_id,
            {newest: "long_entry"},
        )


@pytest.fixture
def venue(mock_exchange: Mock, monkeypatch: pytest.MonkeyPatch) -> Iterator[Mock]:
    """The connector the bot names in place of a venue, a mock of the protocol."""
    mock_exchange.place_market_order.return_value = PlacedOrder(order_id="market-1")
    monkeypatch.setattr(
        sys.modules[__name__],
        "make_exchange",
        lambda **settings: mock_exchange,
        raising=False,
    )
    with time_machine.travel(datetime(2026, 1, 1, tzinfo=UTC), tick=False):
        yield mock_exchange
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.close()
        root.removeHandler(handler)


@pytest.mark.parametrize(
    "second",
    [
        {"no_such_key": 1},
        {"stop_loss_pct": 0.07},
        {"leverage": 3.0},
        {"margin_mode": "cross"},
    ],
    ids=["refused section", "stop_loss_pct", "leverage", "margin_mode"],
)
def test_an_edit_breaking_a_stacked_symbol_runs_its_three_profiles_in_exit_only_mode(
    tmp_path, venue, second
):
    working_copy = _stacked_bot()
    folder = _seeded(tmp_path, working_copy)
    _save(folder, _stacked_bot(**second))

    exit_code = _cycle(folder)

    assert exit_code == 0
    assert _entered(venue) == [ETH]
    assert (folder / ".rtlab" / "bot.working.toml").read_text() == working_copy
    assert f"with {BTC} in exit-only mode" in _errors(folder)[-1]


def _stacked_bot(**second: Any) -> str:
    return _bot(
        _section(BTC, "alpha"),
        _section(BTC, "beta", **second),
        _section(BTC, "gamma"),
        _section(ETH, "delta"),
    )


def _seeded(tmp_path: Path, text: str) -> Path:
    folder = tmp_path / "impulse-bot"
    (folder / ".rtlab").mkdir(parents=True)
    _save(folder, text)
    (folder / ".rtlab" / "bot.working.toml").write_text(text)
    return folder


def _save(folder: Path, text: str) -> None:
    (folder / "bot.toml").write_text(text)


def _cycle(folder: Path) -> int:
    return main(
        folder=folder,
        logfile_override=None,
        console=False,
        logfiles_retention_days=None,
    )


def _entered(venue: Mock) -> list[Symbol]:
    return [
        placed.kwargs["symbol"]
        for placed in venue.place_market_order.await_args_list
        if placed.kwargs["side"] is OrderSide.BUY
    ]


def _errors(folder: Path) -> list[str]:
    log = (folder / "logs" / "live.log").read_text().splitlines()
    return [line for line in log if ": ERROR: " in line]


def _bot(*sections: str) -> str:
    return (
        f'[strategy]\nstrategy_class = "{__name__}.EnteringImpulse"\n\n'
        + "\n".join(sections)
        + f'\n[live.trading_account]\nexchange = "{__name__}:make_exchange"\n\n'
        '[live.ohlcv_provider]\nohlcv_provider = "mock"\nlookback = 30\n'
    )


def _section(symbol: Symbol, tag: str, **fields: Any) -> str:
    declared = {**PROFILE, "symbol": str(symbol), "tag": tag, **fields}
    return "[[strategy.profiles]]\n" + "".join(
        f"{key} = {json.dumps(value)}\n" for key, value in declared.items()
    )
