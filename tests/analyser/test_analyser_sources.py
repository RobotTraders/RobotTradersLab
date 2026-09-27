import tempfile
import webbrowser
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from robottraderslab._core import (
    Balance,
    Execution,
    OrderSide,
    PositionSide,
    PositionSnapshot,
    Symbol,
)
from robottraderslab.analyser import Analyser
from robottraderslab.bootstrap import ReportConfig

_BTC = Symbol.create("BTC/USDT:USDT")
_ETH = Symbol.create("ETH/USDT:USDT")
_SINCE = datetime(2024, 1, 1, tzinfo=UTC)
_OPENED_AT = datetime(2024, 1, 1, 10, tzinfo=UTC)
_CLOSED_AT = datetime(2024, 1, 2, 10, tzinfo=UTC)
_UNTIL = datetime(2024, 1, 3, tzinfo=UTC)
_OPENING_BALANCE = 1000.0
_BOOKED_PROFIT = 10.0
_FEE = 0.1
_PROFILES = [
    {
        "symbol": str(_BTC),
        "timeframe": "1h",
        "fast_ma_length": 2,
        "slow_ma_length": 3,
    }
]


@pytest.fixture
def fills() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": [str(_BTC), str(_BTC), str(_ETH)],
            "side": ["long", "long", "long"],
            "gross_quantity": [1.0, 1.0, 2.0],
            "net_quantity": [1.0, 1.0, 2.0],
            "price": [100.0, 110.0, 50.0],
            "fee": [_FEE, _FEE, _FEE],
            "fill_type": ["enter_long", "exit_long", "enter_long"],
        },
        index=pd.DatetimeIndex([_OPENED_AT, _CLOSED_AT, _CLOSED_AT]),
    )


@pytest.fixture
def equity_curve() -> pd.Series:
    return pd.Series(
        [_OPENING_BALANCE, _OPENING_BALANCE + _BOOKED_PROFIT - 2 * _FEE],
        index=pd.DatetimeIndex([_SINCE, _UNTIL]),
    )


@pytest.fixture
def executions() -> list[Execution]:
    return [
        Execution(
            execution_id="opened",
            order_id="open-order",
            symbol=_BTC,
            side=OrderSide.BUY,
            price=100.0,
            quantity=1.0,
            timestamp=_OPENED_AT,
            realised_profit=None,
            fee=_FEE,
        ),
        Execution(
            execution_id="closed",
            order_id="close-order",
            symbol=_BTC,
            side=OrderSide.SELL,
            price=110.0,
            quantity=1.0,
            timestamp=_CLOSED_AT,
            realised_profit=_BOOKED_PROFIT,
            fee=_FEE,
        ),
    ]


@pytest.fixture
def open_position() -> dict[Symbol, PositionSnapshot]:
    return {
        _ETH: PositionSnapshot(
            symbol=_ETH,
            side=PositionSide.LONG,
            quantity=2.0,
            average_entry_price=50.0,
            entry_time=_CLOSED_AT,
            leverage=1.0,
            liquidation_price=5.0,
        )
    }


@pytest.fixture
def candles() -> pd.DataFrame:
    hours = pd.date_range(_SINCE.replace(tzinfo=None), periods=48, freq="h")
    close = np.linspace(100.0, 110.0, len(hours))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.ones(len(hours)),
        },
        index=hours,
    )


@pytest.fixture
def charted_provider(ohlcv_provider, candles) -> Mock:
    ohlcv_provider.fetch_ohlcv.return_value = candles
    return ohlcv_provider


@pytest.fixture
def backtest_analyser(fills, equity_curve, charted_provider) -> Analyser:
    return Analyser.from_fills(
        fills,
        equity_curve,
        charted_provider,
        report_config=ReportConfig(),
        profiles=_PROFILES,
    )


@pytest.fixture
def live_analyser(executions, open_position, charted_provider) -> Analyser:
    return Analyser.from_executions(
        executions=executions,
        positions=open_position,
        balances={
            "USDT": Balance(
                locked=0.0, total=_OPENING_BALANCE + _BOOKED_PROFIT - 2 * _FEE
            )
        },
        symbols=[_BTC, _ETH],
        since=_SINCE,
        until=_UNTIL,
        ohlcv_provider=charted_provider,
        report_config=ReportConfig(),
        profiles=_PROFILES,
    )


class TestOnePayloadContract:
    def test_both_sources_hand_out_trades_in_one_shape(
        self, backtest_analyser, live_analyser
    ):
        assert list(backtest_analyser.trades.columns) == list(
            live_analyser.trades.columns
        )

    def test_both_sources_hand_out_open_positions_in_one_shape(
        self, backtest_analyser, live_analyser
    ):
        assert list(backtest_analyser.open_positions.columns) == list(
            live_analyser.open_positions.columns
        )

    def test_a_source_with_nothing_open_keeps_the_shape(
        self, executions, charted_provider, live_analyser
    ):
        nothing_open = Analyser.from_executions(
            executions=executions,
            positions={},
            balances={"USDT": Balance(locked=0.0, total=_OPENING_BALANCE)},
            symbols=[_BTC],
            since=_SINCE,
            until=_UNTIL,
            ohlcv_provider=charted_provider,
            report_config=ReportConfig(),
            profiles=_PROFILES,
        )

        assert list(nothing_open.open_positions.columns) == list(
            live_analyser.open_positions.columns
        )


def _tabs(payload: dict[str, Any]) -> list[str]:
    report = payload["report"]
    tabs = []
    if report["sections"] or report["trades"]["groups"]:
        tabs.append("Report")
    if payload["markets"][0]["trades"]:
        tabs.append("Trades")
    return tabs


def _section_titles(payload: dict[str, Any]) -> list[str]:
    return [section["title"] for section in payload["report"]["sections"]]


@pytest.fixture(autouse=True)
def _chart_boundary(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(webbrowser, "open", lambda url: True)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))


@pytest.fixture
def charted(read_payload) -> Callable[[Analyser], dict[str, Any]]:
    def draw(analyser: Analyser) -> dict[str, Any]:
        written = analyser.plot_candlesticks(indicators_name="futures_ma")
        return read_payload(written[0])

    return draw


class TestSameTabs:
    def test_both_sources_open_to_the_same_tabs(
        self, backtest_analyser, live_analyser, charted
    ):
        backtest_tabs = _tabs(charted(backtest_analyser))
        live_tabs = _tabs(charted(live_analyser))

        assert backtest_tabs == live_tabs
        assert live_tabs == ["Report", "Trades"]

    def test_both_sources_report_the_same_sections(
        self, backtest_analyser, live_analyser, charted
    ):
        assert _section_titles(charted(backtest_analyser)) == _section_titles(
            charted(live_analyser)
        )
