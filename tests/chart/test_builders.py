from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pytest

from robottraderslab._core import (
    ChartLine,
    DrawdownUnit,
    PerformanceReport,
    ReportRow,
    ReportSection,
    ReportTable,
)
from robottraderslab.chart.builders import build_market_view, build_payload
from robottraderslab.chart.payload import (
    ChartPayload,
    MarkerSide,
    MarketView,
    Pane,
    Shape,
)


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
        },
        index=pd.date_range("2024-01-01", periods=2, freq="D"),
    )


@pytest.fixture
def trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["BTC/USDT:USDT"],
            "side": ["long"],
            "entry_time": pd.to_datetime(["2024-01-01"]),
            "entry_price": [100.0],
            "exit_time": pd.to_datetime(["2024-01-02"]),
            "exit_price": [110.0],
            "net_pnl": [9.5],
            "net_pnl_pct": [0.095],
            "entry_reason": ["ma_cross"],
            "exit_reason": ["take_profit"],
        }
    )


@pytest.fixture
def open_trade() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["BTC/USDT:USDT"],
            "side": ["long"],
            "entry_time": pd.to_datetime(["2024-01-01"]),
            "entry_price": [100.0],
            "exit_time": [pd.NaT],
            "exit_price": [0.0],
            "net_pnl": [0.0],
            "net_pnl_pct": [0.0],
            "entry_reason": ["ma_cross"],
            "exit_reason": [None],
        }
    )


@pytest.fixture
def report() -> PerformanceReport:
    return PerformanceReport(
        headline=[ReportRow(label="Performance", value="9.50%")],
        sections=[
            ReportSection(
                title="Returns", rows=[ReportRow(label="Return", value="9.50%")]
            )
        ],
        trades=ReportTable(title="Trades", columns=["All"], groups=[]),
    )


@pytest.fixture
def build_market(trades: pd.DataFrame) -> Callable[..., MarketView]:
    def build(ohlcv: pd.DataFrame, **overrides: Any) -> MarketView:
        arguments: dict[str, Any] = {
            "label": "BTC/USDT:USDT · 1d",
            "settings": [],
            "ohlcv": ohlcv,
            "lines": [],
            "trades": trades.iloc[0:0],
        }
        arguments.update(overrides)
        return build_market_view(**arguments)

    return build


@pytest.fixture
def build(empty_report: PerformanceReport) -> Callable[..., ChartPayload]:
    def build(**overrides: Any) -> ChartPayload:
        arguments: dict[str, Any] = {
            "title": "Impulse",
            "markets": [],
            "report": empty_report,
            "equity_curve": pd.Series(dtype=float),
            "drawdown_curve": pd.Series(dtype=float),
            "drawdown_unit": DrawdownUnit.PERCENT,
            "links": [],
        }
        arguments.update(overrides)
        return build_payload(**arguments)

    return build


class TestBuildMarketView:
    def test_a_candle_is_its_time_followed_by_its_prices(self, build_market, ohlcv):
        market = build_market(ohlcv)

        assert market.candles[0] == [1704067200, 100.0, 102.0, 99.0, 101.0]

    def test_a_naive_index_is_read_as_utc(self, build_market, ohlcv):
        market = build_market(ohlcv)

        assert market.candles[0][0] == 1704067200

    def test_an_offset_index_is_converted_to_utc(self, build_market, ohlcv):
        shifted = ohlcv.set_axis(ohlcv.index.tz_localize("Etc/GMT-2"))

        market = build_market(shifted)

        assert market.candles[0][0] == 1704067200 - 2 * 3600

    def test_a_long_trade_buys_in_and_sells_out(self, build_market, ohlcv, trades):
        market = build_market(ohlcv, trades=trades)

        assert [marker.side for marker in market.markers] == [
            MarkerSide.BUY,
            MarkerSide.SELL,
        ]

    def test_a_short_trade_sells_in_and_buys_out(self, build_market, ohlcv, trades):
        short_trade = trades.assign(side=["short"])

        market = build_market(ohlcv, trades=short_trade)

        assert [marker.side for marker in market.markers] == [
            MarkerSide.SELL,
            MarkerSide.BUY,
        ]

    def test_markers_use_the_trade_reasons_as_labels(self, build_market, ohlcv, trades):
        market = build_market(ohlcv, trades=trades)

        assert [marker.label for marker in market.markers] == [
            "ma_cross",
            "take_profit",
        ]

    def test_markers_are_ordered_by_time(self, build_market, ohlcv):
        two_trades = pd.DataFrame(
            {
                "side": ["long", "long"],
                "entry_time": pd.to_datetime(["2024-01-03", "2024-01-01"]),
                "entry_price": [300.0, 100.0],
                "exit_time": pd.to_datetime(["2024-01-04", "2024-01-02"]),
                "exit_price": [330.0, 110.0],
                "net_pnl": [30.0, 10.0],
                "net_pnl_pct": [0.1, 0.1],
                "entry_reason": ["b", "a"],
                "exit_reason": ["b_out", "a_out"],
            }
        )

        market = build_market(ohlcv, trades=two_trades)

        marker_times = [marker.time for marker in market.markers]
        assert marker_times == sorted(marker_times)

    def test_an_open_trade_contributes_no_exit_marker(
        self, build_market, ohlcv, open_trade
    ):
        market = build_market(ohlcv, trades=open_trade)

        assert [marker.side for marker in market.markers] == [MarkerSide.BUY]

    def test_a_line_keeps_the_pane_it_asked_for(self, build_market, ohlcv):
        lines = [
            ChartLine(
                name="TRIX",
                values=np.array([1.0, 2.0]),
                colour="blue",
                pane="separate",
            )
        ]

        market = build_market(ohlcv, lines=lines)

        assert market.series[0].pane is Pane.SEPARATE

    def test_a_measure_of_distance_from_zero_is_drawn_as_bars(
        self, build_market, ohlcv
    ):
        lines = [
            ChartLine(
                name="Histogram",
                values=np.array([1.0, -2.0]),
                colour="gray",
                pane="separate",
                shape="histogram",
            )
        ]

        market = build_market(ohlcv, lines=lines)

        assert market.series[0].shape is Shape.HISTOGRAM

    def test_a_line_is_drawn_as_a_curve_unless_it_says_otherwise(
        self, build_market, ohlcv
    ):
        lines = [ChartLine(name="TRIX", values=np.array([1.0, 2.0]), colour="blue")]

        market = build_market(ohlcv, lines=lines)

        assert market.series[0].shape is Shape.LINE

    def test_a_line_carries_a_point_per_candle(self, build_market, ohlcv):
        lines = [ChartLine(name="TRIX", values=np.array([1.0, 2.0]), colour="blue")]

        market = build_market(ohlcv, lines=lines)

        assert [point[1] for point in market.series[0].points] == [1.0, 2.0]

    def test_a_line_carries_no_point_where_it_has_no_value(self, build_market, ohlcv):
        lines = [ChartLine(name="TRIX", values=np.array([np.nan, 2.0]), colour="blue")]

        market = build_market(ohlcv, lines=lines)

        assert [point[1] for point in market.series[0].points] == [2.0]

    def test_no_trades_yields_no_markers(self, build_market, ohlcv):
        market = build_market(ohlcv)

        assert market.markers == []


class TestBuildPayload:
    def test_drawdown_values_reach_the_payload_as_given(self, build, ohlcv):
        drawdown = pd.Series([0.0, -10.9], index=ohlcv.index)

        payload = build(drawdown_curve=drawdown)

        assert payload.drawdown[1][1] == pytest.approx(-10.9)

    def test_the_drawdown_unit_travels_with_its_values(self, build, ohlcv):
        drawdown = pd.Series([0.0, -12.05], index=ohlcv.index)

        payload = build(drawdown_curve=drawdown, drawdown_unit=DrawdownUnit.CURRENCY)

        assert payload.drawdown_unit == DrawdownUnit.CURRENCY

    def test_equity_stays_in_account_units(self, build, ohlcv):
        equity = pd.Series([1000.0, 1010.0], index=ohlcv.index)

        payload = build(equity_curve=equity)

        assert payload.equity[1][1] == 1010.0


class TestTradeRows:
    def test_each_trade_yields_a_row(self, build_market, ohlcv, trades):
        market = build_market(ohlcv, trades=trades)

        assert len(market.trades) == 1

    def test_rows_carry_epoch_seconds(self, build_market, ohlcv, trades):
        market = build_market(ohlcv, trades=trades)

        assert market.trades[0].entry_time == 1704067200
        assert market.trades[0].exit_time == 1704153600

    def test_rows_carry_the_traded_prices(self, build_market, ohlcv, trades):
        market = build_market(ohlcv, trades=trades)

        assert market.trades[0].entry_price == 100.0
        assert market.trades[0].exit_price == 110.0

    def test_rows_carry_the_side(self, build_market, ohlcv, trades):
        market = build_market(ohlcv, trades=trades)

        assert market.trades[0].side == "long"

    def test_rows_carry_the_trade_reasons(self, build_market, ohlcv, trades):
        market = build_market(ohlcv, trades=trades)

        assert market.trades[0].entry_reason == "ma_cross"
        assert market.trades[0].exit_reason == "take_profit"

    def test_rows_carry_the_net_pnl(self, build_market, ohlcv, trades):
        market = build_market(ohlcv, trades=trades)

        assert market.trades[0].net_pnl == 9.5

    def test_pnl_fractions_become_percentages(self, build_market, ohlcv, trades):
        market = build_market(ohlcv, trades=trades)

        assert market.trades[0].net_pnl_pct == pytest.approx(9.5)

    def test_a_losing_trade_keeps_its_sign(self, build_market, ohlcv, trades):
        losing = trades.assign(net_pnl=[-9.5], net_pnl_pct=[-0.095])

        market = build_market(ohlcv, trades=losing)

        assert market.trades[0].net_pnl == -9.5
        assert market.trades[0].net_pnl_pct == pytest.approx(-9.5)

    def test_an_open_trade_has_no_exit(self, build_market, ohlcv, open_trade):
        market = build_market(ohlcv, trades=open_trade)

        assert market.trades[0].exit_time is None
        assert market.trades[0].exit_price is None
        assert market.trades[0].exit_reason is None

    def test_an_open_trade_keeps_its_entry(self, build_market, ohlcv, open_trade):
        market = build_market(ohlcv, trades=open_trade)

        assert market.trades[0].entry_time == 1704067200
        assert market.trades[0].entry_reason == "ma_cross"

    def test_no_trades_yields_no_rows(self, build_market, ohlcv):
        market = build_market(ohlcv)

        assert market.trades == []


class TestReport:
    def test_the_report_reaches_the_payload_untouched(self, build, report):
        payload = build(report=report)

        assert payload.report is report
