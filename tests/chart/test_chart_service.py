from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from robottraderslab._core import (
    Candles,
    ChartLine,
    DrawdownUnit,
    OHLCVProviderProtocol,
    PerformanceReport,
    ReportRow,
    ReportSection,
    ReportTable,
)
from robottraderslab.chart import ChartService
from robottraderslab.exceptions import ExchangeCriticalError

_REPORT = PerformanceReport(
    headline=[ReportRow(label="Performance", value="9.50%")],
    sections=[
        ReportSection(title="Returns", rows=[ReportRow(label="Return", value="9.50%")])
    ],
    trades=ReportTable(title="Trades", columns=["All"], groups=[]),
)
_TRADES = pd.DataFrame(
    {
        "symbol": ["BTC/USDT:USDT", "ETH/USDT:USDT", "BTC/USDT:USDT"],
        "side": ["long", "long", "short"],
        "entry_time": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "entry_price": [100.0, 200.0, 300.0],
        "exit_time": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        "exit_price": [110.0, 190.0, 330.0],
        "net_pnl": [10.0, -10.0, -30.0],
        "net_pnl_pct": [0.1, -0.05, -0.1],
        "entry_reason": ["signal", "signal", "signal"],
        "exit_reason": ["profit", "loss", "profit"],
    }
)
_NO_TRADES = _TRADES.iloc[0:0]
_EMPTY_CURVE = pd.Series(dtype=float, index=pd.DatetimeIndex([]))


class RecordingChartLines:
    """Hands back the lines it was given and keeps every parameter set it saw."""

    def __init__(self, lines: list[ChartLine]) -> None:
        self.lines = lines
        self.params_seen: list[dict[str, Any]] = []

    def __call__(
        self, indicators_name: str, candles: Candles, indicators_params: dict[str, Any]
    ) -> list[ChartLine]:
        self.params_seen.append(indicators_params)
        return self.lines


@pytest.fixture
def ohlcv_provider() -> Mock:
    provider = Mock(spec=OHLCVProviderProtocol)
    provider.fetch_ohlcv.return_value = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0],
            "close": [101.0, 102.0, 103.0],
            "volume": [10.0, 11.0, 12.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    return provider


@pytest.fixture
def chart_lines() -> RecordingChartLines:
    return RecordingChartLines([])


@pytest.fixture
def service(
    ohlcv_provider: Mock, chart_lines: RecordingChartLines, browser_openings: list[str]
) -> ChartService:
    return ChartService(ohlcv_provider, chart_lines)


@pytest.fixture
def profiles() -> list[dict[str, Any]]:
    return [
        {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "tag": "alpha",
            "trix_length": 31,
        },
        {
            "symbol": "BTC/USDT:USDT",
            "timeframe": "1h",
            "tag": "beta",
            "trix_length": 43,
        },
        {"symbol": "ETH/USDT:USDT", "timeframe": "4h", "trix_length": 15},
    ]


def _market(payload: dict[str, Any], index: int = 0) -> dict[str, Any]:
    return payload["markets"][index]


def _open(service: ChartService, tmp_path: Path, **overrides: Any) -> Path:
    arguments: dict[str, Any] = {
        "indicators_name": "test.module",
        "symbol": "BTC/USDT:USDT",
        "timeframe": "1d",
        "indicators_params": {},
        "report": _REPORT,
        "equity_curve": _EMPTY_CURVE,
        "drawdown_curve": _EMPTY_CURVE,
        "drawdown_unit": DrawdownUnit.PERCENT,
        "trades_df": _NO_TRADES,
        "destination_dir": tmp_path,
    }
    arguments.update(overrides)
    return service.open_candlestick(**arguments)


def _open_all(
    service: ChartService,
    tmp_path: Path,
    charts: list[dict[str, Any]],
    trades: pd.DataFrame,
    **overrides: Any,
) -> list[Path]:
    arguments: dict[str, Any] = {
        "indicators_name": "test.module",
        "charts": charts,
        "report": _REPORT,
        "equity_curve": _EMPTY_CURVE,
        "drawdown_curve": _EMPTY_CURVE,
        "drawdown_unit": DrawdownUnit.PERCENT,
        "trades_df": trades,
        "destination_dir": tmp_path,
    }
    arguments.update(overrides)
    return service.open_candlesticks(**arguments)


def _write_report(
    service: ChartService,
    destination: Path,
    charts: list[dict[str, Any]],
    trades: pd.DataFrame,
    **overrides: Any,
) -> Path:
    arguments: dict[str, Any] = {
        "indicators_name": "test.module",
        "charts": charts,
        "report": _REPORT,
        "equity_curve": _EMPTY_CURVE,
        "drawdown_curve": _EMPTY_CURVE,
        "drawdown_unit": DrawdownUnit.PERCENT,
        "trades_df": trades,
        "destination": destination,
        "run_label": "test-bot · 2026-09-20_12-04-33",
    }
    arguments.update(overrides)
    return service.write_report(**arguments)


class TestChartService:
    def test_the_written_chart_is_opened_in_the_browser(
        self, service, tmp_path, browser_openings
    ):
        written = _open(service, tmp_path)

        assert browser_openings == [written.as_uri()]

    def test_the_candles_reach_the_document(self, service, tmp_path, read_payload):
        written = _open(service, tmp_path)

        assert len(_market(read_payload(written))["candles"]) == 3

    def test_the_chart_lands_in_the_given_directory(self, service, tmp_path):
        assert _open(service, tmp_path).parent == tmp_path

    def test_symbol_separators_are_stripped_from_the_filename(self, service, tmp_path):
        assert _open(service, tmp_path).name == "BTC-USDT-USDT_1d.html"

    def test_without_a_directory_each_call_gets_a_fresh_one(self, service):
        first = _open(service, None, destination_dir=None)
        second = _open(service, None, destination_dir=None)

        assert first.parent != second.parent
        assert first.exists()
        assert second.exists()

    def test_an_ohlcv_fetch_failure_reaches_the_caller(
        self, service, tmp_path, ohlcv_provider
    ):
        ohlcv_provider.fetch_ohlcv.side_effect = ExchangeCriticalError("bad key")

        with pytest.raises(ExchangeCriticalError, match="bad key"):
            _open(service, tmp_path)


class TestOpenSaved:
    def test_opens_an_already_written_page_without_rendering(
        self, service, tmp_path, browser_openings, ohlcv_provider
    ):
        written = _open(service, tmp_path)
        ohlcv_provider.fetch_ohlcv.reset_mock()

        service.open_saved(written)

        assert browser_openings == [written.as_uri(), written.as_uri()]
        ohlcv_provider.fetch_ohlcv.assert_not_called()


class TestWriteCandlesticks:
    def test_nothing_is_opened_in_the_browser(
        self, service, tmp_path, profiles, browser_openings
    ):
        written = service.write_candlesticks(
            indicators_name="test.module",
            charts=profiles,
            report=_REPORT,
            equity_curve=_EMPTY_CURVE,
            drawdown_curve=_EMPTY_CURVE,
            drawdown_unit=DrawdownUnit.PERCENT,
            trades_df=_NO_TRADES,
            destination_dir=tmp_path,
        )

        assert len(written) == 3
        assert browser_openings == []

    def test_the_title_names_the_strategy(self, service, tmp_path, read_payload):
        written = service.write_candlesticks(
            indicators_name="test_module",
            charts=[{"symbol": "BTC/USDT:USDT", "timeframe": "1d"}],
            report=_REPORT,
            equity_curve=_EMPTY_CURVE,
            drawdown_curve=_EMPTY_CURVE,
            drawdown_unit=DrawdownUnit.PERCENT,
            trades_df=_NO_TRADES,
            destination_dir=tmp_path,
        )

        assert read_payload(written[0])["title"] == "Test Module"


class TestWriteReport:
    def test_nothing_is_opened_in_the_browser(
        self, service, tmp_path, profiles, browser_openings
    ):
        _write_report(service, tmp_path / "report.html", profiles, _NO_TRADES)

        assert browser_openings == []

    def test_one_file_holds_every_market(self, service, tmp_path, profiles):
        destination = _write_report(
            service, tmp_path / "report.html", profiles, _NO_TRADES
        )

        assert destination == tmp_path / "report.html"
        assert list(tmp_path.glob("*.html")) == [destination]

    def test_every_market_is_embedded_with_its_own_label(
        self, service, tmp_path, profiles, read_payload
    ):
        destination = _write_report(
            service, tmp_path / "report.html", profiles, _NO_TRADES
        )

        labels = [market["label"] for market in read_payload(destination)["markets"]]
        assert labels == [
            "BTC/USDT:USDT · 1h · alpha",
            "BTC/USDT:USDT · 1h · beta",
            "ETH/USDT:USDT · 4h",
        ]

    def test_a_page_of_its_own_carries_no_links(
        self, service, tmp_path, profiles, read_payload
    ):
        destination = _write_report(
            service, tmp_path / "report.html", profiles, _NO_TRADES
        )

        assert read_payload(destination)["links"] == []

    def test_each_market_draws_only_its_own_trades(
        self, service, tmp_path, profiles, read_payload
    ):
        trades = pd.DataFrame(
            {
                "symbol": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
                "profile_name": ["BTC/USDT:USDT@1h-alpha", "ETH/USDT:USDT@4h"],
                "side": ["long", "long"],
                "entry_time": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "entry_price": [100.0, 200.0],
                "exit_time": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "exit_price": [110.0, 210.0],
                "net_pnl": [10.0, 10.0],
                "net_pnl_pct": [0.1, 0.05],
                "entry_reason": ["in", "in"],
                "exit_reason": ["out", "out"],
            }
        )

        destination = _write_report(service, tmp_path / "report.html", profiles, trades)

        markets = read_payload(destination)["markets"]
        assert [trade["entry_price"] for trade in markets[0]["trades"]] == [100.0]
        assert [trade["entry_price"] for trade in markets[1]["trades"]] == []
        assert [trade["entry_price"] for trade in markets[2]["trades"]] == [200.0]

    def test_the_configuration_and_run_label_reach_the_page(
        self, service, tmp_path, profiles, read_payload
    ):
        destination = _write_report(
            service,
            tmp_path / "report.html",
            profiles,
            _NO_TRADES,
            configuration='[strategy]\nstrategy_class = "impulse"\n',
            run_label="impulse-bot-example · 2026-09-20_12-04-33",
        )

        drawn = read_payload(destination)
        assert drawn["configuration"] == '[strategy]\nstrategy_class = "impulse"\n'
        assert drawn["title"] == "impulse-bot-example · 2026-09-20_12-04-33"


class TestTradeSelection:
    def test_only_the_charted_symbol_is_drawn(self, service, tmp_path, read_payload):
        written = _open(service, tmp_path, trades_df=_TRADES)

        assert len(_market(read_payload(written))["markers"]) == 4

    def test_a_symbol_the_run_never_traded_is_drawn_without_trades(
        self, service, tmp_path, read_payload
    ):
        written = _open(service, tmp_path, symbol="SOL/USDT:USDT", trades_df=_TRADES)

        drawn = _market(read_payload(written))
        assert drawn["markers"] == []
        assert drawn["trades"] == []

    def test_a_run_without_trades_draws_no_markers(
        self, service, tmp_path, read_payload
    ):
        written = _open(service, tmp_path, trades_df=_NO_TRADES)

        assert _market(read_payload(written))["markers"] == []


class TestProfiles:
    def test_one_chart_per_profile(self, service, tmp_path, profiles):
        assert len(_open_all(service, tmp_path, profiles, _NO_TRADES)) == 3

    def test_profiles_sharing_a_market_land_in_distinct_files(
        self, service, tmp_path, profiles
    ):
        written = _open_all(service, tmp_path, profiles, _NO_TRADES)

        assert [path.name for path in written] == [
            "BTC-USDT-USDT_1h_alpha.html",
            "BTC-USDT-USDT_1h_beta.html",
            "ETH-USDT-USDT_4h.html",
        ]

    def test_every_chart_lands_in_one_directory(self, service, tmp_path, profiles):
        written = _open_all(service, tmp_path, profiles, _NO_TRADES)

        assert {path.parent for path in written} == {tmp_path}

    def test_a_run_without_trades_draws_every_profile_without_markers(
        self, service, tmp_path, profiles, read_payload
    ):
        drawn = [
            _market(read_payload(path))
            for path in _open_all(service, tmp_path, profiles, _NO_TRADES)
        ]

        assert all(chart["markers"] == [] for chart in drawn)
        assert all(chart["trades"] == [] for chart in drawn)

    def test_links_point_at_the_sibling_files(
        self, service, tmp_path, profiles, read_payload
    ):
        written = _open_all(service, tmp_path, profiles, _NO_TRADES)

        hrefs = [link["href"] for link in read_payload(written[0])["links"]]
        assert hrefs == [path.name for path in written]

    def test_the_charted_profile_is_the_current_link(
        self, service, tmp_path, profiles, read_payload
    ):
        written = _open_all(service, tmp_path, profiles, _NO_TRADES)

        current = [link["current"] for link in read_payload(written[1])["links"]]
        assert current == [False, True, False]

    def test_links_name_the_symbol_timeframe_and_tag(
        self, service, tmp_path, profiles, read_payload
    ):
        written = _open_all(service, tmp_path, profiles, _NO_TRADES)

        labels = [link["label"] for link in read_payload(written[0])["links"]]
        assert labels == [
            "BTC/USDT:USDT · 1h · alpha",
            "BTC/USDT:USDT · 1h · beta",
            "ETH/USDT:USDT · 4h",
        ]

    def test_links_carry_each_profiles_settings(
        self, service, tmp_path, profiles, read_payload
    ):
        written = _open_all(service, tmp_path, profiles, _NO_TRADES)

        first_link = read_payload(written[0])["links"][0]
        assert first_link["settings"] == [
            {"name": "Symbol", "value": "BTC/USDT:USDT"},
            {"name": "Timeframe", "value": "1h"},
            {"name": "Tag", "value": "alpha"},
            {"name": "Trix length", "value": "31"},
        ]

    def test_every_chart_carries_the_same_report(
        self, service, tmp_path, profiles, read_payload
    ):
        drawn = [
            read_payload(path)
            for path in _open_all(service, tmp_path, profiles, _NO_TRADES)
        ]

        assert [chart["report"]["headline"] for chart in drawn] == [
            [{"label": "Performance", "value": "9.50%"}]
        ] * 3

    def test_a_lone_profile_has_no_links(
        self, service, tmp_path, profiles, read_payload
    ):
        written = _open_all(service, tmp_path, profiles[:1], _NO_TRADES)

        assert read_payload(written[0])["links"] == []

    def test_the_first_profile_is_the_one_opened(
        self, service, tmp_path, profiles, browser_openings
    ):
        written = _open_all(service, tmp_path, profiles, _NO_TRADES)

        assert browser_openings == [written[0].as_uri()]

    def test_the_whole_profile_reaches_the_indicators(
        self, service, tmp_path, profiles, chart_lines
    ):
        _open_all(service, tmp_path, profiles, _NO_TRADES)

        assert chart_lines.params_seen[0] == profiles[0]


class TestProfileSelection:
    @pytest.fixture
    def trades_of_two_profiles(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "symbol": ["BTC/USDT:USDT"] * 3,
                "profile_name": [
                    "BTC/USDT:USDT@1h-alpha",
                    "BTC/USDT:USDT@2h-alpha",
                    "BTC/USDT:USDT@2h-beta",
                ],
                "side": ["long", "long", "long"],
                "entry_time": pd.to_datetime(["2024-01-01"] * 3),
                "entry_price": [100.0, 200.0, 300.0],
                "exit_time": pd.to_datetime(["2024-01-02"] * 3),
                "exit_price": [110.0, 210.0, 310.0],
                "net_pnl": [10.0, 10.0, 10.0],
                "net_pnl_pct": [0.1, 0.05, 0.03],
                "entry_reason": ["in", "in", "in"],
                "exit_reason": ["out", "out", "out"],
            }
        )

    def test_a_chart_draws_only_its_own_profile(
        self, service, tmp_path, trades_of_two_profiles, read_payload
    ):
        charts = [
            {"symbol": "BTC/USDT:USDT", "timeframe": "2h", "tag": "alpha"},
            {"symbol": "BTC/USDT:USDT", "timeframe": "1h", "tag": "alpha"},
        ]

        written = _open_all(service, tmp_path, charts, trades_of_two_profiles)

        drawn = _market(read_payload(written[0]))
        assert [trade["entry_price"] for trade in drawn["trades"]] == [200.0]

    def test_matching_is_exact_not_a_substring(
        self, service, tmp_path, trades_of_two_profiles, read_payload
    ):
        charts = [{"symbol": "BTC/USDT:USDT", "timeframe": "2h", "tag": "al"}]

        written = _open_all(service, tmp_path, charts, trades_of_two_profiles)

        assert _market(read_payload(written[0]))["trades"] == []

    def test_a_profile_that_never_traded_is_drawn_without_trades(
        self, service, tmp_path, trades_of_two_profiles, read_payload
    ):
        charts = [
            {"symbol": "SOL/USDT:USDT", "timeframe": "2h", "tag": "alpha"},
            {"symbol": "BTC/USDT:USDT", "timeframe": "2h", "tag": "alpha"},
        ]

        written = _open_all(service, tmp_path, charts, trades_of_two_profiles)

        quiet, traded = (_market(read_payload(path)) for path in written)
        assert quiet["trades"] == []
        assert quiet["markers"] == []
        assert [trade["entry_price"] for trade in traded["trades"]] == [200.0]

    def test_trades_that_were_never_named_are_all_drawn(
        self, service, tmp_path, trades_of_two_profiles, read_payload
    ):
        charts = [{"symbol": "BTC/USDT:USDT", "timeframe": "2h", "tag": "alpha"}]

        written = _open_all(
            service,
            tmp_path,
            charts,
            trades_of_two_profiles.drop(columns=["profile_name"]),
        )

        assert len(_market(read_payload(written[0]))["trades"]) == 3


class TestWindowing:
    @pytest.fixture
    def chart_lines(self) -> RecordingChartLines:
        return RecordingChartLines(
            [ChartLine(name="SMA", values=np.array([1.0, 2.0, 3.0]), colour="orange")]
        )

    def test_the_start_date_clips_the_candles(self, service, tmp_path, read_payload):
        written = _open(service, tmp_path, start_date="2024-01-02")

        assert len(_market(read_payload(written))["candles"]) == 2

    def test_the_end_date_clips_the_candles(self, service, tmp_path, read_payload):
        written = _open(service, tmp_path, end_date="2024-01-02")

        assert len(_market(read_payload(written))["candles"]) == 2

    def test_the_window_clips_the_indicators(self, service, tmp_path, read_payload):
        written = _open(
            service, tmp_path, start_date="2024-01-02", end_date="2024-01-02"
        )

        points = _market(read_payload(written))["series"][0]["points"]
        assert [value for _, value in points] == [2.0]

    def test_the_window_clips_the_equity_curve(self, service, tmp_path, read_payload):
        equity = pd.Series(
            [1000.0, 1010.0, 1020.0], index=pd.date_range("2024-01-01", periods=3)
        )

        written = _open(service, tmp_path, equity_curve=equity, start_date="2024-01-03")

        assert [value for _, value in read_payload(written)["equity"]] == [1020.0]

    def test_the_window_clips_the_trades(self, service, tmp_path, read_payload):
        written = _open(service, tmp_path, trades_df=_TRADES, end_date="2024-01-01")

        assert len(_market(read_payload(written))["markers"]) == 2
