import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from robottraderslab._core import (
    Candles,
    ChartLine,
    DrawdownUnit,
    OHLCVProviderProtocol,
    PerformanceReport,
    Symbol,
    TimeFrame,
    profile_identity,
)

from .builders import build_market_view, build_payload
from .payload import ChartLink, MarketView, ProfileSetting
from .renderer import open_in_browser, write_charts

type ChartLinesLoader = Callable[[str, Candles, dict[str, Any]], list[ChartLine]]

_SYMBOL_KEY = "symbol"
_TIMEFRAME_KEY = "timeframe"
_TAG_KEY = "tag"
_PROFILE_NAME_KEY = "profile_name"

_LABEL_SEPARATOR = " · "
_WORD_SEPARATOR = "_"


@dataclass(frozen=True, slots=True)
class _ChartSpec:
    """One chart to draw, its market resolved once for every use it gets."""

    symbol: Symbol
    timeframe: TimeFrame
    tag: str | None
    params: dict[str, Any]

    @classmethod
    def of(cls, chart: dict[str, Any]) -> "_ChartSpec":
        return cls(
            Symbol.create(chart[_SYMBOL_KEY]),
            chart[_TIMEFRAME_KEY],
            chart.get(_TAG_KEY),
            chart,
        )

    @property
    def profile_name(self) -> str:
        return profile_identity(str(self.symbol), self.timeframe, self.tag or "")


@dataclass(frozen=True, slots=True)
class _Sibling:
    """What every chart of a run says about one of them, before knowing
    which is open.
    """

    label: str
    href: str
    settings: list[ProfileSetting]

    def link(self, *, current: bool) -> ChartLink:
        return ChartLink(
            label=self.label, href=self.href, current=current, settings=self.settings
        )


class ChartService:
    """Draws a strategy's indicators and executed trades over its candles."""

    def __init__(
        self, ohlcv_provider: OHLCVProviderProtocol, chart_lines: ChartLinesLoader
    ) -> None:
        """
        Args:
            chart_lines: Resolves a strategy's name and parameters into the
                lines it draws over its candles.
        """
        self._ohlcv_provider = ohlcv_provider
        self._chart_lines = chart_lines

    def open_candlestick(
        self,
        *,
        indicators_name: str,
        symbol: str,
        timeframe: TimeFrame,
        indicators_params: dict[str, object],
        report: PerformanceReport,
        equity_curve: pd.Series,
        drawdown_curve: pd.Series,
        drawdown_unit: DrawdownUnit,
        trades_df: pd.DataFrame,
        start_date: str | None = None,
        end_date: str | None = None,
        destination_dir: Path | None = None,
    ) -> Path:
        """Open an interactive chart of the strategy over the candles it traded.

        Args:
            indicators_name: Entry point name for indicators (e.g., "impulse").
            symbol: Symbol string (e.g., "BTC/USDT:USDT").
            timeframe: Timeframe for both data and indicators.
            indicators_params: Parameters to compute the strategy indicators.
            report: Performance figures already measured and formatted.
            equity_curve: Account equity over time.
            drawdown_curve: How far the account sat below its own peak.
            drawdown_unit: What the drawdown values are counted in.
            trades_df: The run's trades, drawn as entry and exit markers.
            start_date: Start date for the chart window (ISO format).
            end_date: End date for the chart window (ISO format).
            destination_dir: Directory to write the chart into. Defaults to a
                fresh temporary directory, since a chart is opened and never kept.

        Returns:
            Path to the chart that was opened.
        """
        chart = _ChartSpec(Symbol.create(symbol), timeframe, None, indicators_params)
        market = self._build_market_view(
            indicators_name, chart, trades_df, None, start_date, end_date
        )
        payload = build_payload(
            title=_display_title(indicators_name, None),
            markets=[market],
            report=report,
            equity_curve=equity_curve[
                _window_mask(equity_curve.index, start_date, end_date)
            ],
            drawdown_curve=drawdown_curve[
                _window_mask(drawdown_curve.index, start_date, end_date)
            ],
            drawdown_unit=drawdown_unit,
            links=[],
        )
        directory = destination_dir or _fresh_directory()
        destination = directory / _chart_filename(
            chart.symbol, chart.timeframe, chart.tag
        )
        written = write_charts([payload], [destination])[0]
        open_in_browser(written)
        return written

    def open_candlesticks(
        self,
        *,
        indicators_name: str,
        charts: Sequence[dict[str, Any]],
        report: PerformanceReport,
        equity_curve: pd.Series,
        drawdown_curve: pd.Series,
        drawdown_unit: DrawdownUnit,
        trades_df: pd.DataFrame,
        start_date: str | None = None,
        end_date: str | None = None,
        destination_dir: Path | None = None,
    ) -> list[Path]:
        """
        Args:
            indicators_name: Entry point name for indicators (e.g., "impulse").
            charts: One mapping per chart to draw, each holding a symbol, a
                timeframe and the indicator parameters to draw them with.
            report: Performance figures already measured and formatted.
            equity_curve: Account equity over time.
            drawdown_curve: How far the account sat below its own peak.
            drawdown_unit: What the drawdown values are counted in.
            trades_df: The run's trades, drawn as entry and exit markers.
            start_date: Start date for the chart window (ISO format).
            end_date: End date for the chart window (ISO format).
            destination_dir: Directory every chart is written into. Defaults
                to a fresh temporary directory, since the pages are opened
                and never kept.

        Returns:
            Paths to every chart that was written, in the order requested.
        """
        written = self.write_candlesticks(
            indicators_name=indicators_name,
            charts=charts,
            report=report,
            equity_curve=equity_curve,
            drawdown_curve=drawdown_curve,
            drawdown_unit=drawdown_unit,
            trades_df=trades_df,
            destination_dir=destination_dir or _fresh_directory(),
            start_date=start_date,
            end_date=end_date,
        )
        open_in_browser(written[0])
        return written

    def open_saved(self, chart: Path) -> None:
        open_in_browser(chart)

    def write_candlesticks(
        self,
        *,
        indicators_name: str,
        charts: Sequence[dict[str, Any]],
        report: PerformanceReport,
        equity_curve: pd.Series,
        drawdown_curve: pd.Series,
        drawdown_unit: DrawdownUnit,
        trades_df: pd.DataFrame,
        destination_dir: Path,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[Path]:
        """Draw each requested chart into a directory without opening any of them.

        Each chart becomes a document of its own, linked to its siblings, so a
        strategy trading many symbols stays navigable without loading every
        symbol's candles at once.

        Args:
            indicators_name: Entry point name for indicators (e.g., "impulse").
            charts: One mapping per chart to draw, each holding a symbol, a
                timeframe and the indicator parameters to draw them with.
            report: Performance figures already measured and formatted.
            equity_curve: Account equity over time.
            drawdown_curve: How far the account sat below its own peak.
            drawdown_unit: What the drawdown values are counted in.
            trades_df: The run's trades, drawn as entry and exit markers.
            destination_dir: Directory every chart is written into.
            start_date: Start date for the chart window (ISO format).
            end_date: End date for the chart window (ISO format).

        Returns:
            Paths to every chart that was written, in the order requested.
        """
        specs = [_ChartSpec.of(chart) for chart in charts]
        destinations = [
            destination_dir / _chart_filename(spec.symbol, spec.timeframe, spec.tag)
            for spec in specs
        ]
        siblings = _siblings(charts, destinations)
        title = _display_title(indicators_name, None)
        payloads = [
            build_payload(
                title=title,
                markets=[
                    self._build_market_view(
                        indicators_name,
                        spec,
                        trades_df,
                        spec.profile_name,
                        start_date,
                        end_date,
                    )
                ],
                report=report,
                equity_curve=equity_curve[
                    _window_mask(equity_curve.index, start_date, end_date)
                ],
                drawdown_curve=drawdown_curve[
                    _window_mask(drawdown_curve.index, start_date, end_date)
                ],
                drawdown_unit=drawdown_unit,
                links=[
                    sibling.link(current=position == index)
                    for position, sibling in enumerate(siblings)
                ],
            )
            for index, spec in enumerate(specs)
        ]
        return write_charts(payloads, destinations)

    def write_report(
        self,
        *,
        indicators_name: str,
        charts: Sequence[dict[str, Any]],
        report: PerformanceReport,
        equity_curve: pd.Series,
        drawdown_curve: pd.Series,
        drawdown_unit: DrawdownUnit,
        trades_df: pd.DataFrame,
        destination: Path,
        run_label: str,
        configuration: str = "",
    ) -> Path:
        """Draw every requested market into one page, switching between them
        in place.

        Args:
            indicators_name: Entry point name for indicators (e.g., "impulse").
            charts: One mapping per market to draw, each holding a symbol, a
                timeframe and the indicator parameters to draw it with.
            report: Performance figures already measured and formatted.
            equity_curve: Account equity over time.
            drawdown_curve: How far the account sat below its own peak.
            drawdown_unit: What the drawdown values are counted in.
            trades_df: The run's trades, drawn as entry and exit markers.
            destination: File the page is written to.
            run_label: The page's title.
            configuration: The TOML text of the run this page was made from,
                shown beside the trades. Empty when there is none.
        """
        specs = [_ChartSpec.of(chart) for chart in charts]
        payload = build_payload(
            title=_display_title(indicators_name, run_label),
            markets=[
                self._build_market_view(
                    indicators_name, spec, trades_df, spec.profile_name, None, None
                )
                for spec in specs
            ],
            report=report,
            equity_curve=equity_curve,
            drawdown_curve=drawdown_curve,
            drawdown_unit=drawdown_unit,
            links=[],
            configuration=configuration,
        )
        return write_charts([payload], [destination])[0]

    def _build_market_view(
        self,
        indicators_name: str,
        chart: _ChartSpec,
        trades_df: pd.DataFrame,
        profile_name: str | None,
        start_date: str | None,
        end_date: str | None,
    ) -> MarketView:
        ohlcv, lines = self._windowed_market(
            indicators_name, chart, start_date, end_date
        )
        return build_market_view(
            label=_market_label(chart.symbol, chart.timeframe, chart.tag),
            settings=_profile_settings(chart.params),
            ohlcv=ohlcv,
            lines=lines,
            trades=_select_symbol_trades(
                trades_df, chart.symbol, profile_name, start_date, end_date
            ),
        )

    def _windowed_market(
        self,
        indicators_name: str,
        chart: _ChartSpec,
        start_date: str | None,
        end_date: str | None,
    ) -> tuple[pd.DataFrame, list[ChartLine]]:
        """Indicators are computed over the full history first, so a window
        opening mid-series still shows warmed-up values.
        """
        ohlcv = self._ohlcv_provider.fetch_ohlcv(chart.symbol, chart.timeframe)
        lines = self._chart_lines(indicators_name, _to_candles(ohlcv), chart.params)
        window = _window_mask(ohlcv.index, start_date, end_date)
        return ohlcv[window], [
            replace(line, values=line.values[window]) for line in lines
        ]


def _to_candles(ohlcv: pd.DataFrame) -> Candles:
    return Candles(
        open=ohlcv["open"].to_numpy(),
        high=ohlcv["high"].to_numpy(),
        low=ohlcv["low"].to_numpy(),
        close=ohlcv["close"].to_numpy(),
        volume=ohlcv["volume"].to_numpy(),
    )


def _window_mask(
    index: pd.Index, start_date: str | None, end_date: str | None
) -> npt.NDArray[np.bool_]:
    """A chart line carries one value per candle, so the same mask narrows both
    and keeps them aligned.
    """
    inside = np.ones(len(index), dtype=bool)
    if start_date is not None:
        inside &= index >= start_date
    if end_date is not None:
        inside &= index <= end_date
    return inside


def _select_symbol_trades(
    trades_df: pd.DataFrame,
    symbol: Symbol,
    profile_name: str | None,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    """A run whose trades were never named after a profile draws every trade of
    the symbol on each of its charts.
    """
    symbol_trades = trades_df[trades_df["symbol"] == symbol]
    if profile_name is not None and _PROFILE_NAME_KEY in symbol_trades.columns:
        symbol_trades = symbol_trades[symbol_trades[_PROFILE_NAME_KEY] == profile_name]
    entry_times = pd.Index(symbol_trades["entry_time"])
    return symbol_trades[_window_mask(entry_times, start_date, end_date)]


def _siblings(
    charts: Sequence[dict[str, Any]], destinations: Sequence[Path]
) -> list[_Sibling]:
    if len(charts) == 1:
        return []
    return [
        _Sibling(_chart_label(chart), destination.name, _profile_settings(chart))
        for chart, destination in zip(charts, destinations, strict=True)
    ]


def _profile_settings(chart: dict[str, Any]) -> list[ProfileSetting]:
    return [
        ProfileSetting(name=_setting_name(key), value=str(value))
        for key, value in chart.items()
    ]


def _setting_name(key: str) -> str:
    return key.replace(_WORD_SEPARATOR, " ").capitalize()


def _chart_label(chart: dict[str, Any]) -> str:
    return _market_label(chart[_SYMBOL_KEY], chart[_TIMEFRAME_KEY], chart.get(_TAG_KEY))


def _market_label(symbol: object, timeframe: object, tag: str | None) -> str:
    parts = [str(symbol), str(timeframe)]
    if tag:
        parts.append(str(tag))
    return _LABEL_SEPARATOR.join(parts)


def _chart_filename(symbol: Symbol, timeframe: TimeFrame, tag: str | None) -> str:
    parts = [_slugify(str(symbol)), str(timeframe)]
    if tag:
        parts.append(_slugify(str(tag)))
    return "_".join(parts) + ".html"


def _display_title(indicators_name: str, run_label: str | None) -> str:
    if run_label is not None:
        return run_label
    return indicators_name.replace(_WORD_SEPARATOR, " ").title()


def _slugify(symbol: str) -> str:
    return "".join(
        character if character.isalnum() else "-" for character in symbol
    ).strip("-")


def _fresh_directory() -> Path:
    """A page that was never saved is opened once and never kept, so each
    render gets a directory of its own.
    """
    return Path(tempfile.mkdtemp(prefix="robottraderslab-chart-"))
