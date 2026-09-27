from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from robottraderslab._core import (
    FRACTION_TO_PERCENT,
    ChartLine,
    DrawdownUnit,
    PerformanceReport,
)

from .payload import (
    ChartLink,
    ChartPayload,
    ChartSeries,
    MarkerSide,
    MarketView,
    Pane,
    ProfileSetting,
    Rows,
    Shape,
    TradeMarker,
    TradeRow,
)

_CANDLE_COLUMNS = ["open", "high", "low", "close"]
_TRADE_VALUE_COLUMNS = ["entry_price", "exit_price", "net_pnl", "net_pnl_pct"]
_ENTRY_LABEL = "entry"
_EXIT_LABEL = "exit"
_ENTRY_DIRECTIONS = {"long": MarkerSide.BUY, "short": MarkerSide.SELL}
_EXIT_DIRECTIONS = {"long": MarkerSide.SELL, "short": MarkerSide.BUY}


def build_market_view(
    *,
    label: str,
    settings: list[ProfileSetting],
    ohlcv: pd.DataFrame,
    lines: list[ChartLine],
    trades: pd.DataFrame,
) -> MarketView:
    """
    Args:
        label: Display name for this market, e.g. "BTC/USDT:USDT · 1h · alpha".
        settings: The configured values shown when this market is the
            current one of several.
        ohlcv: Frame indexed by time with open, high, low and close columns.
        lines: The strategy's chart lines, each carrying one value per candle.
        trades: Trades for this market, already restricted to the window.
    """
    times = _epoch_seconds(ohlcv.index)
    return MarketView(
        label=label,
        settings=settings,
        candles=_rows(times, ohlcv[_CANDLE_COLUMNS].to_numpy(dtype=float)),
        series=[_build_series(line, times) for line in lines],
        markers=_build_markers(trades),
        trades=_build_trade_rows(trades),
    )


def build_payload(
    *,
    title: str,
    markets: list[MarketView],
    report: PerformanceReport,
    equity_curve: pd.Series,
    drawdown_curve: pd.Series,
    drawdown_unit: DrawdownUnit,
    links: list[ChartLink],
    configuration: str = "",
) -> ChartPayload:
    """
    Args:
        title: Heading shown above the charts.
        markets: The markets this page draws, switched between in place when
            there is more than one.
        report: Performance figures measured and formatted by the caller, since
            what they cover is the caller's to decide.
        equity_curve: Account equity over time, drawn on its own pane.
        drawdown_curve: How far the account sat below its own peak, already in
            the values to be drawn, on its own pane.
        drawdown_unit: What those values are counted in, so the pane's axis
            labels them the way they were measured.
        links: Sibling pages to navigate to, empty when this page stands alone.
        configuration: The TOML text of the run this page was made from, shown
            beside the trades. Empty when there is none to show.
    """
    return ChartPayload(
        title=title,
        markets=markets,
        report=report,
        equity=_build_curve(equity_curve),
        drawdown=_build_curve(drawdown_curve),
        drawdown_unit=drawdown_unit,
        links=links,
        configuration=configuration,
    )


def _build_series(line: ChartLine, times: npt.NDArray[np.int64]) -> ChartSeries:
    """A line that needs more history than the window holds leaves its first
    values empty, and those carry no point.
    """
    values = line.values.astype(float)
    drawn = ~np.isnan(values)
    return ChartSeries(
        name=line.name,
        pane=Pane(line.pane),
        shape=Shape(line.shape),
        colour=line.colour,
        points=_rows(times[drawn], values[drawn]),
    )


def _build_markers(trades: pd.DataFrame) -> list[TradeMarker]:
    """A position still running belongs on the chart where it opened, so exits
    are drawn from the closed trades alone.
    """
    closed = trades[trades["exit_time"].notna()]
    markers = _markers_of(trades, _ENTRY_LABEL, _ENTRY_DIRECTIONS)
    markers.extend(_markers_of(closed, _EXIT_LABEL, _EXIT_DIRECTIONS))
    return sorted(markers, key=lambda marker: marker.time)


def _markers_of(
    trades: pd.DataFrame, phase: str, directions: dict[str, MarkerSide]
) -> list[TradeMarker]:
    """A marker points the way the money moved, so an exit points against its
    trade: closing a long sells, closing a short buys.
    """
    return [
        TradeMarker(time=time, side=directions[trade_side], label=_label(reason, phase))
        for time, reason, trade_side in zip(
            _epoch_seconds(trades[f"{phase}_time"]).tolist(),
            trades[f"{phase}_reason"],
            trades["side"],
        )
    ]


def _build_trade_rows(trades: pd.DataFrame) -> list[TradeRow]:
    return [
        _build_trade_row(entry_time, exit_time, side, entry_reason, exit_reason, values)
        for entry_time, exit_time, side, entry_reason, exit_reason, values in zip(
            _epoch_seconds(trades["entry_time"]).tolist(),
            _exit_epoch_seconds(trades["exit_time"]),
            trades["side"],
            trades["entry_reason"],
            trades["exit_reason"],
            trades[_TRADE_VALUE_COLUMNS].to_numpy(dtype=float).tolist(),
        )
    ]


def _build_trade_row(
    entry_time: int,
    exit_time: int | None,
    side: str,
    entry_reason: str | float | None,
    exit_reason: str | float | None,
    values: list[float],
) -> TradeRow:
    entry_price, exit_price, net_pnl, net_pnl_pct = values
    closed = exit_time is not None
    return TradeRow(
        entry_time=entry_time,
        exit_time=exit_time,
        side=side,
        entry_price=entry_price,
        exit_price=exit_price if closed else None,
        net_pnl=net_pnl,
        net_pnl_pct=net_pnl_pct * FRACTION_TO_PERCENT,
        entry_reason=_label(entry_reason, _ENTRY_LABEL),
        exit_reason=_label(exit_reason, _EXIT_LABEL) if closed else None,
    )


def _exit_epoch_seconds(exit_times: pd.Series) -> list[int | None]:
    return [
        seconds if closed else None
        for seconds, closed in zip(
            _epoch_seconds(exit_times).tolist(), exit_times.notna().tolist()
        )
    ]


def _build_curve(curve: pd.Series) -> Rows:
    return _rows(_epoch_seconds(curve.index), curve.to_numpy(dtype=float))


def _rows(times: npt.NDArray[np.int64], values: npt.NDArray[np.float64]) -> Rows:
    """The times stay whole seconds, so the JSON carries them as integers."""
    columns = np.column_stack([times.astype(object), values.astype(object)])
    return cast(Rows, columns.tolist())


def _epoch_seconds(times: pd.Index | pd.Series) -> npt.NDArray[np.int64]:
    return np.asarray(
        pd.DatetimeIndex(times).as_unit("s").astype("int64"), dtype=np.int64
    )


def _label(reason: str | float | None, fallback: str) -> str:
    return fallback if pd.isna(reason) else str(reason)
