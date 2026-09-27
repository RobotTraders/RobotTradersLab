import json

from robottraderslab._core import (
    PerformanceReport,
    ReportGroup,
    ReportRow,
    ReportSection,
    ReportTable,
    ReportTableRow,
)

from .payload import (
    ChartLink,
    ChartPayload,
    ChartSeries,
    MarketView,
    ProfileSetting,
    TradeMarker,
    TradeRow,
)

_COMPACT_SEPARATORS = (",", ":")
_SCRIPT_SAFE_LESS_THAN = "\\u003c"


def encode_payload(payload: ChartPayload) -> str:
    """Serialise a payload into the JSON embedded in a chart document.

    Candles and points travel as positional arrays, so a long backtest stays
    small on the wire and the renderer expands them. The result is safe to
    inline in a `<script>` block.
    """
    document = {
        "title": payload.title,
        "markets": [_encode_market(market) for market in payload.markets],
        "report": _encode_report(payload.report),
        "equity": payload.equity,
        "drawdown": payload.drawdown,
        "drawdownUnit": str(payload.drawdown_unit),
        "links": [_encode_link(link) for link in payload.links],
        "configuration": payload.configuration,
    }
    encoded = json.dumps(document, separators=_COMPACT_SEPARATORS)
    return encoded.replace("<", _SCRIPT_SAFE_LESS_THAN)


def _encode_market(market: MarketView) -> dict[str, object]:
    return {
        "label": market.label,
        "settings": [_encode_setting(setting) for setting in market.settings],
        "candles": market.candles,
        "series": [_encode_series(series) for series in market.series],
        "markers": [_encode_marker(marker) for marker in market.markers],
        "trades": [_encode_trade(trade) for trade in market.trades],
    }


def _encode_series(series: ChartSeries) -> dict[str, object]:
    return {
        "name": series.name,
        "pane": str(series.pane),
        "shape": str(series.shape),
        "color": series.colour,
        "points": series.points,
    }


def _encode_marker(marker: TradeMarker) -> dict[str, object]:
    return {
        "time": marker.time,
        "side": str(marker.side),
        "label": marker.label,
    }


def _encode_trade(trade: TradeRow) -> dict[str, object]:
    return {
        "entry_time": trade.entry_time,
        "exit_time": trade.exit_time,
        "side": trade.side,
        "entry_price": trade.entry_price,
        "exit_price": trade.exit_price,
        "net_pnl": trade.net_pnl,
        "net_pnl_pct": trade.net_pnl_pct,
        "entry_reason": trade.entry_reason,
        "exit_reason": trade.exit_reason,
    }


def _encode_report(report: PerformanceReport) -> dict[str, object]:
    return {
        "headline": [_encode_row(row) for row in report.headline],
        "sections": [_encode_section(section) for section in report.sections],
        "trades": _encode_table(report.trades),
    }


def _encode_section(section: ReportSection) -> dict[str, object]:
    return {
        "title": section.title,
        "rows": [_encode_row(row) for row in section.rows],
    }


def _encode_row(row: ReportRow) -> dict[str, object]:
    return {"label": row.label, "value": row.value}


def _encode_table(table: ReportTable) -> dict[str, object]:
    return {
        "title": table.title,
        "columns": table.columns,
        "groups": [_encode_group(group) for group in table.groups],
    }


def _encode_group(group: ReportGroup) -> dict[str, object]:
    return {
        "title": group.title,
        "rows": [_encode_table_row(row) for row in group.rows],
    }


def _encode_table_row(row: ReportTableRow) -> dict[str, object]:
    return {"label": row.label, "values": row.values}


def _encode_link(link: ChartLink) -> dict[str, object]:
    return {
        "label": link.label,
        "href": link.href,
        "current": link.current,
        "settings": [_encode_setting(setting) for setting in link.settings],
    }


def _encode_setting(setting: ProfileSetting) -> dict[str, object]:
    return {"name": setting.name, "value": setting.value}
