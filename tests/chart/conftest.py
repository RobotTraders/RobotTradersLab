import pytest

from robottraderslab._core import (
    DrawdownUnit,
    PerformanceReport,
    ReportGroup,
    ReportRow,
    ReportSection,
    ReportTable,
    ReportTableRow,
)
from robottraderslab.chart.payload import (
    ChartLink,
    ChartPayload,
    ChartSeries,
    MarkerSide,
    MarketView,
    Pane,
    ProfileSetting,
    Shape,
    TradeMarker,
    TradeRow,
)


@pytest.fixture
def empty_report() -> PerformanceReport:
    return PerformanceReport(
        headline=[],
        sections=[],
        trades=ReportTable(title="Trades", columns=["All"], groups=[]),
    )


@pytest.fixture
def market() -> MarketView:
    return MarketView(
        label="BTC/USDT:USDT · 1h",
        settings=[ProfileSetting(name="Trix length", value="31")],
        candles=[[1735689600, 1.0, 4.0, 0.5, 3.0]],
        series=[
            ChartSeries(
                name="TRIX",
                pane=Pane.SEPARATE,
                shape=Shape.LINE,
                colour="blue",
                points=[[1735689600, 2.5]],
            )
        ],
        markers=[TradeMarker(time=1735689600, side=MarkerSide.BUY, label="long")],
        trades=[
            TradeRow(
                entry_time=1735689600,
                exit_time=1735693200,
                side="long",
                entry_price=1.0,
                exit_price=3.0,
                net_pnl=2.0,
                net_pnl_pct=200.0,
                entry_reason="ma_cross",
                exit_reason="take_profit",
            )
        ],
    )


@pytest.fixture
def payload(market: MarketView) -> ChartPayload:
    return ChartPayload(
        title="Impulse backtest",
        markets=[market],
        report=PerformanceReport(
            headline=[ReportRow(label="Performance", value="9.50%")],
            sections=[
                ReportSection(
                    title="Returns",
                    rows=[
                        ReportRow(label="Return", value="9.50%"),
                        ReportRow(label="Profit factor", value="∞"),
                    ],
                )
            ],
            trades=ReportTable(
                title="Trades",
                columns=["All", "Long"],
                groups=[
                    ReportGroup(
                        title=None,
                        rows=[ReportTableRow(label="Total trades", values=["4", "2"])],
                    ),
                    ReportGroup(
                        title="Fees",
                        rows=[
                            ReportTableRow(
                                label="Total fees", values=["$12.40", "$6.20"]
                            )
                        ],
                    ),
                ],
            ),
        ),
        equity=[[1735689600, 1000.0]],
        drawdown=[[1735689600, -5.0]],
        drawdown_unit=DrawdownUnit.PERCENT,
        links=[
            ChartLink(
                label="BTC/USDT:USDT · 1h · alpha",
                href="BTC-USDT-USDT_1h_alpha.html",
                current=True,
                settings=[ProfileSetting(name="Trix length", value="31")],
            ),
            ChartLink(
                label="BTC/USDT:USDT · 1h · beta",
                href="BTC-USDT-USDT_1h_beta.html",
                current=False,
                settings=[ProfileSetting(name="Trix length", value="43")],
            ),
        ],
        configuration='[strategy]\nstrategy_class = "impulse"\n',
    )
