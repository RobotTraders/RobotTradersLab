import pandas as pd
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
from robottraderslab.plotting import PlottingService


@pytest.fixture
def equity_curve() -> pd.Series:
    days = pd.Series(
        range(90), index=pd.date_range("2024-01-01", periods=90, freq="D"), dtype=float
    )
    return 1000.0 + days * 2.0 - (days % 9) * 6.0


@pytest.fixture
def drawdown(equity_curve: pd.Series) -> pd.Series:
    peak = equity_curve.cummax()
    return (equity_curve - peak) / peak * 100.0


@pytest.fixture
def report() -> PerformanceReport:
    return PerformanceReport(
        headline=[ReportRow(label="PnL", value="$99.00 (9.90%)")],
        sections=[
            ReportSection(
                title="Overview",
                rows=[
                    ReportRow(label="Period start", value="2024-01-01 00:00"),
                    ReportRow(label="PnL", value="$99.00 (9.90%)"),
                ],
            ),
            ReportSection(
                title="Risk & ratios",
                rows=[ReportRow(label="Sharpe ratio", value="1.23")],
            ),
        ],
        trades=ReportTable(
            title="Trades",
            columns=["All", "Long"],
            groups=[
                ReportGroup(
                    title=None,
                    rows=[ReportTableRow(label="Total trades", values=["9", "5"])],
                ),
                ReportGroup(
                    title="Fees",
                    rows=[
                        ReportTableRow(label="Total fees", values=["$9.90", "$5.50"])
                    ],
                ),
            ],
        ),
    )


@pytest.fixture
def service(tmp_path) -> PlottingService:
    return PlottingService(save_to_path=tmp_path)


class TestPlottingService:
    def test_the_equity_curve_is_saved_under_its_default_name(
        self, service, tmp_path, equity_curve
    ):
        service.plot_equity_curve(equity_curve, roi=0.099, max_drawdown=-0.012)

        assert (tmp_path / "equity_curve.png").exists()

    def test_an_equity_curve_measured_from_no_balance(
        self, service, tmp_path, equity_curve
    ):
        service.plot_equity_curve(equity_curve, roi=None, max_drawdown=None)

        assert (tmp_path / "equity_curve.png").exists()

    def test_an_equity_curve_over_a_reference_price(
        self, service, tmp_path, equity_curve
    ):
        service.plot_equity_curve(
            equity_curve,
            roi=0.099,
            max_drawdown=-0.012,
            price_series=equity_curve * 40.0,
            filename="overlay",
            price_label="BTC/USDT:USDT",
        )

        assert (tmp_path / "overlay.png").exists()

    def test_the_drawdown_is_saved_under_its_default_name(
        self, service, tmp_path, drawdown
    ):
        service.plot_drawdown(drawdown, unit=DrawdownUnit.PERCENT)

        assert (tmp_path / "drawdown.png").exists()

    def test_a_drawdown_counted_in_currency(self, service, tmp_path, equity_curve):
        service.plot_drawdown(
            equity_curve - equity_curve.cummax(),
            unit=DrawdownUnit.CURRENCY,
            filename="currency",
        )

        assert (tmp_path / "currency.png").exists()

    def test_the_cumulative_pnl_is_saved_under_its_default_name(
        self, service, tmp_path
    ):
        service.plot_cumulative_pnl_by_trade(
            pd.Series([1.0, 0.5, 2.5]), show_percentage=True
        )

        assert (tmp_path / "cumulative_pnl_by_trade.png").exists()

    def test_the_monthly_performance_is_named_after_its_year(
        self, service, tmp_path, equity_curve
    ):
        service.plot_monthly_performance(equity_curve, year=2024)

        assert (tmp_path / "monthly_performance_2024.png").exists()

    def test_a_year_the_run_never_reached_cannot_be_drawn(self, service, equity_curve):
        with pytest.raises(ValueError, match="no point in 2019"):
            service.plot_monthly_performance(equity_curve, year=2019)

    def test_a_reading_on_new_year_midnight_closes_the_year_before(self, service):
        closing_2023 = pd.Series(
            [100.0, 110.0], index=pd.to_datetime(["2023-12-31", "2024-01-01"])
        )

        with pytest.raises(ValueError, match="no point in 2024"):
            service.plot_monthly_performance(closing_2023, year=2024)

    def test_the_summary_is_saved_under_its_default_name(
        self, service, tmp_path, equity_curve, drawdown, report
    ):
        service.plot_performance_summary(
            equity_curve,
            drawdown=drawdown,
            drawdown_unit=DrawdownUnit.PERCENT,
            report=report,
            price_series=equity_curve * 40.0,
            price_label="BTC/USDT:USDT",
        )

        assert (tmp_path / "performance_summary.png").exists()

    def test_a_summary_measured_from_no_balance(
        self, service, tmp_path, equity_curve, report
    ):
        service.plot_performance_summary(
            equity_curve,
            drawdown=equity_curve - equity_curve.cummax(),
            drawdown_unit=DrawdownUnit.CURRENCY,
            report=report,
        )

        assert (tmp_path / "performance_summary.png").exists()

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_without_a_path_the_figure_is_shown_and_nothing_is_written(
        self, tmp_path, monkeypatch, drawdown
    ):
        monkeypatch.chdir(tmp_path)

        PlottingService(save_to_path=None).plot_drawdown(
            drawdown, unit=DrawdownUnit.PERCENT
        )

        assert list(tmp_path.iterdir()) == []
