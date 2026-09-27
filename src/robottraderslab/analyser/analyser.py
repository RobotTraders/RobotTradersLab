import logging
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from robottraderslab._core import (
    FRACTION_TO_PERCENT,
    Balance,
    Currency,
    DrawdownUnit,
    Execution,
    OHLCVProviderProtocol,
    PerformanceReport,
    PositionSnapshot,
    Symbol,
    TimeFrame,
)
from robottraderslab.bootstrap import (
    BotConfig,
    ReportConfig,
    load_lightweight_chart_indicators,
)
from robottraderslab.chart import ChartService
from robottraderslab.exceptions import ExchangeCriticalError, StrategyCriticalError
from robottraderslab.reporting import (
    create_run_folder,
    print_filtered_analysis,
    print_long_short_comparison,
    print_performance_summary,
    print_symbol_analysis,
    run_folder_name,
    timestamp_now,
)

from .analysis_inputs import AnalysisInputs
from .analysis_tables import (
    build_filtered_table,
    build_long_short_table,
    build_reason_tables,
    build_symbol_analysis,
)
from .backtest_report import build_backtest_report, resolve_reports_dir
from .equity_metrics import (
    EquityMetricsMixin,
    EquityMetricsResult,
    compute_equity_metrics,
)
from .execution_inputs import build_execution_inputs
from .fill_inputs import build_fill_inputs
from .performance_report import build_performance_report
from .profile_names import name_profiles
from .summary_metrics import SummaryMetrics
from .trade_filter import TradeFilter
from .trade_metrics import (
    TradeDrawdownMetrics,
    TradeMetricsMixin,
    TradeMetricsResult,
    calculate_trade_based_max_drawdown,
    compute_trade_metrics,
)

if TYPE_CHECKING:
    from robottraderslab.plotting import PlottingService

logger = logging.getLogger(__name__)

_REPORT_HTML_NAME = "report.html"


class Analyser(TradeMetricsMixin, EquityMetricsMixin):
    """Every figure a run shows is read from metrics measured once at
    construction, so the report, the prints and the figures agree.
    """

    def __init__(
        self,
        trades: pd.DataFrame,
        open_positions: pd.DataFrame,
        equity_curve: pd.Series,
        trade_metrics: TradeMetricsResult,
        equity_metrics: EquityMetricsResult,
        trade_drawdown: TradeDrawdownMetrics,
        plotting_service: "PlottingService",
        chart_service: ChartService,
        reference_price: pd.Series | None = None,
        reference_symbol_str: str | None = None,
        profiles: Sequence[dict[str, Any]] = (),
    ) -> None:
        self._trades = trades
        self._open_positions = open_positions
        self._equity_curve = equity_curve
        self._trade_metrics = trade_metrics
        self._equity_metrics = equity_metrics
        self._trade_drawdown = trade_drawdown
        self._plotting_service = plotting_service
        self._chart_service = chart_service
        self._reference_price = reference_price
        self._reference_symbol_str = reference_symbol_str
        self._profiles = profiles
        self._saved_run_report: Path | None = None

    @classmethod
    def from_analysis_inputs(
        cls,
        analysis_input: AnalysisInputs,
        ohlcv_provider: OHLCVProviderProtocol,
        report_config: ReportConfig,
        profiles: Sequence[dict[str, Any]] = (),
    ) -> "Analyser":
        """Every measure is taken once here, so the report, the prints and the
        figures agree on the run.

        Args:
            ohlcv_provider: OHLCV source for the interactive chart and the
                intrabar drawdown of the TradingView method.
            profiles: The configurations the strategy ran, each holding a
                symbol, a timeframe and indicator parameters.
        """
        from robottraderslab.plotting import PlottingService

        trades = name_profiles(analysis_input.trades, profiles)
        open_positions = analysis_input.open_positions
        equity_curve = analysis_input.equity_curve

        trade_metrics = compute_trade_metrics(trades)
        equity_metrics = compute_equity_metrics(
            equity_curve,
            trades=trades,
            ohlcv_provider=ohlcv_provider,
            price_series=analysis_input.reference_price,
            reference_name=_reference_display_name(analysis_input.reference_symbol_str),
            risk_free_rate=report_config.risk_free_rate,
            annualization_factor=report_config.annualization_factor,
            calculation_method=report_config.calculation_method,
            initial_balance=analysis_input.initial_balance,
        )

        return cls(
            trades=trades,
            open_positions=open_positions,
            equity_curve=equity_curve,
            trade_metrics=trade_metrics,
            equity_metrics=equity_metrics,
            trade_drawdown=calculate_trade_based_max_drawdown(equity_curve, trades),
            plotting_service=PlottingService(None),
            chart_service=ChartService(
                ohlcv_provider, load_lightweight_chart_indicators
            ),
            reference_price=analysis_input.reference_price,
            reference_symbol_str=analysis_input.reference_symbol_str,
            profiles=profiles,
        )

    @classmethod
    def from_executions(
        cls,
        *,
        executions: Sequence[Execution],
        positions: Mapping[Symbol, PositionSnapshot],
        balances: Mapping[Currency, Balance],
        symbols: Iterable[Symbol],
        since: datetime,
        until: datetime,
        ohlcv_provider: OHLCVProviderProtocol,
        report_config: ReportConfig,
        profiles: Sequence[dict[str, Any]] = (),
    ) -> "Analyser":
        """The venue is the only record a live account keeps. The report covers
        exactly the window the executions were read for, however far back
        they reach.

        Args:
            executions: Every execution read, reaching back far enough to
                establish what was already open at `since` where the venue
                allows it, and covering the window itself.
            positions: The venue's open positions.
            balances: What the venue states the account holds now.
            symbols: The symbols the report covers; they must settle in one
                currency.
            since: Start of the reported window.
            until: End of the reported window.
            ohlcv_provider: Candle source for the interactive chart and the
                reference price.
            profiles: The configurations the strategy runs, each holding a
                symbol, a timeframe and indicator parameters.

        Raises:
            StrategyCriticalError: If the symbols settle in more than one
                currency.
        """
        return cls.from_analysis_inputs(
            build_execution_inputs(
                executions=executions,
                positions=positions,
                balances=balances,
                symbols=symbols,
                since=since,
                until=until,
                ohlcv_provider=ohlcv_provider,
                report_config=report_config,
            ),
            ohlcv_provider,
            report_config,
            profiles=profiles,
        )

    @classmethod
    def from_fills(
        cls,
        fills: pd.DataFrame,
        equity_curve: pd.Series,
        ohlcv_provider: OHLCVProviderProtocol,
        *,
        report_config: ReportConfig,
        profiles: Sequence[dict[str, Any]] = (),
    ) -> "Analyser":
        """The curve's first point is the balance the run opened on, so every
        ratio has a balance to divide by.

        Args:
            fills: One row per fill, indexed by its timestamp, carrying the
                symbol, side, gross and net quantities, price, fee and fill
                type.
            equity_curve: Equity by timestamp.
            ohlcv_provider: Candle source for the interactive chart and the
                reference price.
            profiles: The configurations the strategy ran, each holding a
                symbol, a timeframe and indicator parameters.
        """
        return cls.from_analysis_inputs(
            build_fill_inputs(
                fills, equity_curve, ohlcv_provider, report_config=report_config
            ),
            ohlcv_provider,
            report_config,
            profiles=profiles,
        )

    @property
    def closed_trades_count(self) -> int:
        return self._trade_metrics.total_trades

    @property
    def equity_curve(self) -> pd.Series:
        return self._equity_curve.copy()

    @property
    def open_positions(self) -> pd.DataFrame:
        return self._open_positions.copy()

    @property
    def reference_price(self) -> pd.Series | None:
        return self._reference_price

    @property
    def trades(self) -> pd.DataFrame:
        return self._trades.copy()

    def compute_filtered_metrics(self, trade_filter: TradeFilter) -> TradeMetricsResult:
        return compute_trade_metrics(self._trades, trade_filter)

    def get_summary_metrics(self) -> SummaryMetrics:
        """Return the subset of metrics used by grid search optimisation."""
        return SummaryMetrics(
            sharpe_ratio=self._equity_metrics.sharpe_ratio,
            roi=self._equity_metrics.roi,
            max_drawdown=self._equity_metrics.max_drawdown,
            win_rate=self._trade_metrics.win_rate,
            risk_reward_ratio=self._trade_metrics.risk_reward_ratio,
            closed_trades=self._trade_metrics.total_trades,
        )

    def performance_report(self) -> PerformanceReport:
        """Describe how the run performed, whatever ends up showing it."""
        return build_performance_report(
            self._trade_metrics,
            self._equity_metrics,
            self._trade_drawdown,
            _metrics_per_side(self._trades),
        )

    def plot_candlestick(
        self,
        *,
        indicators_name: str,
        symbol: str,
        timeframe: TimeFrame,
        indicators_params: dict[str, Any],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> Path:
        """The window dates clip what is drawn; the indicators are computed over
        the whole history first.

        Args:
            indicators_name: Entry point name for indicators (e.g., "impulse").
            symbol: Trading symbol to resolve (e.g., "BTC/USDT:USDT").
            indicators_params: Parameters for indicators (e.g., {"ma_length": 20}).
            start_date: Start date for the chart window (ISO format).
            end_date: End date for the chart window (ISO format).
        """
        drawdown_curve, drawdown_unit = self._stated_drawdown_curve()
        return self._chart_service.open_candlestick(
            indicators_name=indicators_name,
            symbol=symbol,
            timeframe=timeframe,
            indicators_params=indicators_params,
            report=self.performance_report(),
            equity_curve=self._equity_curve,
            drawdown_curve=drawdown_curve,
            drawdown_unit=drawdown_unit,
            trades_df=self._trades,
            start_date=start_date,
            end_date=end_date,
        )

    def plot_candlesticks(
        self,
        *,
        indicators_name: str,
        profiles: Sequence[dict[str, Any]] = (),
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[Path]:
        """Opens a saved run's page as it stands, ignoring every argument below;
        on an unsaved run it draws one chart per configuration instead, each
        a document of its own linked to its siblings. A strategy that
        declares several configurations in its config has them picked up
        automatically, so their symbols, timeframes and indicator parameters
        are not retyped here.

        Args:
            indicators_name: Entry point name for indicators (e.g., "impulse").
            profiles: Configurations to draw, each holding a symbol, a
                timeframe and indicator parameters. Defaults to the ones the
                strategy ran. Ignored when the run was saved.
            start_date: Start date for the chart window (ISO format). Ignored
                when the run was saved.
            end_date: End date for the chart window (ISO format). Ignored when
                the run was saved.

        Returns:
            Paths to every chart that was written, in the order requested.
        """
        if self._saved_run_report is not None:
            self._chart_service.open_saved(self._saved_run_report)
            return [self._saved_run_report]

        drawdown_curve, drawdown_unit = self._stated_drawdown_curve()
        return self._chart_service.open_candlesticks(
            indicators_name=indicators_name,
            charts=profiles or self._require_profiles(),
            report=self.performance_report(),
            equity_curve=self._equity_curve,
            drawdown_curve=drawdown_curve,
            drawdown_unit=drawdown_unit,
            trades_df=self._trades,
            start_date=start_date,
            end_date=end_date,
        )

    def plot_cumulative_pnl_by_trade(
        self,
        title: str | None = None,
        filename: str | None = None,
        *,
        show_percentage: bool = True,
    ) -> None:
        """The percentage is a share of the balance the run opened on.

        Raises:
            ValueError: If the run states no opening balance to take the share of.
        """
        self._plotting_service.plot_cumulative_pnl_by_trade(
            self._cumulative_pnl_series(show_percentage=show_percentage),
            title=title,
            filename=filename,
            show_percentage=show_percentage,
        )

    def plot_drawdown(
        self,
        title: str | None = None,
        filename: str | None = None,
    ) -> None:
        drawdown, unit = self._stated_drawdown_curve()
        self._plotting_service.plot_drawdown(
            drawdown=drawdown,
            unit=unit,
            title=title,
            filename=filename,
        )

    def plot_equity_curve(
        self,
        plot_price: bool = True,
        title: str | None = None,
        filename: str | None = None,
    ) -> None:
        self._plotting_service.plot_equity_curve(
            equity_curve=self._equity_metrics.display_equity_curve,
            roi=self._equity_metrics.roi,
            max_drawdown=self._equity_metrics.max_drawdown,
            price_series=self._reference_price if plot_price else None,
            title=title,
            filename=filename,
            price_label=self._reference_symbol_str,
        )

    def plot_monthly_performance(
        self,
        year: int | Literal["all"] | None = None,
        title: str | None = None,
        filename: str | None = None,
    ) -> None:
        plotting = self._plotting_service
        if year == "all":
            years = pd.DatetimeIndex(self._equity_curve.index).year.unique()
            for yr in years:
                plotting.plot_monthly_performance(
                    equity_curve=self._equity_curve,
                    year=yr,
                    title=title,
                    filename=f"{filename}_{yr}" if filename else None,
                )
        else:
            year = year or pd.DatetimeIndex(self._equity_curve.index).year[-1]
            plotting.plot_monthly_performance(
                equity_curve=self._equity_curve,
                year=year,
                title=title,
                filename=filename,
            )

    def plot_performance_summary(
        self,
        filename: str | None = None,
    ) -> None:
        drawdown, drawdown_unit = self._stated_drawdown_curve()
        self._plotting_service.plot_performance_summary(
            equity_curve=self._equity_metrics.display_equity_curve,
            drawdown=drawdown,
            drawdown_unit=drawdown_unit,
            report=self.performance_report(),
            filename=filename,
            price_series=self._reference_price,
            price_label=self._reference_symbol_str,
        )

    def print_filtered_metrics(self, trade_filter: TradeFilter) -> None:
        metrics = self.compute_filtered_metrics(trade_filter)
        table = build_filtered_table(metrics, self._require_initial_equity())
        print_filtered_analysis(table)

    def print_long_short_analysis(self) -> None:
        """
        Raises:
            ValueError: If the run took no long trade, or no short trade.
        """
        table = build_long_short_table(self._trades, self._require_initial_equity())
        print_long_short_comparison(table)

    def print_performance_summary(self) -> None:
        print_performance_summary(
            self.performance_report(), build_reason_tables(self._trades)
        )

    def print_symbol_analysis(self, symbol: str) -> None:
        """
        Raises:
            ValueError: If the run took no trade on the symbol.
        """
        analysis = build_symbol_analysis(
            self._trades, symbol, self._require_initial_equity()
        )
        print_symbol_analysis(analysis)

    def save_run(self, bot_config: BotConfig, indicators_name: str) -> Path:
        """A later `plot_candlesticks` opens that page without rendering again.

        Args:
            bot_config: The configuration this run replayed; its `[report]`
                section names where the folder is written and its config file
                is copied into it and read for the page's configuration view.
            indicators_name: Entry point name for indicators (e.g., "impulse").

        Returns:
            The run's folder.

        Raises:
            ValueError: If `bot_config` carries no configuration file to copy.
        """
        config_file = bot_config.config_file
        if config_file is None:
            raise ValueError(
                "This bot config carries no configuration file to save the run's "
                "folder from."
            )
        timestamp = timestamp_now()
        report = build_backtest_report(
            self.performance_report(), self._trades, self._require_initial_equity()
        )
        directory = create_run_folder(
            report,
            resolve_reports_dir(bot_config),
            run_folder_name(config_file.stem, timestamp),
            config_file,
        )
        if self._profiles:
            self._saved_run_report = self._write_run_report(
                directory, config_file, indicators_name, timestamp
            )
        return directory

    def _cumulative_pnl_series(self, *, show_percentage: bool) -> pd.Series:
        chronological_trades = self._trades.sort_values(
            "entry_time", kind="stable"
        ).reset_index(drop=True)
        cumulative_pnl = chronological_trades["net_pnl"].cumsum()
        if show_percentage:
            return cumulative_pnl / self._require_initial_equity() * FRACTION_TO_PERCENT
        return cumulative_pnl

    def _require_initial_equity(self) -> float:
        initial_equity = self._equity_metrics.initial_equity
        if initial_equity is None:
            raise ValueError(
                "This measurement is a share of the initial equity; the "
                "report was built without one to divide by."
            )
        return initial_equity

    def _require_profiles(self) -> Sequence[dict[str, Any]]:
        if not self._profiles:
            raise RuntimeError(
                "No strategy profiles available. Pass profiles to "
                "plot_candlesticks, or use BacktestOutputs.create_analyser() "
                "with a strategy that declares them in its config."
            )
        return self._profiles

    def _stated_drawdown_curve(self) -> tuple[pd.Series, DrawdownUnit]:
        shares = self._equity_metrics.drawdown_percentage
        if shares is None:
            return self._equity_metrics.absolute_drawdown, DrawdownUnit.CURRENCY
        return shares * FRACTION_TO_PERCENT, DrawdownUnit.PERCENT

    def _write_run_report(
        self, directory: Path, config_file: Path, indicators_name: str, timestamp: str
    ) -> Path | None:
        """A chart failure never costs the folder its configuration and its
        report, since drawing is a presentation detail and the run's figures
        are already measured and written.
        """
        drawdown_curve, drawdown_unit = self._stated_drawdown_curve()
        try:
            return self._chart_service.write_report(
                indicators_name=indicators_name,
                charts=self._profiles,
                report=self.performance_report(),
                equity_curve=self._equity_curve,
                drawdown_curve=drawdown_curve,
                drawdown_unit=drawdown_unit,
                trades_df=self._trades,
                destination=directory / _REPORT_HTML_NAME,
                configuration=config_file.read_text(encoding="utf-8"),
                run_label=f"{config_file.stem} · {timestamp}",
            )
        except (Exception, ExchangeCriticalError, StrategyCriticalError):
            logger.warning(
                "No chart page written for `%s`", indicators_name, exc_info=True
            )
            return None


def _reference_display_name(symbol_str: str | None) -> str | None:
    if symbol_str is None:
        return None
    try:
        return str(Symbol.create(symbol_str).base)
    except ValueError:
        return symbol_str


def _metrics_per_side(trades: pd.DataFrame) -> list[TradeMetricsResult]:
    if trades.empty or "side" not in trades.columns:
        return []

    taken = [
        trade_filter
        for trade_filter in (TradeFilter(side="long"), TradeFilter(side="short"))
        if not trade_filter.apply_filter(trades).empty
    ]
    return [compute_trade_metrics(trades, trade_filter) for trade_filter in taken]
