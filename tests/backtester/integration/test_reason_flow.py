import asyncio
import io
from collections.abc import Callable
from datetime import datetime

import pandas as pd
import pytest

from robottraderslab import Symbol, TimeFrame
from robottraderslab._core import PlacementReserve
from robottraderslab.analyser import create_trade_aggregation
from robottraderslab.backtester.backtester import Backtester
from robottraderslab.backtester.simulator import (
    CacheFillRecorder,
    FuturesSimulationEngine,
    SimulatedFuturesExchange,
    fee_model_for,
)
from robottraderslab.exceptions import StrategyCriticalError
from robottraderslab.futures import FuturesAccount
from robottraderslab.ohlcv_provider import CSVOHLCVProvider
from robottraderslab.strategies import (
    AccountSnapshots,
    BookKeeper,
    Execution,
    OHLCVs,
    StrategyProtocol,
    StrategyRequirements,
)

BTCUSDT_PERP = Symbol.create("BTC/USDT:USDT")


@pytest.fixture
def csv_3_daily_prices():
    return io.StringIO("""\
date,open,high,low,close,volume
2024-01-01T00:00:00+00:00,100,110,100,110,10
2024-01-02T00:00:00+00:00,110,120,110,120,10
2024-01-03T00:00:00+00:00,120,130,120,130,10
2024-01-04T00:00:00+00:00,130,140,130,140,10
""")


@pytest.fixture
def fill_recorder():
    return CacheFillRecorder()


@pytest.fixture
def simulation_engine(fill_recorder):
    from robottraderslab.backtester.simulator import FeeRates

    return FuturesSimulationEngine(
        initial_balance={"USDT": 10_000.0},
        fee_rates=FeeRates(maker=0.0, taker=0.0),
        fill_recorder=fill_recorder,
        margin_currency="USDT",
        fee_model=fee_model_for("cost", PlacementReserve()),
    )


@pytest.fixture
def simulated_exchange(simulation_engine, fill_recorder):
    return SimulatedFuturesExchange(simulation_engine, fill_recorder)


@pytest.fixture
def account(simulated_exchange):
    return FuturesAccount(simulated_exchange)


class StrategyFixture(StrategyProtocol):
    market_type = "futures"

    def __init__(self, csv: io.StringIO, handler: Callable):
        self.ohlcv_provider = CSVOHLCVProvider(
            file=csv,
            symbol=str(BTCUSDT_PERP),
            timeframe="1d",
        )
        self.handler = handler
        self.count = 0

    def book_trading_actions(
        self,
        ohlcvs: OHLCVs,
        account_snapshots: AccountSnapshots,
        timestamp: datetime,
        bookkeeper: BookKeeper,
        triggered_timeframes: set[TimeFrame],
    ) -> None:
        self.handler(self.count, timestamp, bookkeeper, triggered_timeframes)
        self.count += 1

    async def setup(self, requirements: StrategyRequirements) -> None:
        requirements.ohlcv.add(BTCUSDT_PERP, "1d")

    def generate_trading_signals(self, ohlcvs: OHLCVs) -> None:
        pass


class DescribingStrategyFixture(StrategyFixture):
    def __init__(
        self,
        csv: io.StringIO,
        handler: Callable,
        account: FuturesAccount,
        describe_fills: Callable[[Execution], str | None],
    ):
        super().__init__(csv, handler)
        self.account = account
        self.describe_fills = describe_fills

    async def setup(self, requirements: StrategyRequirements) -> None:
        await super().setup(requirements)
        requirements.account.add(self.account, describe_fills=self.describe_fills)


class TestReasonFlow:
    """
    Integration tests for reason field tracking through the entire system.

    Tests the complete flow from order placement with reason -> BookKeeper ->
    FuturesSimulationEngine -> FillRecorder -> TradeRecord aggregation.
    """

    @pytest.fixture(autouse=True)
    def _setup(self, account, simulation_engine, fill_recorder):
        self.account: FuturesAccount = account
        self.simulation_engine: FuturesSimulationEngine = simulation_engine
        self.fill_recorder: CacheFillRecorder = fill_recorder

    def _create_backtester(self, csv, handler) -> Backtester:
        strategy = StrategyFixture(csv, handler)
        return Backtester(
            strategy,
            self.simulation_engine,
            strategy.ohlcv_provider,
            self.fill_recorder,
            "2024-01-01",
            "2024-01-04",
        )

    def test_entry_order_reason_flows_to_transaction(self, csv_3_daily_prices):
        """
        Test that an entry order's reason flows correctly to the fill record.

        Verifies that when an order is placed with a reason, that reason is
        captured in the fill record created by the SimulationEngine.
        """

        def place_entry_with_reason(
            count: int,
            timestamp: datetime,
            bookkeeper: BookKeeper,
            triggered_timeframes: set[TimeFrame],
        ) -> None:
            if count == 0:
                action = (
                    self.account.long_entry(BTCUSDT_PERP, 1.0)
                    .reason("MA crossover")
                    .build()
                )
                bookkeeper.add(action)

        backtester = self._create_backtester(
            csv_3_daily_prices, place_entry_with_reason
        )
        asyncio.run(backtester.run())

        transactions = self.fill_recorder.get_fills()
        assert len(transactions) == 1

        entry_tx = transactions.iloc[0]
        assert entry_tx["fill_type"] == "enter_long"
        assert entry_tx["reason"] == "MA crossover"

    def test_exit_order_reason_flows_to_transaction(self, csv_3_daily_prices):
        """
        Test that an exit order's reason flows correctly to the fill record.

        Verifies that exit reasons are properly recorded in fills.
        """

        def place_entry_and_exit_with_reasons(
            count: int,
            timestamp: datetime,
            bookkeeper: BookKeeper,
            triggered_timeframes: set[TimeFrame],
        ) -> None:
            match count:
                case 0:
                    action = (
                        self.account.long_entry(BTCUSDT_PERP, 1.0)
                        .reason("entry signal")
                        .build()
                    )
                    bookkeeper.add(action)
                case 1:
                    action = (
                        self.account.long_exit(BTCUSDT_PERP, 1.0)
                        .reason("stop loss")
                        .build()
                    )
                    bookkeeper.add(action)

        backtester = self._create_backtester(
            csv_3_daily_prices, place_entry_and_exit_with_reasons
        )
        asyncio.run(backtester.run())

        transactions = self.fill_recorder.get_fills()
        assert len(transactions) == 2

        exit_tx = transactions.iloc[1]
        assert exit_tx["fill_type"] == "exit_long"
        assert exit_tx["reason"] == "stop loss"

    def test_traderecord_captures_entry_reason_from_transaction(
        self, csv_3_daily_prices
    ):
        """
        Test that TradeRecord.from_entry() picks up the reason from fill.

        Verifies that when creating a TradeRecord from an entry transaction,
        the entry_reason field is populated correctly.
        """

        def place_entry_with_reason(
            count: int,
            timestamp: datetime,
            bookkeeper: BookKeeper,
            triggered_timeframes: set[TimeFrame],
        ) -> None:
            if count == 0:
                action = (
                    self.account.long_entry(BTCUSDT_PERP, 1.0)
                    .reason("oversold RSI")
                    .build()
                )
                bookkeeper.add(action)

        backtester = self._create_backtester(
            csv_3_daily_prices, place_entry_with_reason
        )
        asyncio.run(backtester.run())

        fills_df = self.fill_recorder.get_fills()
        aggregator = create_trade_aggregation(fills_df)

        open_trades = aggregator.open_trades
        assert len(open_trades) == 1
        assert open_trades.iloc[0]["entry_reason"] == "oversold RSI"

    def test_traderecord_captures_exit_reason_from_transaction(
        self, csv_3_daily_prices
    ):
        """
        Test that TradeRecord.fill_exit() picks up the exit reason from fill.

        Verifies that when filling exit details in a TradeRecord, the exit_reason
        field is populated correctly from the exit fill.
        """

        def place_entry_and_exit_with_reasons(
            count: int,
            timestamp: datetime,
            bookkeeper: BookKeeper,
            triggered_timeframes: set[TimeFrame],
        ) -> None:
            match count:
                case 0:
                    action = (
                        self.account.long_entry(BTCUSDT_PERP, 1.0)
                        .reason("breakout")
                        .build()
                    )
                    bookkeeper.add(action)
                case 1:
                    action = (
                        self.account.long_exit(BTCUSDT_PERP, 1.0)
                        .reason("take profit")
                        .build()
                    )
                    bookkeeper.add(action)

        backtester = self._create_backtester(
            csv_3_daily_prices, place_entry_and_exit_with_reasons
        )
        asyncio.run(backtester.run())

        fills_df = self.fill_recorder.get_fills()
        aggregator = create_trade_aggregation(fills_df)

        assert len(aggregator._completed_trades) == 1

        trade_record = aggregator._completed_trades[0]
        assert trade_record.exit_reason == "take profit"

    def test_full_trade_has_both_entry_and_exit_reasons(self, csv_3_daily_prices):
        """
        Test that a complete trade has both entry_reason and exit_reason populated.

        Verifies the end-to-end flow: order placement with reasons -> fills
        with reasons -> TradeRecord with both reasons populated.
        """

        def place_entry_and_exit_with_reasons(
            count: int,
            timestamp: datetime,
            bookkeeper: BookKeeper,
            triggered_timeframes: set[TimeFrame],
        ) -> None:
            match count:
                case 0:
                    action = (
                        self.account.long_entry(BTCUSDT_PERP, 1.0)
                        .reason("MA golden cross")
                        .build()
                    )
                    bookkeeper.add(action)
                case 1:
                    action = (
                        self.account.long_exit(BTCUSDT_PERP, 1.0)
                        .reason("MA death cross")
                        .build()
                    )
                    bookkeeper.add(action)

        backtester = self._create_backtester(
            csv_3_daily_prices, place_entry_and_exit_with_reasons
        )
        asyncio.run(backtester.run())

        fills_df = self.fill_recorder.get_fills()
        assert len(fills_df) == 2

        entry_tx = fills_df.iloc[0]
        exit_tx = fills_df.iloc[1]
        assert entry_tx["reason"] == "MA golden cross"
        assert exit_tx["reason"] == "MA death cross"

        aggregator = create_trade_aggregation(fills_df)
        assert len(aggregator._completed_trades) == 1

        trade_record = aggregator._completed_trades[0]
        assert trade_record.entry_reason == "MA golden cross"
        assert trade_record.exit_reason == "MA death cross"

    def test_describer_names_a_fill_whose_order_states_no_reason(
        self, csv_3_daily_prices
    ):
        def place_entry_with_a_take_profit(
            count: int,
            timestamp: datetime,
            bookkeeper: BookKeeper,
            triggered_timeframes: set[TimeFrame],
        ) -> None:
            if count == 0:
                action = (
                    self.account.long_entry(BTCUSDT_PERP, 1.0)
                    .take_profit(115.0)
                    .build()
                )
                bookkeeper.add(action)

        def describe(fill: Execution) -> str | None:
            return "exit at reference" if fill.kind == "take-profit" else None

        strategy = DescribingStrategyFixture(
            csv_3_daily_prices, place_entry_with_a_take_profit, self.account, describe
        )
        backtester = Backtester(
            strategy,
            self.simulation_engine,
            strategy.ohlcv_provider,
            self.fill_recorder,
            "2024-01-01",
            "2024-01-04",
        )
        asyncio.run(backtester.run())

        fills_df = self.fill_recorder.get_fills()
        assert list(fills_df["fill_type"]) == ["enter_long", "exit_long"]
        assert pd.isna(fills_df.iloc[0]["reason"])
        assert fills_df.iloc[1]["reason"] == "exit at reference"

    def test_two_accounts_declaring_describers_stop_the_backtest(
        self, csv_3_daily_prices, simulated_exchange
    ):
        class TwoDescribersStrategy(StrategyFixture):
            async def setup(self, requirements: StrategyRequirements) -> None:
                await super().setup(requirements)
                first = FuturesAccount(simulated_exchange, name="first")
                second = FuturesAccount(simulated_exchange, name="second")
                requirements.account.add(first, describe_fills=lambda fill: "one")
                requirements.account.add(second, describe_fills=lambda fill: "two")

        strategy = TwoDescribersStrategy(csv_3_daily_prices, lambda *args: None)
        backtester = Backtester(
            strategy,
            self.simulation_engine,
            strategy.ohlcv_provider,
            self.fill_recorder,
            "2024-01-01",
            "2024-01-04",
        )

        with pytest.raises(StrategyCriticalError, match="2 accounts declare"):
            asyncio.run(backtester.run())

    def test_orders_without_reason_result_in_none(self, csv_3_daily_prices):
        """
        Test that orders placed without a reason result in None values.

        Verifies that the system correctly handles the absence of reasons,
        storing None in fills and TradeRecords.
        """

        def place_entry_and_exit_without_reasons(
            count: int,
            timestamp: datetime,
            bookkeeper: BookKeeper,
            triggered_timeframes: set[TimeFrame],
        ) -> None:
            match count:
                case 0:
                    action = self.account.long_entry(BTCUSDT_PERP, 1.0).build()
                    bookkeeper.add(action)
                case 1:
                    action = self.account.long_exit(BTCUSDT_PERP, 1.0).build()
                    bookkeeper.add(action)

        backtester = self._create_backtester(
            csv_3_daily_prices, place_entry_and_exit_without_reasons
        )
        asyncio.run(backtester.run())

        fills_df = self.fill_recorder.get_fills()
        assert len(fills_df) == 2

        entry_tx = fills_df.iloc[0]
        exit_tx = fills_df.iloc[1]
        assert entry_tx["reason"] is None
        assert exit_tx["reason"] is None

        aggregator = create_trade_aggregation(fills_df)
        assert len(aggregator._completed_trades) == 1

        trade_record = aggregator._completed_trades[0]
        assert trade_record.entry_reason is None
        assert trade_record.exit_reason is None

    def test_multiple_trades_with_different_reasons(self, csv_3_daily_prices):
        """
        Test that multiple trades maintain distinct reasons.

        Verifies that when multiple entry/exit cycles occur with different reasons,
        each trade record correctly maintains its own entry and exit reasons.
        """

        def place_multiple_trades_with_reasons(
            count: int,
            timestamp: datetime,
            bookkeeper: BookKeeper,
            triggered_timeframes: set[TimeFrame],
        ) -> None:
            match count:
                case 0:
                    action = (
                        self.account.long_entry(BTCUSDT_PERP, 0.5)
                        .reason("first entry")
                        .build()
                    )
                    bookkeeper.add(action)
                case 1:
                    action = (
                        self.account.long_exit(BTCUSDT_PERP, 0.5)
                        .reason("first exit")
                        .build()
                    )
                    bookkeeper.add(action)
                    action = (
                        self.account.long_entry(BTCUSDT_PERP, 0.5)
                        .reason("second entry")
                        .build()
                    )
                    bookkeeper.add(action)
                case 2:
                    action = (
                        self.account.long_exit(BTCUSDT_PERP, 0.5)
                        .reason("second exit")
                        .build()
                    )
                    bookkeeper.add(action)

        backtester = self._create_backtester(
            csv_3_daily_prices, place_multiple_trades_with_reasons
        )
        asyncio.run(backtester.run())

        fills_df = self.fill_recorder.get_fills()
        assert len(fills_df) == 4

        aggregator = create_trade_aggregation(fills_df)
        assert len(aggregator._completed_trades) == 2

        trades_sorted = sorted(aggregator._completed_trades, key=lambda t: t.entry_time)

        first_trade = trades_sorted[0]
        assert first_trade.entry_reason == "first entry"
        assert first_trade.exit_reason == "first exit"

        second_trade = trades_sorted[1]
        assert second_trade.entry_reason == "second entry"
        assert second_trade.exit_reason == "second exit"

    def test_partial_exit_preserves_entry_reason(self, csv_3_daily_prices):
        """
        Test that partial exits maintain the correct entry reason.

        Verifies that when a position is partially exited, the resulting trade
        records all carry the same entry reason from the original entry.
        """

        def place_entry_with_partial_exits(
            count: int,
            timestamp: datetime,
            bookkeeper: BookKeeper,
            triggered_timeframes: set[TimeFrame],
        ) -> None:
            match count:
                case 0:
                    action = (
                        self.account.long_entry(BTCUSDT_PERP, 1.0)
                        .reason("strong signal")
                        .build()
                    )
                    bookkeeper.add(action)
                case 1:
                    action = (
                        self.account.long_exit(BTCUSDT_PERP, 0.5)
                        .reason("partial TP1")
                        .build()
                    )
                    bookkeeper.add(action)
                case 2:
                    action = (
                        self.account.long_exit(BTCUSDT_PERP, 0.5)
                        .reason("partial TP2")
                        .build()
                    )
                    bookkeeper.add(action)

        backtester = self._create_backtester(
            csv_3_daily_prices, place_entry_with_partial_exits
        )
        asyncio.run(backtester.run())

        fills_df = self.fill_recorder.get_fills()
        assert len(fills_df) == 3

        entry_tx = fills_df.iloc[0]
        exit_tx1 = fills_df.iloc[1]
        exit_tx2 = fills_df.iloc[2]

        assert entry_tx["reason"] == "strong signal"
        assert exit_tx1["reason"] == "partial TP1"
        assert exit_tx2["reason"] == "partial TP2"

        aggregator = create_trade_aggregation(fills_df)
        assert len(aggregator._completed_trades) >= 1

        for trade_record in aggregator._completed_trades:
            assert trade_record.entry_reason == "strong signal"
            assert trade_record.exit_reason in ["partial TP1", "partial TP2"]
