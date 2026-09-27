from datetime import datetime
from unittest.mock import Mock

import pytest

from robottraderslab import Symbol, TimeFrame
from robottraderslab._core import OHLCVProviderProtocol, PlacementReserve
from robottraderslab.backtester.backtester import Backtester
from robottraderslab.backtester.simulator import (
    CacheFillRecorder,
    FeeRates,
    FuturesSimulationEngine,
    SimulatedFuturesExchange,
    fee_model_for,
)
from robottraderslab.exceptions import StrategyCriticalError
from robottraderslab.futures import FuturesAccount
from robottraderslab.ohlcv_provider import MockOHLCVProvider
from robottraderslab.strategies import (
    AccountSnapshots,
    BookKeeper,
    OHLCVs,
    StrategyProtocol,
    StrategyRequirements,
)
from robottraderslab.strategies.futures import AvailableBalanceRatio

AUD_CAD_PERP = Symbol.create("AUD/CAD:USD")
BTC_ETH_PERP = Symbol.create("BTC/ETH:BTC")
BTC_USDT_PERP = Symbol.create("BTC/USDT:USDT")


class PassiveStrategy(StrategyProtocol):
    market_type = "futures"

    def __init__(self, symbol: Symbol):
        self._symbol = symbol

    async def setup(self, requirements: StrategyRequirements) -> None:
        requirements.ohlcv.add(self._symbol, "1d")

    def generate_trading_signals(self, ohlcvs: OHLCVs) -> None: ...

    def book_trading_actions(
        self,
        ohlcvs: OHLCVs,
        account_snapshots: AccountSnapshots,
        timestamp: datetime,
        bookkeeper: BookKeeper,
        triggered_timeframes: set[TimeFrame],
    ) -> None: ...


def make_engine(margin_currency: str) -> FuturesSimulationEngine:
    return FuturesSimulationEngine(
        initial_balance={margin_currency: 10_000.0},
        fee_rates=FeeRates(maker=0.0, taker=0.0),
        fill_recorder=CacheFillRecorder(),
        margin_currency=margin_currency,
        fee_model=fee_model_for("cost", PlacementReserve()),
    )


def make_backtester(
    strategy: PassiveStrategy,
    engine: FuturesSimulationEngine,
    provider: OHLCVProviderProtocol,
) -> Backtester:
    return Backtester(
        strategy,
        engine,
        provider,
        CacheFillRecorder(),
        "2024-01-01",
        "2024-01-05",
    )


async def test_cross_quoted_symbol_registers_conversion_pair():
    engine = make_engine("USD")
    backtester = make_backtester(
        PassiveStrategy(AUD_CAD_PERP), engine, MockOHLCVProvider()
    )

    await backtester.run()

    assert engine.converter.registered_pair("CAD") == (Symbol("CAD", "USD"), False)


async def test_base_margined_symbol_converts_through_itself():
    engine = make_engine("BTC")
    backtester = make_backtester(
        PassiveStrategy(BTC_ETH_PERP), engine, MockOHLCVProvider()
    )

    await backtester.run()

    assert engine.converter.registered_pair("ETH") == (BTC_ETH_PERP, True)


async def test_margin_quoted_symbol_needs_no_conversion():
    engine = make_engine("USDT")
    backtester = make_backtester(
        PassiveStrategy(BTC_USDT_PERP), engine, MockOHLCVProvider()
    )

    await backtester.run()

    assert engine.converter.registered_pair("USDT") is None


class FirstCandleStrategy(PassiveStrategy):
    def __init__(self, symbol: Symbol, account: FuturesAccount):
        super().__init__(symbol)
        self._account = account
        self._placed = False

    async def setup(self, requirements: StrategyRequirements) -> None:
        await super().setup(requirements)
        requirements.account.add(self._account, symbols=[self._symbol], balances=True)

    def book_trading_actions(
        self,
        ohlcvs: OHLCVs,
        account_snapshots: AccountSnapshots,
        timestamp: datetime,
        bookkeeper: BookKeeper,
        triggered_timeframes: set[TimeFrame],
    ) -> None:
        if not self._placed:
            action = (
                self._account.long_entry(self._symbol)
                .size(
                    AvailableBalanceRatio(0.1), 1.0, account_snapshots.of(self._account)
                )
                .build()
            )
            bookkeeper.add(action)
            self._placed = True


async def test_order_on_first_candle_sizes_with_seeded_rate():
    engine = make_engine("USD")
    account = FuturesAccount(SimulatedFuturesExchange(engine, CacheFillRecorder()))
    backtester = make_backtester(
        FirstCandleStrategy(AUD_CAD_PERP, account),
        engine,
        MockOHLCVProvider(initial_price=1.0),
    )

    await backtester.run()

    assert len(engine.open_positions) == 1


async def test_no_conversion_data_available():
    engine = make_engine("USD")
    provider = Mock(spec=OHLCVProviderProtocol)
    provider.fetch_ohlcv.side_effect = RuntimeError("symbol not supported")
    backtester = make_backtester(PassiveStrategy(AUD_CAD_PERP), engine, provider)

    with pytest.raises(StrategyCriticalError, match="No conversion data"):
        await backtester.run()
