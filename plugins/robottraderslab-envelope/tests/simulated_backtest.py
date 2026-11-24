from pathlib import Path
from typing import Any

from robottraderslab.backtester.backtester import Backtester
from robottraderslab.backtester.simulator import SimulatedFuturesExchange
from robottraderslab.exchanges import OHLCVProviderProtocol
from robottraderslab.strategies import StrategyProtocol, TradingMode, TradingSystem
from robottraderslab.strategies.futures import FuturesAccount
from robottraderslab_envelope import EnvelopeStrategy


def create_exchange[E: SimulatedFuturesExchange](kind: type[E]) -> E:
    return kind.create_from_settings(
        initial_balance={"USDT": 10_000.0},
        maker_fee_rate=0.0,
        taker_fee_rate=0.0,
        fee_mode="cost",
    )


def create_envelope[S: EnvelopeStrategy](
    kind: type[S],
    exchange: SimulatedFuturesExchange,
    profile: dict[str, Any],
    entry_order: str,
) -> S:
    return kind(
        account=FuturesAccount(exchange=exchange),
        trading_system=TradingSystem(trading_mode=TradingMode.BACKTEST),
        config_dir=Path("bot-config-dir"),
        profiles=[profile],
        entry_order=entry_order,
    )


def create_backtester(
    strategy: StrategyProtocol,
    exchange: SimulatedFuturesExchange,
    provider: OHLCVProviderProtocol,
    start_date: str,
    end_date: str,
) -> Backtester:
    return Backtester(
        strategy,
        exchange.simulation_engine,
        provider,
        exchange.fill_recorder,
        start_date,
        end_date,
    )
