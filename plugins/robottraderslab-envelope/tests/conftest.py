import asyncio
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from rung_ids import rung_client_order_id

from robottraderslab import Symbol
from robottraderslab.exchanges import (
    FuturesExchangeProtocol,
    Instrument,
)
from robottraderslab.futures import read_account_state
from robottraderslab.strategies import (
    AccountSnapshot,
    AccountSnapshots,
    Balance,
    Execution,
    OrderProtocol,
    OrderSide,
    OrderType,
    PositionSide,
    StrategyRequirements,
    TradingMode,
    TradingSystem,
)
from robottraderslab.strategies.futures import (
    FuturesAccount,
    FuturesCapabilities,
    MarginMode,
    MarginSettings,
    PositionSnapshot,
)
from robottraderslab_envelope import EnvelopeStrategy
from robottraderslab_envelope.profile_config import ProfileConfig

BTC = Symbol.create("BTC/USDT:USDT")
ACCOUNT_NAME = "test"
FIRST_CANDLE = datetime(2024, 2, 19)
_LONG_LIQUIDATION_REACH = 0.5
_SHORT_LIQUIDATION_REACH = 1.5


@pytest.fixture
def mock_exchange() -> Mock:
    """Mocked futures exchange with a flat account and no open positions."""
    exchange = Mock(spec=FuturesExchangeProtocol)
    exchange.get_account_state.side_effect = partial(read_account_state, exchange)
    exchange.placement_reserve_rate = 0.0
    exchange.get_balances.return_value = {
        "USDT": Balance(locked=0.0, total=10_000.0),
    }
    exchange.get_open_positions.return_value = {}
    exchange.get_equity.return_value = 10_000.0
    exchange.get_quote_conversion_rate.return_value = 1.0
    exchange.get_open_orders.return_value = []
    exchange.get_executions_since.return_value = []
    exchange.get_margin_settings.side_effect = lambda symbols: {
        symbol: MarginSettings(leverage=None, margin_mode=MarginMode.CROSS)
        for symbol in symbols
    }
    exchange.get_instrument.return_value = Instrument(
        quantity_step=1e-8,
        minimum_order_value=None,
        maximum_leverage=None,
        maximum_order_quantity=None,
        maximum_market_order_quantity=None,
    )
    exchange.capabilities = FuturesCapabilities()
    return exchange


@pytest.fixture(params=["trigger", "limit"])
def entry_order(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def make_strategy(
    mock_exchange: Mock,
) -> Callable[..., EnvelopeStrategy]:
    """Factory for creating EnvelopeStrategy backed by the mocked exchange."""

    def _make(
        profiles: list[dict[str, Any]],
        trading_mode: TradingMode = TradingMode.BACKTEST,
        **strategy_kwargs: Any,
    ) -> EnvelopeStrategy:
        account = FuturesAccount(exchange=mock_exchange, name="test")
        strategy = EnvelopeStrategy(
            account=account,
            trading_system=TradingSystem(trading_mode=trading_mode),
            config_dir=Path("bot-config-dir"),
            profiles=profiles,
            **strategy_kwargs,
        )
        asyncio.run(
            account._read_instruments({profile.symbol for profile in strategy.profiles})
        )
        return strategy

    return _make


@pytest.fixture
def make_entry_fill() -> Callable[..., Execution]:
    def _make(
        client_order_id: str | None,
        kind: OrderType,
        side: OrderSide = OrderSide.BUY,
        *,
        execution_id: str = "fired-1",
        quantity: float = 0.5,
        price: float = 95.0,
        timestamp: datetime = FIRST_CANDLE + timedelta(hours=5),
    ) -> Execution:
        return Execution(
            execution_id=execution_id,
            order_id=execution_id,
            symbol=BTC,
            side=side,
            price=price,
            quantity=quantity,
            timestamp=timestamp,
            kind=kind,
            client_order_id=client_order_id,
        )

    return _make


@pytest.fixture
def make_band_fill(
    make_entry_fill: Callable[..., Execution],
) -> Callable[..., Execution]:
    def _make(
        profile: ProfileConfig,
        side: PositionSide,
        band_index: int,
        *,
        quantity: float = 1.0,
        minute: int | None = None,
    ) -> Execution:
        order_side = OrderSide.BUY if side is PositionSide.LONG else OrderSide.SELL
        filled_at = band_index if minute is None else minute
        return make_entry_fill(
            rung_client_order_id(profile.timeframe, order_side, band_index),
            OrderType.LIMIT,
            order_side,
            execution_id=f"band-{side.name.lower()}{band_index}-{filled_at}",
            quantity=quantity,
            price=90.0,
            timestamp=FIRST_CANDLE + timedelta(minutes=filled_at),
        )

    return _make


@pytest.fixture
def make_position() -> Callable[..., PositionSnapshot]:
    def _make(
        side: PositionSide, quantity: float = 1.0, entry_price: float = 100.0
    ) -> PositionSnapshot:
        reach = (
            _LONG_LIQUIDATION_REACH
            if side is PositionSide.LONG
            else _SHORT_LIQUIDATION_REACH
        )
        return PositionSnapshot(
            symbol=BTC,
            side=side,
            quantity=quantity,
            average_entry_price=entry_price,
            entry_time=FIRST_CANDLE,
            leverage=5.0,
            liquidation_price=entry_price * reach,
            stop_loss_price=None,
            take_profit_price=None,
        )

    return _make


@pytest.fixture
def make_account_state() -> Callable[..., AccountSnapshots]:
    def _make(
        *,
        open_orders: Iterable[OrderProtocol] = (),
        holding: PositionSnapshot | None = None,
        executions: Iterable[Execution] = (),
        total_balance: float = 10_000.0,
    ) -> AccountSnapshots:
        snapshot = AccountSnapshot(
            account_name=ACCOUNT_NAME,
            balances={"USDT": Balance(locked=0.0, total=total_balance)},
            open_orders=list(open_orders),
            positions={} if holding is None else {BTC: holding},
            executions=list(executions),
            executions_declared=True,
            margin_settings={
                BTC: MarginSettings(leverage=None, margin_mode=MarginMode.CROSS)
            },
            conversion_rates={BTC: 1.0},
        )
        return AccountSnapshots({ACCOUNT_NAME: snapshot})

    return _make


@pytest.fixture
def requirements() -> StrategyRequirements:
    return StrategyRequirements()


@pytest.fixture
def fold_fills(
    requirements: StrategyRequirements,
    make_account_state: Callable[..., AccountSnapshots],
) -> Callable[..., AccountSnapshots]:
    def _fold(
        fills: Iterable[Execution],
        held: PositionSnapshot | None,
        *,
        window_start: datetime = FIRST_CANDLE,
    ) -> AccountSnapshots:
        state = make_account_state(holding=held, executions=fills)
        requirements.tracker.refresh_all(
            state, window_start, window_start + timedelta(days=1)
        )
        return state

    return _fold


@pytest.fixture
def hold_bands(
    fold_fills: Callable[..., AccountSnapshots],
    make_band_fill: Callable[..., Execution],
    make_position: Callable[..., PositionSnapshot],
) -> Callable[..., None]:
    def _hold(
        strategy: EnvelopeStrategy, side: PositionSide, *band_indexes: int
    ) -> None:
        profile = strategy.profiles[0]
        fold_fills(
            [make_band_fill(profile, side, index) for index in band_indexes],
            make_position(side, float(len(band_indexes))),
        )

    return _hold


@pytest.fixture
def make_profile_settings() -> Callable[..., dict[str, Any]]:
    def _make(symbol: str, timeframe: str, **envelope_keys: Any) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "average_type": "SMA",
            "average_period": 100,
            "stop_loss_pct": 0.30,
            "total_balance_ratio": 0.5,
        } | envelope_keys

    return _make


@pytest.fixture
def band_strategy(
    make_strategy: Callable[..., EnvelopeStrategy],
    make_profile_settings: Callable[..., dict[str, Any]],
) -> EnvelopeStrategy:
    return make_strategy(
        [
            make_profile_settings(
                "BTC/USDT:USDT",
                "1d",
                long_envelopes=[0.05, 0.10, 0.15],
                short_envelopes=[0.05],
            ),
            make_profile_settings("ETH/USDT:USDT", "1h", envelopes=[0.03, 0.06]),
        ]
    )


@pytest.fixture
def profiles(band_strategy: EnvelopeStrategy) -> list[ProfileConfig]:
    return band_strategy.profiles
