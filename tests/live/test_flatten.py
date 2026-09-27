import logging
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from robottraderslab import Symbol
from robottraderslab._core import OHLCVRow, PlacementReserve, profile_tag
from robottraderslab.backtester.simulator import (
    CacheFillRecorder,
    FeeRates,
    FuturesSimulationEngine,
    SimulatedFuturesExchange,
    fee_model_for,
)
from robottraderslab.exceptions import (
    ExchangeCriticalError,
    ExchangeRecoverableError,
    StrategyCriticalError,
)
from robottraderslab.exchanges import (
    FuturesExchangeProtocol,
    OrderProtocol,
    OrderSide,
    PlacedOrder,
    PositionSide,
    PositionSnapshot,
    client_order_id_carrying,
)
from robottraderslab.live.flatten import main

_BTC = Symbol.create("BTC/USDT:USDT")
_ETH = Symbol.create("ETH/USDT:USDT")
_SOL = Symbol.create("SOL/USDT:USDT")
_TIMEFRAME = "1d"
_PRICE = 100.0
_TAKER_FEE = 0.01
_INITIAL_BALANCE = 10_000.0
_MARGIN_CURRENCY = "USDT"
_ALL_FILLS = datetime.min.replace(tzinfo=UTC)

_MA_PROFILE = """
[[strategy.profiles]]
symbol = "{symbol}"
timeframe = "1d"
fast_ma_length = 2
slow_ma_length = 3
total_balance_ratio = 0.1
tag = "{tag}"
"""

type _MakeMaBot = Callable[[Sequence[tuple[Symbol, str]]], Path]


def _order(symbol: Symbol) -> Mock:
    order = Mock(spec=OrderProtocol)
    order.symbol = symbol
    return order


def _position(
    symbol: Symbol, side: PositionSide = PositionSide.LONG, quantity: float = 1.0
) -> PositionSnapshot:
    return PositionSnapshot(
        symbol=symbol,
        side=side,
        quantity=quantity,
        average_entry_price=100.0,
        entry_time=datetime(2026, 1, 1, tzinfo=UTC),
        leverage=1.0,
        liquidation_price=10.0,
    )


_plugged: dict[str, Any] = {}


def make_exchange(**settings: Any) -> AsyncMock:
    """The connector the test bot names in place of a venue, handed what the
    account's secrets entry carries.
    """
    _plugged["settings"] = settings
    return _plugged["exchange"]


_PLUGGED_EXCHANGE = f"{__name__}:make_exchange"


def _bot(exchange: str) -> str:
    return f"""
secrets_file = "creds.toml"

[strategy]
strategy_class = "impulse"

[live.trading_account]
exchange = "{exchange}"
secret_name = "my_account"

[live.ohlcv_provider]
ohlcv_provider = "bitget"
"""


_CREDENTIALS = """
[[secrets]]
name = "my_account"
api_key = "shh"
"""


@pytest.fixture
def exchange() -> Iterator[AsyncMock]:
    exchange = AsyncMock(spec=FuturesExchangeProtocol)
    exchange.get_open_orders.return_value = []
    exchange.get_open_positions.return_value = {}
    _plugged["exchange"] = exchange
    yield exchange
    _plugged.clear()


@pytest.fixture
def config_file(tmp_path: Path, exchange: AsyncMock) -> Path:
    return _write_bot(tmp_path, _PLUGGED_EXCHANGE)


def _write_bot(directory: Path, exchange: str) -> Path:
    (directory / "creds.toml").write_text(_CREDENTIALS)
    path = directory / "impulse-bot-example.toml"
    path.write_text(_bot(exchange))
    return path


def _run(config_file: Path, **overrides: Any) -> int:
    arguments: dict[str, Any] = {
        "config": config_file,
        "symbol": None,
        "external": False,
    }
    return main(**{**arguments, **overrides})


def test_cancels_happen_before_the_close_on_the_same_symbol(exchange, config_file):
    exchange.get_open_positions.return_value = {_BTC: _position(_BTC)}

    _run(config_file, symbol="BTC/USDT:USDT")

    call_names = [call[0] for call in exchange.mock_calls]
    assert call_names.index("cancel_orders_for_symbol") < call_names.index(
        "place_market_order"
    )


def test_the_close_order_is_reduce_only_opposite_side_and_full_quantity(
    exchange, config_file
):
    exchange.get_open_positions.return_value = {
        _BTC: _position(_BTC, side=PositionSide.LONG, quantity=2.5)
    }

    _run(config_file, symbol="BTC/USDT:USDT")

    exchange.place_market_order.assert_awaited_once_with(
        _BTC, OrderSide.SELL, 2.5, reduce_only=True
    )


def test_a_short_position_closes_with_a_buy(exchange, config_file):
    exchange.get_open_positions.return_value = {
        _BTC: _position(_BTC, side=PositionSide.SHORT, quantity=1.5)
    }

    _run(config_file, symbol="BTC/USDT:USDT")

    exchange.place_market_order.assert_awaited_once_with(
        _BTC, OrderSide.BUY, 1.5, reduce_only=True
    )


def test_omitting_symbol_targets_the_union_of_orders_and_positions(
    exchange, config_file
):
    exchange.get_open_orders.side_effect = [[_order(_BTC)], []]
    exchange.get_open_positions.side_effect = [{_ETH: _position(_ETH)}, {}]

    exit_code = _run(config_file)

    cancelled_symbols = {
        call.args[0] for call in exchange.cancel_orders_for_symbol.await_args_list
    }
    assert cancelled_symbols == {_BTC, _ETH}
    assert exit_code == 0


def test_a_recoverable_error_on_one_symbol_does_not_stop_the_others(
    exchange, config_file, caplog
):
    exchange.get_open_orders.side_effect = [
        [_order(_BTC), _order(_ETH)],
        [_order(_BTC)],
    ]
    exchange.cancel_orders_for_symbol.side_effect = [
        ExchangeRecoverableError("no permission"),
        None,
    ]

    with caplog.at_level(logging.WARNING):
        exit_code = _run(config_file)

    assert exchange.cancel_orders_for_symbol.await_count == 2
    assert exit_code == 1
    assert "Could not fully flatten" in caplog.text
    assert str(_BTC) in caplog.text


def test_a_critical_error_propagates(exchange, config_file):
    exchange.get_open_orders.return_value = [_order(_BTC)]
    exchange.cancel_orders_for_symbol.side_effect = ExchangeCriticalError(
        "auth revoked"
    )

    with pytest.raises(ExchangeCriticalError, match="auth revoked"):
        _run(config_file)


def test_an_already_flat_account_makes_no_venue_writes(exchange, config_file):
    exit_code = _run(config_file)

    exchange.cancel_orders_for_symbol.assert_not_awaited()
    exchange.place_market_order.assert_not_awaited()
    assert exit_code == 0


def test_the_account_is_the_one_the_configuration_file_trades(config_file):
    _run(config_file)

    assert _plugged["settings"]["api_key"] == "shh"


def test_a_bot_without_a_live_section_is_refused(tmp_path):
    backtest_only = tmp_path / "backtest-only.toml"
    backtest_only.write_text('[strategy]\nstrategy_class = "impulse"\n')

    with pytest.raises(StrategyCriticalError, match=r"\[live\]"):
        main(config=backtest_only, symbol=None, external=False)


def test_a_missing_configuration_file_is_refused(tmp_path):
    with pytest.raises(StrategyCriticalError, match="absent.toml"):
        main(config=tmp_path / "absent.toml", symbol=None, external=False)


def test_a_venue_the_engine_cannot_load_is_refused(tmp_path):
    ghost = _write_bot(tmp_path, "ghost")

    with pytest.raises(ExchangeCriticalError, match="ghost"):
        main(config=ghost, symbol=None, external=False)


@pytest.mark.parametrize(
    ("open_orders", "remaining_orders"),
    [([], []), ([_order(_BTC)], []), ([_order(_BTC)], [_order(_BTC)])],
    ids=["already flat", "flat after cancelling", "not flat"],
)
def test_every_verdict_names_the_bots_configuration_file(
    exchange, config_file, caplog, open_orders, remaining_orders
):
    exchange.get_open_orders.side_effect = [open_orders, remaining_orders]

    with caplog.at_level(logging.INFO):
        _run(config_file)

    assert "impulse-bot-example.toml" in caplog.text


class _FillingSimulator(SimulatedFuturesExchange):
    """The simulator fills a market order on the next candle it is handed, and
    a venue fills one as it is placed.
    """

    async def place_market_order(
        self, symbol: Symbol, side: OrderSide, quantity: float, **kwargs: Any
    ) -> PlacedOrder:
        placed = await super().place_market_order(symbol, side, quantity, **kwargs)
        moment = datetime.now(UTC) - timedelta(days=1)
        self.simulation_engine.simulate_on_current_ohlcvs(
            moment,
            {
                symbol: OHLCVRow(
                    timestamp=moment,
                    open=_PRICE,
                    high=_PRICE,
                    low=_PRICE,
                    close=_PRICE,
                    volume=1.0,
                )
            },
        )
        return placed


class _RefusingSimulator(_FillingSimulator):
    refused: Symbol | None = None

    async def place_market_order(
        self, symbol: Symbol, side: OrderSide, quantity: float, **kwargs: Any
    ) -> PlacedOrder:
        if symbol == self.refused:
            raise ExchangeRecoverableError("not enough margin")
        return await super().place_market_order(symbol, side, quantity, **kwargs)


@pytest.fixture
def simulator() -> Iterator[_FillingSimulator]:
    yield from _plug_simulator(FeeRates(maker=0.0, taker=0.0), _FillingSimulator)


@pytest.fixture
def charging_simulator() -> Iterator[_FillingSimulator]:
    """A venue paying its fee out of the position it opens, so an entry lands
    short of the quantity its order asked for.
    """
    yield from _plug_simulator(FeeRates(maker=0.0, taker=_TAKER_FEE), _FillingSimulator)


@pytest.fixture
def refusing_simulator() -> Iterator[_RefusingSimulator]:
    yield from _plug_simulator(FeeRates(maker=0.0, taker=0.0), _RefusingSimulator)


@pytest.fixture
def make_ma_bot(tmp_path: Path) -> _MakeMaBot:
    def _make(profiles: Sequence[tuple[Symbol, str]]) -> Path:
        (tmp_path / "creds.toml").write_text(_CREDENTIALS)
        path = tmp_path / "ma-bot-example.toml"
        path.write_text(_ma_bot(profiles))
        return path

    return _make


def _plug_simulator[T: _FillingSimulator](
    fee_rates: FeeRates, simulator_class: type[T]
) -> Iterator[T]:
    fill_recorder = CacheFillRecorder()
    simulation_engine = FuturesSimulationEngine(
        initial_balance={_MARGIN_CURRENCY: _INITIAL_BALANCE},
        fee_rates=fee_rates,
        fill_recorder=fill_recorder,
        margin_currency=_MARGIN_CURRENCY,
        fee_model=fee_model_for("cost", PlacementReserve()),
    )
    simulation_engine.enable_execution_recording()
    exchange = simulator_class(simulation_engine, fill_recorder)
    _plugged["exchange"] = exchange
    yield exchange
    _plugged.clear()


def _ma_bot(profiles: Sequence[tuple[Symbol, str]]) -> str:
    declared = "".join(
        _MA_PROFILE.format(symbol=symbol, tag=tag) for symbol, tag in profiles
    )
    return f"""
secrets_file = "creds.toml"

[strategy]
strategy_class = "futures_ma"
{declared}
[live.trading_account]
exchange = "{_PLUGGED_EXCHANGE}"
secret_name = "my_account"

[live.ohlcv_provider]
ohlcv_provider = "bitget"
"""


async def _a_profile_fills(
    exchange: _FillingSimulator,
    symbol: Symbol,
    side: OrderSide,
    quantity: float,
    tag: str,
) -> None:
    await exchange.place_market_order(
        symbol,
        side,
        quantity,
        client_order_id=client_order_id_carrying(profile_tag(_TIMEFRAME, tag)),
    )


async def _a_hand_fills(
    exchange: _FillingSimulator, symbol: Symbol, side: OrderSide, quantity: float
) -> None:
    await exchange.place_market_order(symbol, side, quantity)


def _run_external(config_file: Path) -> int:
    return _run(config_file, external=True)


async def test_a_leftover_beside_the_bots_own_position_goes_alone(
    simulator, make_ma_bot
):
    bot = make_ma_bot([(_BTC, "p1")])
    await _a_profile_fills(simulator, _BTC, OrderSide.BUY, 0.02, "p1")
    await _a_hand_fills(simulator, _BTC, OrderSide.BUY, 0.06)

    exit_code = _run_external(bot)

    assert exit_code == 0
    assert simulator.simulation_engine.open_positions[_BTC].quantity == 0.02
    assert simulator.simulation_engine.open_positions[_BTC].side == PositionSide.LONG


async def test_a_hand_made_short_netting_against_the_bots_long_is_bought_back(
    simulator, make_ma_bot
):
    bot = make_ma_bot([(_BTC, "p1")])
    await _a_profile_fills(simulator, _BTC, OrderSide.BUY, 0.05, "p1")
    await _a_hand_fills(simulator, _BTC, OrderSide.SELL, 0.03)

    exit_code = _run_external(bot)

    assert exit_code == 0
    assert simulator.simulation_engine.open_positions[_BTC].quantity == 0.05
    assert simulator.simulation_engine.open_positions[_BTC].side == PositionSide.LONG


async def test_a_symbol_holding_only_its_profiles_position_is_named_clean(
    simulator, make_ma_bot, caplog
):
    bot = make_ma_bot([(_BTC, "p1")])
    await _a_profile_fills(simulator, _BTC, OrderSide.BUY, 0.02, "p1")

    with caplog.at_level(logging.INFO):
        exit_code = _run_external(bot)

    assert exit_code == 0
    assert "is clean" in caplog.text
    assert len(simulator.simulation_engine.get_executions_since(_ALL_FILLS)) == 1


async def test_a_symbol_the_bot_does_not_declare_is_left_alone(
    simulator, make_ma_bot, caplog
):
    bot = make_ma_bot([(_BTC, "p1"), (_SOL, "p1")])
    await _a_profile_fills(simulator, _BTC, OrderSide.BUY, 0.02, "p1")
    await _a_hand_fills(simulator, _BTC, OrderSide.BUY, 0.06)
    await _a_hand_fills(simulator, _ETH, OrderSide.BUY, 0.5)

    with caplog.at_level(logging.INFO):
        exit_code = _run_external(bot)

    assert exit_code == 0
    assert simulator.simulation_engine.open_positions[_ETH].quantity == 0.5
    assert str(_ETH) not in caplog.text


async def test_a_venue_settling_short_of_the_profiles_total_fails_the_run(
    charging_simulator, make_ma_bot, caplog
):
    bot = make_ma_bot([(_BTC, "p1")])
    await _a_profile_fills(charging_simulator, _BTC, OrderSide.BUY, 0.05, "p1")
    await _a_hand_fills(charging_simulator, _BTC, OrderSide.SELL, 0.03)

    with caplog.at_level(logging.ERROR):
        exit_code = _run_external(bot)

    assert exit_code == 1
    assert "still holds what no profile owns" in caplog.text
    assert str(_BTC) in caplog.text


async def test_a_strategy_tracking_no_profile_is_refused(simulator, tmp_path):
    (tmp_path / "creds.toml").write_text(_CREDENTIALS)
    untracked = tmp_path / "ma-bot-example.toml"
    untracked.write_text(_ma_bot([]))

    with pytest.raises(StrategyCriticalError, match="tracks no profile"):
        _run_external(untracked)


async def test_a_hand_holding_the_venue_on_the_other_side_is_refused(
    simulator, make_ma_bot, caplog
):
    bot = make_ma_bot([(_BTC, "p1")])
    await _a_hand_fills(simulator, _BTC, OrderSide.SELL, 0.10)
    await _a_profile_fills(simulator, _BTC, OrderSide.BUY, 0.03, "p1")

    with caplog.at_level(logging.WARNING):
        exit_code = _run_external(bot)

    assert exit_code == 1
    assert "through flat" in caplog.text
    assert simulator.simulation_engine.open_positions[_BTC].quantity == 0.07
    assert simulator.simulation_engine.open_positions[_BTC].side == PositionSide.SHORT


async def test_a_refused_close_on_one_symbol_does_not_stop_the_others(
    refusing_simulator, make_ma_bot, caplog
):
    bot = make_ma_bot([(_BTC, "p1"), (_SOL, "p1")])
    await _a_profile_fills(refusing_simulator, _BTC, OrderSide.BUY, 0.02, "p1")
    await _a_hand_fills(refusing_simulator, _BTC, OrderSide.BUY, 0.06)
    await _a_profile_fills(refusing_simulator, _SOL, OrderSide.BUY, 1.0, "p1")
    await _a_hand_fills(refusing_simulator, _SOL, OrderSide.BUY, 2.0)
    refusing_simulator.refused = _BTC

    with caplog.at_level(logging.WARNING):
        exit_code = _run_external(bot)

    assert exit_code == 1
    assert refusing_simulator.simulation_engine.open_positions[_SOL].quantity == 1.0
    assert "Could not close" in caplog.text


def test_another_symbols_open_order_does_not_fail_the_verdict(
    exchange, config_file, caplog
):
    exchange.get_open_orders.side_effect = [[], [_order(_ETH)]]

    with caplog.at_level(logging.INFO):
        exit_code = _run(config_file, symbol="BTC/USDT:USDT")

    assert exit_code == 0
    assert "is flat on BTC/USDT:USDT" in caplog.text
