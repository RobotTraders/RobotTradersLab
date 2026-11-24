import logging
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import ANY, AsyncMock, Mock

import pytest
from live_bot_helpers import AN_OLD_MOMENT_NS, seed_old_copy, seeded
from live_cycle_helpers import make_bot_folder

from robottraderslab import Symbol
from robottraderslab._internal import EX_TEMPFAIL, hold_instance_lock
from robottraderslab.backtester.simulator import (
    CacheFillRecorder,
    FeeRates,
    FuturesSimulationEngine,
    SimulatedFuturesExchange,
    fee_model_for,
)
from robottraderslab.exceptions import (
    BotCriticalError,
    ExchangeCriticalError,
    ExchangeRecoverableError,
    ProfileConfigError,
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
from robottraderslab.ma_strategy.futures_ma_strategy import (
    FuturesMAStrategy,
    ProfileConfig,
)
from robottraderslab.strategies import ProfileStrategy
from robottraderslab_core.engine import bot_segment
from robottraderslab_core.exchanges import PlacementReserve, tag_of
from robottraderslab_core.ohlcv import OHLCVRow
from robottraderslab_core.strategies import profile_tag

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

_REFUSED_PROFILE = """
[[strategy.profiles]]
symbol = "{symbol}"
timeframe = "1d"
fast_ma_length = 2
slow_ma_length = 3
total_balance_ratio = 0.1
refused = true
"""

type _MakeMaBot = Callable[[Sequence[tuple[Symbol, str]]], Path]
type _MakeRefusingMaBot = Callable[[Sequence[tuple[Symbol, str]], Symbol], Path]


@dataclass(frozen=True, kw_only=True)
class _RefusableProfile(ProfileConfig):
    refused: bool = False

    def __post_init__(self) -> None:
        if self.refused:
            raise ProfileConfigError("this configuration is refused")
        super().__post_init__()


class RefusingMAStrategy(FuturesMAStrategy, ProfileStrategy[_RefusableProfile]):
    pass


_REFUSING_STRATEGY = f"{__name__}.RefusingMAStrategy"


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
        stop_loss_price=None,
        take_profit_price=None,
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
    folder = make_bot_folder(directory, "impulse-bot-example")
    path = folder / "bot.toml"
    path.write_text(_bot(exchange))
    return path


def _run(config_file: Path, **overrides: Any) -> int:
    arguments: dict[str, Any] = {
        "folder": config_file.parent,
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
        _BTC, OrderSide.SELL, 2.5, reduce_only=True, client_order_id=ANY
    )


def test_a_short_position_closes_with_a_buy(exchange, config_file):
    exchange.get_open_positions.return_value = {
        _BTC: _position(_BTC, side=PositionSide.SHORT, quantity=1.5)
    }

    _run(config_file, symbol="BTC/USDT:USDT")

    exchange.place_market_order.assert_awaited_once_with(
        _BTC, OrderSide.BUY, 1.5, reduce_only=True, client_order_id=ANY
    )


def test_the_close_carries_the_bots_id_under_the_flatten_tag(exchange, config_file):
    exchange.get_open_positions.return_value = {_BTC: _position(_BTC)}

    _run(config_file, symbol="BTC/USDT:USDT")

    close_id = exchange.place_market_order.await_args.kwargs["client_order_id"]
    assert close_id.startswith(bot_segment("impulse-bot-example"))
    assert tag_of(close_id) == "flatten"


@pytest.mark.parametrize("external", [False, True], ids=["full", "external"])
def test_a_config_trading_in_another_process_is_left_alone(
    exchange, config_file, caplog, external
):
    with hold_instance_lock(config_file), caplog.at_level(logging.WARNING):
        exit_code = _run(config_file, external=external)

    assert exit_code == EX_TEMPFAIL
    assert exchange.mock_calls == []
    assert f"{config_file.parent.name} is already trading" in caplog.text


def test_the_open_orders_are_read_with_their_trigger_entries(exchange, config_file):
    exchange.get_open_orders.side_effect = [[_order(_BTC)], []]

    _run(config_file)

    assert exchange.get_open_orders.await_args_list
    assert all(
        call.kwargs == {"with_triggers": True}
        for call in exchange.get_open_orders.await_args_list
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
    backtest_only = tmp_path / "backtest-only"
    backtest_only.mkdir()
    (backtest_only / "bot.toml").write_text('[strategy]\nstrategy_class = "impulse"\n')

    with pytest.raises(BotCriticalError, match=r"\[live\]"):
        main(folder=backtest_only, symbol=None, external=False)


def test_a_venue_the_engine_cannot_load_is_refused(tmp_path):
    ghost = _write_bot(tmp_path, "ghost")

    with pytest.raises(ExchangeCriticalError, match="ghost"):
        main(folder=ghost.parent, symbol=None, external=False)


@pytest.mark.parametrize(
    ("open_orders", "remaining_orders"),
    [([], []), ([_order(_BTC)], []), ([_order(_BTC)], [_order(_BTC)])],
    ids=["already flat", "flat after cancelling", "not flat"],
)
def test_every_verdict_names_the_bot(
    exchange, config_file, caplog, open_orders, remaining_orders
):
    exchange.get_open_orders.side_effect = [open_orders, remaining_orders]

    with caplog.at_level(logging.INFO):
        _run(config_file)

    assert "impulse-bot-example" in caplog.text


class _FillingSimulator(SimulatedFuturesExchange):
    """The simulator fills a market order on the next candle it is handed, and
    a venue fills one as it is placed.
    """

    placed_ids: list[str | None]

    def __init__(self, *arguments: Any, **keywords: Any) -> None:
        super().__init__(*arguments, **keywords)
        self.placed_ids = []

    async def place_market_order(
        self, symbol: Symbol, side: OrderSide, quantity: float, **kwargs: Any
    ) -> PlacedOrder:
        self.placed_ids.append(kwargs.get("client_order_id"))
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
        return _write_ma_bot(tmp_path, _ma_bot(profiles))

    return _make


@pytest.fixture
def make_refusing_ma_bot(tmp_path: Path) -> _MakeRefusingMaBot:
    def _make(profiles: Sequence[tuple[Symbol, str]], refused: Symbol) -> Path:
        return _write_ma_bot(
            tmp_path, _ma_bot(profiles, _REFUSING_STRATEGY, refused=refused)
        )

    return _make


def _write_ma_bot(directory: Path, text: str) -> Path:
    folder = make_bot_folder(directory, "ma-bot-example")
    path = folder / "bot.toml"
    path.write_text(text)
    return path


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


def _ma_bot(
    profiles: Sequence[tuple[Symbol, str]],
    strategy_class: str = "futures_ma",
    refused: Symbol | None = None,
) -> str:
    declared = "".join(
        _MA_PROFILE.format(symbol=symbol, tag=tag) for symbol, tag in profiles
    )
    if refused is not None:
        declared += _REFUSED_PROFILE.format(symbol=refused)
    return f"""
secrets_file = "creds.toml"

[strategy]
strategy_class = "{strategy_class}"
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


async def test_a_symbol_sitting_out_is_named_with_the_command_that_closes_it(
    simulator, make_refusing_ma_bot, caplog
):
    bot = make_refusing_ma_bot([(_BTC, "p1")], _ETH)
    await _a_profile_fills(simulator, _BTC, OrderSide.BUY, 0.02, "p1")
    caplog.clear()

    with caplog.at_level(logging.WARNING):
        exit_code = _run_external(bot)

    assert exit_code == 0
    assert [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING
    ] == [
        "`ETH/USDT:USDT` sits out the run because one of its profiles was "
        f"refused, so this command leaves it and `rtlab flatten {bot.parent} "
        "ETH/USDT:USDT` closes it"
    ]


async def test_a_symbol_sitting_out_is_left_alone_while_the_others_are_closed(
    simulator, make_refusing_ma_bot
):
    bot = make_refusing_ma_bot([(_BTC, "p1")], _ETH)
    await _a_profile_fills(simulator, _BTC, OrderSide.BUY, 0.02, "p1")
    await _a_hand_fills(simulator, _BTC, OrderSide.BUY, 0.06)
    await _a_hand_fills(simulator, _ETH, OrderSide.BUY, 0.5)
    simulator.placed_ids.clear()

    exit_code = _run_external(bot)

    assert exit_code == 0
    assert len(simulator.placed_ids) == 1
    assert simulator.simulation_engine.open_positions[_BTC].quantity == 0.02
    assert simulator.simulation_engine.open_positions[_ETH].quantity == 0.5


async def test_a_symbol_the_bot_does_not_declare_is_left_alone(
    simulator, make_ma_bot, caplog
):
    bot = make_ma_bot([(_BTC, "p1"), (_SOL, "p1")])
    await _a_profile_fills(simulator, _BTC, OrderSide.BUY, 0.02, "p1")
    await _a_hand_fills(simulator, _BTC, OrderSide.BUY, 0.06)
    await _a_hand_fills(simulator, _ETH, OrderSide.BUY, 0.5)
    caplog.clear()

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
    untracked = make_bot_folder(tmp_path, "ma-bot-example") / "bot.toml"
    untracked.write_text(_ma_bot([]))

    with pytest.raises(BotCriticalError, match="tracks no profile"):
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


async def test_the_external_close_carries_the_bots_id_and_no_profile_owns_it(
    simulator, make_ma_bot
):
    bot = make_ma_bot([(_BTC, "p1")])
    await _a_profile_fills(simulator, _BTC, OrderSide.BUY, 0.02, "p1")
    await _a_hand_fills(simulator, _BTC, OrderSide.BUY, 0.06)
    simulator.placed_ids.clear()

    first_exit_code = _run_external(bot)
    second_exit_code = _run_external(bot)

    (close_id,) = simulator.placed_ids
    assert close_id.startswith(bot_segment("ma-bot-example"))
    assert tag_of(close_id) == "flatten"
    assert first_exit_code == 0
    assert second_exit_code == 0
    assert simulator.simulation_engine.open_positions[_BTC].quantity == 0.02


_NOT_TOML = "[strategy\n"
_NO_LIVE_SECTION = '[strategy]\nstrategy_class = "impulse"\n'


@pytest.mark.parametrize("symbol", [None, "BTC/USDT:USDT"], ids=["all", "symbol"])
def test_flattening_leaves_the_working_copy_untouched(exchange, config_file, symbol):
    exchange.get_open_positions.return_value = {_BTC: _position(_BTC)}
    copy = seed_old_copy(config_file.parent)

    _run(config_file, symbol=symbol)

    assert copy.stat().st_mtime_ns == AN_OLD_MOMENT_NS


async def test_the_external_flatten_leaves_the_working_copy_untouched(
    simulator, make_ma_bot
):
    bot = make_ma_bot([(_BTC, "p1")])
    copy = seed_old_copy(bot.parent)
    await _a_profile_fills(simulator, _BTC, OrderSide.BUY, 0.02, "p1")
    await _a_hand_fills(simulator, _BTC, OrderSide.BUY, 0.06)

    _run_external(bot)

    assert copy.stat().st_mtime_ns == AN_OLD_MOMENT_NS


@pytest.mark.parametrize(
    "edited_bot_toml",
    [_NOT_TOML, _bot("ghost")],
    ids=["not valid TOML", "another venue"],
)
def test_an_edited_bot_toml_does_not_change_the_account_flattened(
    exchange, config_file, edited_bot_toml
):
    exchange.get_open_positions.return_value = {_BTC: _position(_BTC, quantity=2.5)}
    seed_old_copy(config_file.parent)
    config_file.write_text(edited_bot_toml)

    _run(config_file, symbol="BTC/USDT:USDT")

    exchange.place_market_order.assert_awaited_once_with(
        _BTC, OrderSide.SELL, 2.5, reduce_only=True, client_order_id=ANY
    )


@pytest.mark.parametrize(
    "edited_bot_toml",
    [_NOT_TOML, _ma_bot([(_BTC, "p1")])],
    ids=["not valid TOML", "fewer profiles"],
)
async def test_an_edited_bot_toml_does_not_change_the_profiles_attributed_to(
    simulator, make_ma_bot, edited_bot_toml
):
    bot = make_ma_bot([(_BTC, "p1"), (_BTC, "p2")])
    seed_old_copy(bot.parent)
    bot.write_text(edited_bot_toml)
    await _a_profile_fills(simulator, _BTC, OrderSide.BUY, 0.02, "p1")
    await _a_profile_fills(simulator, _BTC, OrderSide.BUY, 0.03, "p2")
    await _a_hand_fills(simulator, _BTC, OrderSide.BUY, 0.06)

    exit_code = _run_external(bot)

    assert exit_code == 0
    assert simulator.simulation_engine.open_positions[_BTC].quantity == 0.05


def test_a_copy_without_a_live_section_is_refused_naming_the_copy(
    exchange, config_file
):
    seeded(config_file.parent, _NO_LIVE_SECTION)
    config_file.write_text(_bot(_PLUGGED_EXCHANGE))

    with pytest.raises(BotCriticalError, match=r"bot\.working\.toml.*\[live\]"):
        _run(config_file, symbol="BTC/USDT:USDT")
