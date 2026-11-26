import logging
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import time_machine
from live_bot_helpers import BOT_FOLDER, FAILING_VENUE, VENUE, forget_probes
from live_cycle_helpers import BTC, make_bot_folder

from robottraderslab import Symbol
from robottraderslab.bootstrap import EXCHANGE_ENTRY_POINT_GROUP
from robottraderslab.exceptions import ExchangeCriticalError
from robottraderslab.exchanges import (
    MarginMode,
    MarginSettings,
    OrderProtocol,
    OrderSide,
    OrderType,
    PlacedOrder,
)
from robottraderslab.futures import FuturesAccount, FuturesCapabilities


@pytest.fixture
def venue(
    make_live_venue: Callable[[], Mock],
    make_registered_adapter: Callable[..., None],
    candle_boundary_clock: None,
) -> Mock:
    """The venue a bot of `live_bot_helpers` trades on, registered as its connector.

    It holds each symbol at leverage 1 until the cycle sets another, and keeps
    the settings its connector was built with in `received_settings`.
    """
    plugged = make_live_venue()
    plugged.capabilities = FuturesCapabilities(
        reserved_order_fields=frozenset({"reduceOnly"})
    )
    plugged.received_settings = {}
    _hold_margin_settings(plugged)

    def make(**settings: Any) -> Mock:
        plugged.received_settings = settings
        return plugged

    def refuse(**settings: Any) -> Mock:
        raise ExchangeCriticalError("the venue refuses these keys")

    make_registered_adapter(make, VENUE, group=EXCHANGE_ENTRY_POINT_GROUP)
    make_registered_adapter(refuse, FAILING_VENUE, group=EXCHANGE_ENTRY_POINT_GROUP)
    return plugged


@pytest.fixture
def bot_folder(tmp_path: Path, venue: Mock) -> Iterator[Path]:
    """A bot folder holding its secrets file, its log closed after the test.

    The slots the probe strategies record start empty.
    """
    folder = make_bot_folder(tmp_path, BOT_FOLDER)
    forget_probes()
    yield folder
    forget_probes()
    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.close()
        root.removeHandler(handler)


@pytest.fixture
def candle_boundary_clock() -> Iterator[None]:
    with time_machine.travel(datetime(2026, 1, 1, tzinfo=UTC), tick=False):
        yield


@pytest.fixture
def exchange(make_live_venue: Callable[[], Mock]) -> Mock:
    return make_live_venue()


@pytest.fixture
def account(exchange: Mock) -> FuturesAccount:
    return FuturesAccount(exchange)


@pytest.fixture
def make_resting(
    create_resting_order: Callable[..., OrderProtocol],
) -> Callable[..., OrderProtocol]:
    def make(**fields) -> OrderProtocol:
        defaults = {
            "reduce_only": False,
            "side": OrderSide.SELL,
            "quantity": 0.5,
            "remaining_quantity": 0.5,
        }
        return create_resting_order(**{**defaults, **fields})

    return make


@pytest.fixture
def make_live_venue(make_venue: Callable[..., Mock]) -> Callable[[], Mock]:
    def make() -> Mock:
        venue = make_venue()
        venue.capabilities = FuturesCapabilities()
        venue.get_open_positions.return_value = {}
        venue.get_open_orders.return_value = []
        venue.place_market_order.return_value = PlacedOrder(order_id="market-1")
        venue.place_limit_order.return_value = PlacedOrder(order_id="limit-1")
        venue.update_position_stop_loss.return_value = PlacedOrder(order_id="sl-1")
        venue.update_position_take_profit.return_value = PlacedOrder(order_id="tp-1")
        venue.place_orders.side_effect = lambda requests: [
            PlacedOrder(order_id=f"batch-{index}") for index, _ in enumerate(requests)
        ]
        venue.modify_orders.side_effect = lambda requests: [
            PlacedOrder(order_id=request.order_id) for request in requests
        ]
        return venue

    return make


@pytest.fixture
def make_protection(
    create_resting_order: Callable[..., OrderProtocol],
) -> Callable[..., OrderProtocol]:
    def make(
        order_id: str, kind: OrderType = OrderType.STOP_LOSS, symbol: Symbol = BTC
    ) -> OrderProtocol:
        return create_resting_order(
            order_id=order_id,
            symbol=symbol,
            kind=kind,
            limit_price=None,
            reduce_only=True,
        )

    return make


def _hold_margin_settings(venue: Mock) -> None:
    """Make the venue hold each symbol's leverage at 1 until a cycle sets another."""
    held: dict[Symbol, MarginSettings] = {}

    def margin_settings(symbols: Any) -> dict[Symbol, MarginSettings]:
        return {
            symbol: held.get(
                symbol, MarginSettings(leverage=1.0, margin_mode=MarginMode.ISOLATED)
            )
            for symbol in symbols
        }

    def set_leverage(symbol: Symbol, leverage: float) -> None:
        held[symbol] = MarginSettings(
            leverage=leverage, margin_mode=MarginMode.ISOLATED
        )

    venue.get_margin_settings.side_effect = margin_settings
    venue.set_leverage.side_effect = set_leverage
