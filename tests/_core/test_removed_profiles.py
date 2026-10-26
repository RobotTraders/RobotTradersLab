import asyncio
import logging
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest

from robottraderslab import Symbol
from robottraderslab._core import (
    clean_removed_profiles,
    name_process,
)
from robottraderslab.exchanges import (
    Execution,
    FuturesExchangeProtocol,
    OrderSide,
    PositionSide,
    PositionSnapshot,
    next_client_order_id,
)
from robottraderslab.futures import ClosePositionAction
from robottraderslab.strategies import (
    AccountSnapshot,
    AccountSnapshots,
    BookKeeper,
    OrderProtocol,
    StrategyRequirements,
)
from robottraderslab.strategies.futures import CancelOrderByIdAction

ACCOUNT_NAME = "test"
DOGE = Symbol.create("DOGE/USDT:USDT")
BTC = Symbol.create("BTC/USDT:USDT")
LAUNCH = datetime(2026, 10, 22, 8, 0, tzinfo=UTC)
FILLED_AT = datetime(2026, 10, 20, tzinfo=UTC)
THIS_BOT = "impulse-kraken"
ANOTHER_BOT = "envelope-bitget"


class _Account:
    def __init__(self) -> None:
        self.name = ACCOUNT_NAME
        self.reads: list[str] = []
        self.longer_read: list[Execution] = []

    def cancel_order(self, symbol: Symbol, order_id: str) -> CancelOrderByIdAction:
        return CancelOrderByIdAction(
            exchange=Mock(spec=FuturesExchangeProtocol),
            symbol=symbol,
            order_id=order_id,
        )

    def _close_removed_position(self, symbol: Symbol) -> ClosePositionAction:
        return ClosePositionAction(
            exchange=Mock(spec=FuturesExchangeProtocol), symbol=symbol, tag="flatten"
        )

    async def _executions_since(
        self, since: datetime, symbols: object, **_: object
    ) -> list[Execution]:
        self.reads.append(since.date().isoformat())
        return self.longer_read


def _booked(bookkeeper: BookKeeper) -> list[tuple[str, ...]]:
    return [
        ("close", str(action.symbol), action.tag)
        if isinstance(action, ClosePositionAction)
        else ("cancel", str(action.symbol), action.order_id)
        for action in bookkeeper.list_actions()
    ]


@pytest.fixture
def account() -> _Account:
    return _Account()


@pytest.fixture
def own_id() -> str:
    name_process(THIS_BOT, LAUNCH)
    return next_client_order_id("1h-alpha")


@pytest.fixture
def other_bot_id() -> str:
    name_process(ANOTHER_BOT, LAUNCH)
    identifier = next_client_order_id("1h-alpha")
    name_process(THIS_BOT, LAUNCH)
    return identifier


def _order(order_id: str, symbol: Symbol, client_order_id: str) -> Mock:
    order = Mock(spec=OrderProtocol)
    order.order_id = order_id
    order.symbol = symbol
    order.client_order_id = client_order_id
    return order


def _position(symbol: Symbol, quantity: float) -> PositionSnapshot:
    return PositionSnapshot(
        symbol=symbol,
        side=PositionSide.LONG,
        quantity=quantity,
        average_entry_price=100.0,
        entry_time=FILLED_AT,
        leverage=1.0,
        liquidation_price=0.0,
    )


def _fill(symbol: Symbol, quantity: float, client_order_id: str) -> Execution:
    return Execution(
        execution_id=f"{symbol}-{quantity}-{client_order_id}",
        order_id="order",
        symbol=symbol,
        side=OrderSide.BUY,
        price=100.0,
        quantity=quantity,
        timestamp=FILLED_AT,
        client_order_id=client_order_id,
    )


def _clean(
    account: _Account,
    *,
    positions: dict[Symbol, PositionSnapshot],
    orders: list[Mock],
    executions: list[Execution],
    clean: bool = True,
    declares_orders: bool = True,
    declared: Symbol = BTC,
) -> BookKeeper:
    bookkeeper = BookKeeper()
    requirements = StrategyRequirements().account
    requirements.add(
        account, symbols=(declared,), positions=True, open_orders=declares_orders
    )
    snapshots = AccountSnapshots(
        {
            ACCOUNT_NAME: AccountSnapshot(
                account_name=ACCOUNT_NAME,
                positions=positions,
                open_orders=orders,
                executions=executions,
            )
        }
    )
    asyncio.run(
        clean_removed_profiles(
            requirements,
            snapshots,
            bookkeeper,
            clean=clean,
            max_attempts=1,
            base_delay=0.0,
        )
    )
    return bookkeeper


def test_a_removed_profiles_position_is_closed_before_its_orders_are_cancelled(
    account, own_id, caplog
):
    rungs = [_order(f"rung-{number}", DOGE, own_id) for number in (1, 2, 3)]

    with caplog.at_level(logging.WARNING):
        bookkeeper = _clean(
            account,
            positions={DOGE: _position(DOGE, 40.0)},
            orders=rungs,
            executions=[_fill(DOGE, 40.0, own_id)],
        )

    assert _booked(bookkeeper) == [
        ("close", str(DOGE), "flatten"),
        ("cancel", str(DOGE), "rung-1"),
        ("cancel", str(DOGE), "rung-2"),
        ("cancel", str(DOGE), "rung-3"),
    ]
    assert str(DOGE) in caplog.text
    assert "closing +40" in caplog.text
    assert "rung-1, rung-2, rung-3" in caplog.text


def test_a_position_the_bots_fills_do_not_explain_is_left_and_its_rung_cancelled(
    account, own_id, caplog
):
    with caplog.at_level(logging.WARNING):
        bookkeeper = _clean(
            account,
            positions={DOGE: _position(DOGE, 40.0)},
            orders=[_order("rung", DOGE, own_id)],
            executions=[_fill(DOGE, 10.0, own_id)],
        )

    assert _booked(bookkeeper) == [("cancel", str(DOGE), "rung")]
    assert "do not explain" in caplog.text


def test_another_bots_orders_on_the_symbol_are_left(
    account, own_id, other_bot_id, caplog
):
    with caplog.at_level(logging.INFO):
        bookkeeper = _clean(
            account,
            positions={},
            orders=[
                _order("mine", DOGE, own_id),
                _order("theirs", DOGE, other_bot_id),
            ],
            executions=[],
        )

    assert _booked(bookkeeper) == [("cancel", str(DOGE), "mine")]
    assert "not placed by this bot" in caplog.text


def test_a_symbol_the_bot_never_touched_is_left_without_a_read(account, other_bot_id):
    bookkeeper = _clean(
        account,
        positions={DOGE: _position(DOGE, 40.0)},
        orders=[_order("theirs", DOGE, other_bot_id)],
        executions=[_fill(DOGE, 40.0, other_bot_id)],
    )

    assert _booked(bookkeeper) == []
    assert account.reads == []


def test_a_symbol_typed_over_with_another_listed_contract_is_cleaned(
    account, own_id, caplog
):
    with caplog.at_level(logging.WARNING):
        bookkeeper = _clean(
            account,
            positions={BTC: _position(BTC, 1.0)},
            orders=[_order("rung", BTC, own_id)],
            executions=[_fill(BTC, 1.0, own_id)],
            declared=Symbol.create("BTC/USDC:USDC"),
        )

    assert _booked(bookkeeper) == [
        ("close", str(BTC), "flatten"),
        ("cancel", str(BTC), "rung"),
    ]
    assert str(BTC) in caplog.text


def test_a_declared_symbol_is_not_cleaned(account, own_id):
    bookkeeper = _clean(
        account,
        positions={BTC: _position(BTC, 1.0)},
        orders=[_order("rung", BTC, own_id)],
        executions=[_fill(BTC, 1.0, own_id)],
    )

    assert _booked(bookkeeper) == []


def test_the_key_off_names_what_would_be_cleaned_and_books_nothing(
    account, own_id, caplog
):
    with caplog.at_level(logging.WARNING):
        bookkeeper = _clean(
            account,
            positions={DOGE: _position(DOGE, 40.0)},
            orders=[_order("rung", DOGE, own_id)],
            executions=[_fill(DOGE, 40.0, own_id)],
            clean=False,
        )

    assert _booked(bookkeeper) == []
    assert "would close +40" in caplog.text


def test_a_position_older_than_the_window_is_explained_by_the_longer_read(
    account, own_id
):
    account.longer_read = [_fill(DOGE, 40.0, own_id)]

    bookkeeper = _clean(
        account,
        positions={DOGE: _position(DOGE, 40.0)},
        orders=[_order("rung", DOGE, own_id)],
        executions=[],
    )

    assert _booked(bookkeeper) == [
        ("close", str(DOGE), "flatten"),
        ("cancel", str(DOGE), "rung"),
    ]
    read_from = datetime.fromisoformat(account.reads[0]).date()
    assert read_from <= (datetime.now(UTC) - timedelta(days=89)).date()


def test_an_account_declaring_no_open_orders_is_not_searched(account, own_id):
    bookkeeper = _clean(
        account,
        positions={DOGE: _position(DOGE, 40.0)},
        orders=[],
        executions=[_fill(DOGE, 40.0, own_id)],
        declares_orders=False,
    )

    assert _booked(bookkeeper) == []
