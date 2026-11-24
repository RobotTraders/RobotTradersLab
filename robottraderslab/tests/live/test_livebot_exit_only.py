import logging
from collections.abc import Callable
from unittest.mock import Mock

import pytest
from live_cycle_helpers import (
    BTC,
    ETH,
    Booking,
    booking,
    cancelled_ids,
    long_position,
    market_orders,
    run,
)

from robottraderslab import Symbol
from robottraderslab.exchanges import (
    OrderSide,
    OrderType,
)
from robottraderslab.futures import FuturesAccount, FuturesCapabilities
from robottraderslab.strategies import (
    BookKeeper,
    Profile,
    ProfileStrategy,
    TradingMode,
    TradingSystem,
)

DOGE = Symbol.create("DOGE/USDT:USDT")
NOT_THIS_BOT = "aaaaaa000000-1-grid"

pytestmark = pytest.mark.usefixtures("candle_boundary_clock")


@pytest.fixture
def exchange(make_live_venue: Callable[[], Mock]) -> Mock:
    return make_live_venue()


def _lines(caplog: pytest.LogCaptureFixture, level: int, text: str) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno == level and text in record.getMessage()
    ]


def test_an_exit_is_sent_an_entry_dropped_and_the_other_symbol_trades(
    account, exchange, caplog
):
    caplog.set_level(logging.INFO)
    exchange.get_open_positions.return_value = {BTC: long_position(BTC, 1.0)}

    def book(bookkeeper: BookKeeper) -> None:
        bookkeeper.add(account.long_exit(BTC, 1.0, reduce_only=False))
        bookkeeper.add(account.long_entry(BTC, 1.0))
        bookkeeper.add(account.long_entry(ETH, 2.0))

    run(booking(account, book), BTC)

    assert market_orders(exchange) == [
        (BTC, OrderSide.SELL, 1.0),
        (ETH, OrderSide.BUY, 2.0),
    ]
    assert len(_lines(caplog, logging.INFO, "exit-only mode drops")) == 1


def test_a_filter_that_fails_stops_the_cycle_before_any_order_is_sent(
    account, exchange
):
    exchange.get_open_positions.return_value = {BTC: long_position(BTC, 1.0)}
    exchange.get_open_orders.return_value = [object()]

    def book(bookkeeper: BookKeeper) -> None:
        bookkeeper.add(account.long_entry(BTC, 1.0))

    with pytest.raises(AttributeError):
        run(booking(account, book), BTC)

    exchange.place_market_order.assert_not_awaited()


def test_an_exit_larger_than_the_position_is_cut_to_it_without_its_stop_loss(
    account, exchange, caplog
):
    exchange.get_open_positions.return_value = {BTC: long_position(BTC, 1.0)}

    def book(bookkeeper: BookKeeper) -> None:
        bookkeeper.add(account.long_exit(BTC, 3.0, reduce_only=False).stop_loss(90.0))

    run(booking(account, book), BTC)

    sent = exchange.place_market_order.await_args.kwargs
    assert sent["quantity"] == pytest.approx(1.0)
    assert sent["stop_loss"] is None
    assert len(_lines(caplog, logging.WARNING, "exit-only mode cuts")) == 1


def test_a_move_of_the_stop_loss_is_sent(account, exchange, make_protection):
    exchange.get_open_positions.return_value = {BTC: long_position(BTC, 1.0)}
    exchange.get_open_orders.return_value = [make_protection("sl-1")]

    run(
        booking(
            account,
            lambda bookkeeper: bookkeeper.add(account.move_stop_loss(BTC, 95.0)),
        ),
        BTC,
    )

    exchange.update_position_stop_loss.assert_awaited_once()


def test_a_move_of_the_take_profit_is_sent(account, exchange, make_protection):
    exchange.get_open_positions.return_value = {BTC: long_position(BTC, 1.0)}
    exchange.get_open_orders.return_value = [
        make_protection("tp-1", OrderType.TAKE_PROFIT)
    ]

    run(
        booking(
            account,
            lambda bookkeeper: bookkeeper.add(account.move_take_profit(BTC, 120.0)),
        ),
        BTC,
    )

    exchange.update_position_take_profit.assert_awaited_once()


def test_a_leverage_setting_is_sent(account, exchange):
    exchange.get_open_positions.return_value = {BTC: long_position(BTC, 1.0)}

    run(
        booking(
            account, lambda bookkeeper: bookkeeper.add(account.set_leverage(BTC, 3.0))
        ),
        BTC,
    )

    exchange.set_leverage.assert_awaited_once_with(BTC, 3.0)


@pytest.mark.parametrize(
    ("positions", "cancel_awaited"),
    [({}, True), ({BTC: long_position(BTC, 1.0)}, False)],
    ids=["flat symbol", "symbol holding a position"],
)
def test_a_cancel_of_every_order_is_sent_only_on_a_symbol_holding_no_position(
    account, exchange, positions, cancel_awaited
):
    exchange.get_open_positions.return_value = positions

    def book(bookkeeper: BookKeeper) -> None:
        bookkeeper.add(account.cancel_orders(BTC))

    run(booking(account, book), BTC)

    assert exchange.cancel_orders_for_symbol.await_count == int(cancel_awaited)


def test_a_cancel_of_a_stop_loss_is_dropped_while_the_position_stands(
    account, exchange, make_protection, caplog
):
    caplog.set_level(logging.INFO)
    exchange.get_open_positions.return_value = {BTC: long_position(BTC, 1.0)}
    exchange.get_open_orders.return_value = [make_protection("sl-1")]

    def book(bookkeeper: BookKeeper) -> None:
        bookkeeper.add(account.cancel_order(BTC, "sl-1"))

    run(booking(account, book), BTC)

    exchange.cancel_order_by_id.assert_not_awaited()
    assert len(_lines(caplog, logging.INFO, "exit-only mode drops")) == 1


def test_a_price_move_of_a_resting_exit_is_sent(account, exchange, make_resting):
    exchange.get_open_positions.return_value = {BTC: long_position(BTC, 1.0)}
    exchange.get_open_orders.return_value = [make_resting(order_id="rung-1")]

    def book(bookkeeper: BookKeeper) -> None:
        bookkeeper.add(
            account.modify_order(
                "rung-1", account.long_exit(BTC, 0.5, reduce_only=False).limit(106.0)
            )
        )

    run(booking(account, book), BTC)

    exchange.modify_orders.assert_awaited_once()


def test_a_modification_raising_a_resting_orders_quantity_is_dropped(
    account, exchange, make_resting
):
    exchange.get_open_positions.return_value = {BTC: long_position(BTC, 1.0)}
    exchange.get_open_orders.return_value = [make_resting(order_id="rung-1")]

    def book(bookkeeper: BookKeeper) -> None:
        bookkeeper.add(
            account.modify_order(
                "rung-1", account.long_exit(BTC, 0.6, reduce_only=False).limit(106.0)
            )
        )

    run(booking(account, book), BTC)

    exchange.modify_orders.assert_not_awaited()


def test_a_batch_spanning_an_exit_only_symbol_and_another_filters_only_the_first(
    account, exchange
):
    def book(bookkeeper: BookKeeper) -> None:
        bookkeeper.add(
            account.place_orders(
                [
                    account.long_entry(BTC, 1.0).limit(99.0),
                    account.long_entry(ETH, 2.0).limit(98.0),
                ]
            )
        )

    run(booking(account, book), BTC)

    requests = exchange.place_orders.await_args.args[0]
    assert [request.symbol for request in requests] == [ETH]
    assert requests[0].quantity == 2.0


def test_a_cycle_closing_the_whole_position_cancels_its_stop_loss_and_every_resting_order(
    account, exchange, make_protection, make_resting
):
    exchange.get_open_positions.return_value = {BTC: long_position(BTC, 1.0)}
    exchange.get_open_orders.return_value = [
        make_protection("sl-1"),
        make_resting(order_id="rung-1", side=OrderSide.BUY),
        make_resting(order_id="rung-2", quantity=0.3, remaining_quantity=0.3),
    ]

    def book(bookkeeper: BookKeeper) -> None:
        bookkeeper.add(account.long_exit(BTC, 1.0, reduce_only=False))
        bookkeeper.add(account.cancel_order(BTC, "sl-1"))

    run(booking(account, book), BTC)

    assert market_orders(exchange) == [(BTC, OrderSide.SELL, 1.0)]
    assert sorted(cancelled_ids(exchange)) == ["rung-1", "rung-2", "sl-1"]


def test_the_resting_orders_are_cancelled_before_the_orders_the_strategy_books(
    account, exchange, make_resting
):
    exchange.get_open_positions.return_value = {BTC: long_position(BTC, 1.0)}
    exchange.get_open_orders.return_value = [
        make_resting(order_id="rung-1", side=OrderSide.BUY)
    ]

    def book(bookkeeper: BookKeeper) -> None:
        bookkeeper.add(account.long_exit(BTC, 1.0, reduce_only=False))

    run(booking(account, book), BTC)

    names = [name for name, _, _ in exchange.method_calls]
    assert names.index("cancel_order_by_id") < names.index("place_market_order")


def test_an_action_waiting_on_a_dropped_action_is_sent(account, exchange):
    exchange.get_open_positions.return_value = {BTC: long_position(BTC, 1.0)}

    def book(bookkeeper: BookKeeper) -> None:
        entry = bookkeeper.add(account.long_entry(BTC, 1.0))
        bookkeeper.add(account.long_exit(BTC, 1.0, reduce_only=False), after=entry)

    run(booking(account, book), BTC)

    assert market_orders(exchange) == [(BTC, OrderSide.SELL, 1.0)]


def test_the_resting_entries_beyond_what_the_position_holds_are_cancelled(
    account, exchange, make_protection, make_resting
):
    exchange.get_open_positions.return_value = {BTC: long_position(BTC, 1.0)}
    exchange.get_open_orders.return_value = [
        make_resting(order_id="on-the-positions-side", side=OrderSide.BUY),
        make_resting(order_id="older-exit", quantity=0.6, remaining_quantity=0.6),
        make_resting(order_id="newer-exit", quantity=0.6, remaining_quantity=0.6),
        make_resting(order_id="reduce-only-exit", quantity=5.0, reduce_only=True),
        make_protection("sl-1"),
        make_protection("tp-1", OrderType.TAKE_PROFIT),
    ]

    run(booking(account, lambda bookkeeper: None), BTC)

    assert sorted(cancelled_ids(exchange)) == ["newer-exit", "on-the-positions-side"]


def test_every_resting_entry_is_cancelled_once_a_stop_loss_closed_the_position(
    account, exchange, make_resting
):
    exchange.get_open_orders.return_value = [
        make_resting(order_id="buy-rung", side=OrderSide.BUY),
        make_resting(order_id="sell-rung", side=OrderSide.SELL),
    ]

    run(booking(account, lambda bookkeeper: None), BTC)

    assert sorted(cancelled_ids(exchange)) == ["buy-rung", "sell-rung"]


class _SlotsStrategy(ProfileStrategy[Profile]):
    market_type = "futures"

    def __init__(self, account: FuturesAccount) -> None:
        super().__init__(
            account=account,
            trading_system=TradingSystem(trading_mode=TradingMode.LIVE),
            config_dir=None,
            profiles=[{"symbol": str(BTC), "timeframe": "1d"}],
        )

    async def setup(self, requirements):
        for symbol in (BTC, ETH):
            requirements.ohlcv.add(symbol, "1d")
        requirements.account.add(
            self.account, symbols=[BTC, ETH], positions=True, open_orders=True
        )

    def book_general_actions(
        self, ohlcvs, account_snapshots, timestamp, bookkeeper, triggered_timeframes
    ):
        bookkeeper.add(self.account.long_entry(BTC, 1.0))
        bookkeeper.add(self.account.long_entry(ETH, 2.0))

    def book_final_actions(
        self, ohlcvs, account_snapshots, timestamp, bookkeeper, triggered_timeframes
    ):
        bookkeeper.add(self.account.long_entry(BTC, 3.0))
        bookkeeper.add(self.account.long_entry(ETH, 4.0))


def test_the_entries_of_the_general_and_the_final_slot_are_dropped_on_an_exit_only_symbol(
    account, exchange
):
    run(_SlotsStrategy(account), BTC)

    assert market_orders(exchange) == [
        (ETH, OrderSide.BUY, 2.0),
        (ETH, OrderSide.BUY, 4.0),
    ]


def test_the_entries_of_the_general_and_the_final_slot_are_sent_on_a_symbol_running_as_usual(
    account, exchange
):
    run(_SlotsStrategy(account))

    assert market_orders(exchange) == [
        (BTC, OrderSide.BUY, 1.0),
        (ETH, OrderSide.BUY, 2.0),
        (BTC, OrderSide.BUY, 3.0),
        (ETH, OrderSide.BUY, 4.0),
    ]


def test_the_sweeps_cancel_of_a_stop_loss_on_a_flat_exit_only_symbol_is_sent(
    account, exchange, make_protection
):
    exchange.get_open_orders.return_value = [make_protection("sl-1")]

    run(booking(account, lambda bookkeeper: None), BTC)

    exchange.cancel_order_by_id.assert_awaited_once_with(BTC, "sl-1")


def test_a_strategy_declaring_no_state_has_its_exit_only_symbol_read_and_its_trigger_cancelled(
    account, exchange, make_resting, make_protection
):
    exchange.get_open_positions.return_value = {DOGE: long_position(DOGE, 40.0)}
    exchange.get_open_orders.return_value = [
        make_resting(order_id="trigger-1", kind=OrderType.TRIGGER, side=OrderSide.BUY),
        make_resting(order_id="doge-rung", symbol=DOGE, side=OrderSide.BUY),
        make_protection("sl-1"),
    ]

    run(Booking(account, lambda bookkeeper: None), BTC)

    exchange.get_open_positions.assert_awaited_once()
    assert exchange.get_open_orders.await_args.kwargs == {"with_triggers": True}
    assert cancelled_ids(exchange) == ["trigger-1"]
    exchange.place_market_order.assert_not_awaited()


def test_a_trigger_carrying_no_id_is_cancelled_where_the_venue_keeps_none(
    account, exchange, make_resting
):
    exchange.capabilities = FuturesCapabilities(labels_resting_triggers=False)
    exchange.get_open_orders.return_value = [
        make_resting(
            order_id="trigger-1",
            kind=OrderType.TRIGGER,
            side=OrderSide.BUY,
            client_order_id=None,
        )
    ]

    run(booking(account, lambda bookkeeper: None), BTC)

    assert cancelled_ids(exchange) == ["trigger-1"]


def test_a_trigger_carrying_no_id_is_left_where_the_venue_keeps_it(
    account, exchange, make_resting
):
    exchange.get_open_orders.return_value = [
        make_resting(
            order_id="trigger-1",
            kind=OrderType.TRIGGER,
            side=OrderSide.BUY,
            client_order_id=None,
        ),
        make_resting(order_id="foreign", client_order_id=NOT_THIS_BOT),
    ]

    run(booking(account, lambda bookkeeper: None), BTC)

    exchange.cancel_order_by_id.assert_not_awaited()


def test_a_cycle_with_no_exit_only_symbol_reads_what_the_strategy_declared_and_sends_everything(
    account, exchange
):
    def book(bookkeeper: BookKeeper) -> None:
        bookkeeper.add(account.long_entry(BTC, 1.0))

    run(Booking(account, book))

    assert market_orders(exchange) == [(BTC, OrderSide.BUY, 1.0)]
    exchange.get_open_positions.assert_not_awaited()
    exchange.get_open_orders.assert_not_awaited()
