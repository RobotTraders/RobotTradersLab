import asyncio
import logging
from collections.abc import Callable, Collection, Mapping
from datetime import UTC, datetime
from unittest.mock import Mock

import pytest
from dry_run_helpers import dry_run_at_debug
from live_cycle_helpers import (
    BTC,
    ETH,
    Booking,
    BookingStrategy,
    booking,
    cancelled_ids,
    livebot,
    long_position,
    make_provider,
    market_orders,
    messages_at,
    run,
)

from robottraderslab import Symbol
from robottraderslab.exceptions import SymbolNotListedError
from robottraderslab.exchanges import (
    BaseExchangeAction,
    Closure,
    Execution,
    Instrument,
    MarginMode,
    MarginSettings,
    OrderSide,
    OrderType,
    PositionSide,
    TradingStatus,
    client_order_id_carrying,
)
from robottraderslab.futures import (
    FuturesAccount,
    OrderModification,
)
from robottraderslab.strategies import (
    BookKeeper,
    TrackedPosition,
    TrackingId,
    profile_tag,
)

VENUE = "bitget"
ACCOUNT = "bench-live"

NEW_POSITIONS_CLOSED = Closure(TradingStatus.CLOSED_TO_NEW_POSITIONS, "limit_open")
EVERY_ORDER_CLOSED = Closure(TradingStatus.CLOSED_TO_EVERY_ORDER, "maintain")

pytestmark = pytest.mark.usefixtures("candle_boundary_clock")


@pytest.fixture
def other_exchange(make_live_venue: Callable[[], Mock]) -> Mock:
    return make_live_venue()


@pytest.fixture
def account(exchange: Mock) -> FuturesAccount:
    return FuturesAccount(exchange, name=ACCOUNT, venue=VENUE)


@pytest.fixture
def other_account(other_exchange: Mock) -> FuturesAccount:
    return FuturesAccount(other_exchange, name="other-account", venue=VENUE)


@pytest.fixture
def report_statuses(exchange: Mock) -> Callable[..., None]:
    def report(
        closures: Mapping[Symbol, Closure],
        venue: Mock = exchange,
        unlisted: Collection[Symbol] = (),
    ) -> None:
        async def get_instrument(symbol: Symbol) -> Instrument:
            if symbol in unlisted:
                raise SymbolNotListedError(f"{symbol} is not listed")
            return Instrument(
                quantity_step=1e-8,
                minimum_order_value=None,
                maximum_leverage=None,
                maximum_order_quantity=None,
                maximum_market_order_quantity=None,
                closure=closures.get(symbol),
            )

        venue.get_instrument.side_effect = get_instrument

    return report


class _TwoAccounts(BookingStrategy):
    def __init__(
        self,
        accounts: tuple[FuturesAccount, FuturesAccount],
        book: Callable[[BookKeeper], None],
    ) -> None:
        super().__init__(book)
        self._accounts = accounts

    async def setup(self, requirements):
        requirements.ohlcv.add(BTC, "1d")
        for account in self._accounts:
            requirements.account.add(
                account, symbols=[BTC], positions=True, open_orders=True
            )


def _entries_and_an_exit(account: FuturesAccount, bookkeeper: BookKeeper) -> None:
    bookkeeper.add(account.long_exit(BTC, 1.0, reduce_only=False))
    bookkeeper.add(account.long_entry(BTC, 1.0))
    bookkeeper.add(account.long_entry(ETH, 2.0))


class TestClosedToNewPositions:
    def test_an_entry_is_dropped_while_the_exit_and_the_other_symbol_trade(
        self, account, exchange, report_statuses
    ):
        report_statuses({BTC: NEW_POSITIONS_CLOSED})
        exchange.get_open_positions.return_value = {BTC: long_position(BTC, 1.0)}

        run(
            booking(
                account, lambda bookkeeper: _entries_and_an_exit(account, bookkeeper)
            )
        )

        assert market_orders(exchange) == [
            (BTC, OrderSide.SELL, 1.0),
            (ETH, OrderSide.BUY, 2.0),
        ]

    def test_a_move_of_the_stop_loss_is_sent(
        self, account, exchange, report_statuses, make_protection
    ):
        report_statuses({BTC: NEW_POSITIONS_CLOSED})
        exchange.get_open_positions.return_value = {BTC: long_position(BTC, 1.0)}
        exchange.get_open_orders.return_value = [make_protection("sl-1")]

        run(
            booking(
                account,
                lambda bookkeeper: bookkeeper.add(account.move_stop_loss(BTC, 95.0)),
            )
        )

        exchange.update_position_stop_loss.assert_awaited_once()

    def test_a_trigger_entry_is_cancelled_of_a_strategy_declaring_no_state(
        self, account, exchange, report_statuses, make_resting
    ):
        report_statuses({BTC: NEW_POSITIONS_CLOSED})
        exchange.get_open_orders.return_value = [
            make_resting(
                order_id="trigger-1", kind=OrderType.TRIGGER, side=OrderSide.BUY
            )
        ]

        run(Booking(account, lambda bookkeeper: None))

        assert cancelled_ids(exchange) == ["trigger-1"]

    def test_a_symbol_run_in_exit_only_mode_by_configuration_is_restricted_once(
        self, account, exchange, report_statuses, make_resting
    ):
        report_statuses({BTC: NEW_POSITIONS_CLOSED})
        exchange.get_open_orders.return_value = [
            make_resting(order_id="rung-1", side=OrderSide.BUY)
        ]

        run(booking(account, lambda bookkeeper: None), BTC)

        assert cancelled_ids(exchange) == ["rung-1"]

    def test_it_joins_the_symbols_run_in_exit_only_mode_by_configuration(
        self, account, exchange, report_statuses
    ):
        report_statuses({ETH: NEW_POSITIONS_CLOSED})
        exchange.get_open_positions.return_value = {BTC: long_position(BTC, 1.0)}

        run(
            booking(
                account, lambda bookkeeper: _entries_and_an_exit(account, bookkeeper)
            ),
            BTC,
        )

        assert market_orders(exchange) == [(BTC, OrderSide.SELL, 1.0)]


class TestClosedToEveryOrder:
    def test_nothing_is_sent_on_the_symbol_while_the_other_symbol_trades(
        self, account, exchange, report_statuses, make_protection
    ):
        report_statuses({BTC: EVERY_ORDER_CLOSED})
        exchange.get_open_positions.return_value = {BTC: long_position(BTC, 1.0)}
        exchange.get_open_orders.return_value = [make_protection("sl-1")]

        def book(bookkeeper: BookKeeper) -> None:
            _entries_and_an_exit(account, bookkeeper)
            bookkeeper.add(account.move_stop_loss(BTC, 95.0))
            bookkeeper.add(account.set_leverage(BTC, 3.0))

        run(booking(account, book))

        assert market_orders(exchange) == [(ETH, OrderSide.BUY, 2.0)]
        exchange.update_position_stop_loss.assert_not_awaited()
        exchange.set_leverage.assert_not_awaited()

    def test_an_action_naming_no_exchange_is_never_held_back(
        self, account, report_statuses, caplog
    ):
        report_statuses({BTC: EVERY_ORDER_CLOSED})

        run(
            booking(
                account,
                lambda bookkeeper: bookkeeper.add(BaseExchangeAction(symbol=BTC)),
            )
        )

        assert (
            "Internal error in BaseExchangeAction (BTC/USDT:USDT)"
            in messages_at(caplog, logging.ERROR)[0]
        )

    def test_a_batch_spanning_the_symbol_and_another_keeps_the_other_members(
        self, account, exchange, report_statuses
    ):
        report_statuses({BTC: EVERY_ORDER_CLOSED})

        def book(bookkeeper: BookKeeper) -> None:
            bookkeeper.add(
                account.place_orders(
                    [
                        account.long_entry(BTC, 1.0).limit(99.0),
                        account.long_entry(ETH, 2.0).limit(98.0),
                    ]
                )
            )

        run(booking(account, book))

        requests = exchange.place_orders.await_args.args[0]
        assert [request.symbol for request in requests] == [ETH]

    def test_a_modification_spanning_the_symbol_and_another_keeps_the_other_members(
        self, account, exchange, report_statuses, make_resting
    ):
        report_statuses({BTC: EVERY_ORDER_CLOSED})
        exchange.get_open_orders.return_value = [
            make_resting(order_id="rung-btc", symbol=BTC),
            make_resting(order_id="rung-eth", symbol=ETH),
        ]

        def book(bookkeeper: BookKeeper) -> None:
            bookkeeper.add(
                account.modify_orders(
                    [
                        OrderModification(
                            order_id="rung-btc",
                            order=account.long_entry(BTC, 1.0).limit(99.0),
                        ),
                        OrderModification(
                            order_id="rung-eth",
                            order=account.long_entry(ETH, 2.0).limit(98.0),
                        ),
                    ]
                )
            )

        run(booking(account, book))

        requests = exchange.modify_orders.await_args.args[0]
        assert [request.order_id for request in requests] == ["rung-eth"]

    def test_the_cancels_the_engine_books_itself_are_dropped(
        self, account, exchange, report_statuses, make_protection
    ):
        report_statuses({BTC: EVERY_ORDER_CLOSED})
        exchange.get_open_orders.return_value = [
            make_protection("sl-btc", symbol=BTC),
            make_protection("sl-eth", symbol=ETH),
        ]

        run(booking(account, lambda bookkeeper: None))

        assert cancelled_ids(exchange) == ["sl-eth"]

    def test_the_cancels_of_exit_only_mode_are_dropped(
        self, account, exchange, report_statuses, make_resting
    ):
        report_statuses({BTC: EVERY_ORDER_CLOSED})
        exchange.get_open_orders.return_value = [
            make_resting(order_id="rung-1", side=OrderSide.BUY)
        ]

        run(booking(account, lambda bookkeeper: None), BTC)

        exchange.cancel_order_by_id.assert_not_awaited()

    def test_no_margin_setting_is_sent_on_the_symbol(
        self, account, exchange, report_statuses
    ):
        report_statuses({BTC: EVERY_ORDER_CLOSED})
        held = MarginSettings(leverage=3.0, margin_mode=MarginMode.ISOLATED)
        target = MarginSettings(leverage=5.0, margin_mode=MarginMode.ISOLATED)
        exchange.get_margin_settings.return_value = {BTC: held, ETH: held}

        class Strategy(Booking):
            async def setup(self, requirements):
                for symbol in (BTC, ETH):
                    requirements.ohlcv.add(symbol, "1d")
                requirements.account.add(
                    account, margin_targets={BTC: target, ETH: target}
                )

        run(Strategy(account, lambda bookkeeper: None))

        exchange.set_leverage.assert_awaited_once_with(ETH, 5.0)

    def test_the_tracked_exposure_keeps_its_value(
        self, account, exchange, report_statuses
    ):
        report_statuses({BTC: EVERY_ORDER_CLOSED})
        alpha = TrackingId(f"{BTC}@1d#alpha")
        exchange.get_open_positions.return_value = {BTC: long_position(BTC, 1.0)}
        exchange.get_executions_since.return_value = [
            Execution(
                execution_id="entry-1",
                order_id="entry-1",
                symbol=BTC,
                side=OrderSide.BUY,
                price=100.0,
                quantity=1.0,
                timestamp=datetime(2024, 1, 1, tzinfo=UTC),
                client_order_id=client_order_id_carrying(profile_tag("1d", "alpha")),
            )
        ]

        trackers = []

        class Strategy(Booking):
            async def setup(self, requirements):
                requirements.ohlcv.add(BTC, "1d")
                requirements.account.add(account, symbols=[BTC])
                tracker = requirements.tracker.add(
                    account=account,
                    symbols={alpha: BTC},
                    tags={alpha: profile_tag("1d", "alpha")},
                )
                trackers.append(tracker)

        bot = livebot(
            Strategy(
                account,
                lambda bookkeeper: bookkeeper.add(
                    account.long_exit(BTC, 1.0, reduce_only=False)
                ),
            ),
        )

        asyncio.run(bot.run())

        exchange.place_market_order.assert_not_awaited()
        assert trackers[0].get(alpha) == TrackedPosition(PositionSide.LONG, 1.0)


class TestWarning:
    def test_one_line_names_each_halted_symbol_the_venue_and_the_status_word(
        self, account, exchange, report_statuses, caplog
    ):
        report_statuses({BTC: NEW_POSITIONS_CLOSED, ETH: EVERY_ORDER_CLOSED})

        run(booking(account, lambda bookkeeper: None))

        assert messages_at(caplog, logging.WARNING) == [
            "BTC/USDT:USDT on bitget (account bench-live) is "
            "closed_to_new_positions: the venue reports status limit_open",
            "ETH/USDT:USDT on bitget (account bench-live) is "
            "closed_to_every_order: the venue reports status maintain",
        ]

    async def test_a_dry_run_logs_a_halt_at_debug_and_no_warning(
        self, account, exchange, report_statuses, caplog
    ):
        report_statuses({BTC: EVERY_ORDER_CLOSED})
        strategy = booking(account, lambda bookkeeper: None)

        await dry_run_at_debug(caplog, strategy, make_provider())

        assert messages_at(caplog, logging.WARNING) == []
        assert messages_at(caplog, logging.DEBUG) == [
            "BTC/USDT:USDT on bitget (account bench-live) is "
            "closed_to_every_order: the venue reports status maintain"
        ]

    def test_an_account_built_without_a_venue_is_named_by_its_exchange_type(
        self, exchange, report_statuses, caplog
    ):
        report_statuses({BTC: EVERY_ORDER_CLOSED})

        run(booking(FuturesAccount(exchange, name=ACCOUNT), lambda bookkeeper: None))

        assert messages_at(caplog, logging.WARNING) == [
            f"BTC/USDT:USDT on {type(exchange).__name__} (account bench-live) is "
            "closed_to_every_order: the venue reports status maintain"
        ]

    def test_a_symbol_the_venue_does_not_list_is_not_logged_as_halted(
        self, account, exchange, report_statuses, caplog
    ):
        report_statuses({BTC: EVERY_ORDER_CLOSED}, unlisted=[ETH])

        run(
            booking(
                account, lambda bookkeeper: bookkeeper.add(account.long_entry(BTC, 1.0))
            )
        )

        assert market_orders(exchange) == []
        assert messages_at(caplog, logging.WARNING) == [
            "BTC/USDT:USDT on bitget (account bench-live) is closed_to_every_order: "
            "the venue reports status maintain"
        ]

    def test_an_open_symbol_logs_nothing(self, account, report_statuses, caplog):
        report_statuses({})

        run(booking(account, lambda bookkeeper: None))

        assert messages_at(caplog, logging.WARNING) == []


class TestTradingResumes:
    def test_a_listing_reporting_the_symbol_open_again_trades_it_as_usual(
        self, account, exchange, report_statuses
    ):
        bot = livebot(
            booking(
                account, lambda bookkeeper: bookkeeper.add(account.long_entry(BTC, 1.0))
            )
        )
        report_statuses({BTC: EVERY_ORDER_CLOSED})
        asyncio.run(bot.run())

        report_statuses({})
        asyncio.run(bot.run())

        assert market_orders(exchange) == [(BTC, OrderSide.BUY, 1.0)]


class TestTwoAccountsOnOneSymbol:
    def test_a_symbol_closed_to_every_order_on_one_account_trades_on_the_other(
        self, account, other_account, exchange, other_exchange, report_statuses
    ):
        report_statuses({BTC: EVERY_ORDER_CLOSED})

        def book(bookkeeper: BookKeeper) -> None:
            bookkeeper.add(account.long_entry(BTC, 1.0))
            bookkeeper.add(other_account.long_entry(BTC, 2.0))

        run(_TwoAccounts((other_account, account), book))

        assert market_orders(exchange) == []
        assert market_orders(other_exchange) == [(BTC, OrderSide.BUY, 2.0)]

    def test_a_symbol_closed_to_new_positions_on_one_account_is_judged_against_its_own_position(
        self, account, other_account, exchange, other_exchange, report_statuses
    ):
        report_statuses({BTC: NEW_POSITIONS_CLOSED})
        exchange.get_open_positions.return_value = {BTC: long_position(BTC, 1.0)}
        other_exchange.get_open_positions.return_value = {BTC: long_position(BTC, 5.0)}

        def book(bookkeeper: BookKeeper) -> None:
            for booking_account in (account, other_account):
                bookkeeper.add(booking_account.long_exit(BTC, 3.0, reduce_only=False))
                bookkeeper.add(booking_account.long_entry(BTC, 1.0))

        run(_TwoAccounts((other_account, account), book))

        assert market_orders(exchange) == [(BTC, OrderSide.SELL, 1.0)]
        assert market_orders(other_exchange) == [
            (BTC, OrderSide.SELL, 3.0),
            (BTC, OrderSide.BUY, 1.0),
        ]

    def test_only_the_halted_account_reads_the_state_exit_only_mode_needs(
        self, account, other_account, exchange, other_exchange, report_statuses
    ):
        report_statuses({BTC: NEW_POSITIONS_CLOSED})

        class Declaring(_TwoAccounts):
            async def setup(self, requirements):
                requirements.ohlcv.add(BTC, "1d")
                for declared in self._accounts:
                    requirements.account.add(declared, symbols=[BTC])

        run(Declaring((other_account, account), lambda bookkeeper: None))

        exchange.get_open_positions.assert_awaited_once()
        other_exchange.get_open_positions.assert_not_awaited()

    def test_a_margin_setting_is_sent_on_the_account_whose_listing_reports_the_symbol_open(
        self, account, other_account, exchange, other_exchange, report_statuses
    ):
        report_statuses({BTC: EVERY_ORDER_CLOSED})
        held = MarginSettings(leverage=3.0, margin_mode=MarginMode.ISOLATED)
        target = MarginSettings(leverage=5.0, margin_mode=MarginMode.ISOLATED)
        exchange.get_margin_settings.return_value = {BTC: held}
        other_exchange.get_margin_settings.return_value = {BTC: held}

        class Targeting(_TwoAccounts):
            async def setup(self, requirements):
                requirements.ohlcv.add(BTC, "1d")
                for declared in self._accounts:
                    requirements.account.add(declared, margin_targets={BTC: target})

        run(Targeting((other_account, account), lambda bookkeeper: None))

        exchange.set_leverage.assert_not_awaited()
        other_exchange.set_leverage.assert_awaited_once_with(BTC, 5.0)
