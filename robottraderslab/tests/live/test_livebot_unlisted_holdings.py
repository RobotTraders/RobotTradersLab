import asyncio
import logging
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from unittest.mock import Mock

import pytest
from dry_run_helpers import dry_run_at_debug, sent
from live_cycle_helpers import (
    BTC,
    ETH,
    Booking,
    booking,
    cancelled_ids,
    long_position,
    make_provider,
    market_orders,
    messages_at,
    run,
)

from robottraderslab import Symbol, TimeFrame
from robottraderslab.exceptions import (
    ExchangeRecoverableError,
    SymbolNotListedError,
)
from robottraderslab.exchanges import (
    Closure,
    Instrument,
    OrderSide,
    OrderType,
    TradingStatus,
)
from robottraderslab.futures import FuturesAccount
from robottraderslab.live.livebot import LiveBot, utc_now
from robottraderslab.strategies import OHLCVs, StrategyProtocol, StrategyRequirements

EVERY_ORDER_CLOSED = Closure(TradingStatus.CLOSED_TO_EVERY_ORDER, "maintain")
ONE_O_CLOCK = datetime(2026, 1, 1, 1, 0, tzinfo=UTC)

pytestmark = pytest.mark.usefixtures("candle_boundary_clock")


class TestASymbolTheListingLacksAndTheAccountHolds:
    def test_the_position_is_named_each_cycle_and_nothing_is_sent_on_it(
        self, account, exchange, make_listing_lacking, caplog
    ):
        make_listing_lacking(BTC)
        exchange.get_open_positions.return_value = {BTC: long_position(BTC, 1.0)}

        cancelling = booking(
            account,
            lambda bookkeeper: bookkeeper.add(account.cancel_order(BTC, "rung-1")),
        )

        run(cancelling)
        run(cancelling)

        line = (
            "BTC/USDT:USDT: the venue lists no contract for it, yet the account "
            "holds long 1; its position is not settled and nothing is sent on it"
        )
        assert _holding_lines(caplog, logging.ERROR) == [line, line]
        assert sent(exchange) == []

    def test_a_protection_left_on_it_is_named_and_not_swept(
        self, account, exchange, make_listing_lacking, make_protection, caplog
    ):
        make_listing_lacking(BTC)
        exchange.get_open_orders.return_value = [make_protection("sl-1")]

        run(booking(account, lambda bookkeeper: None))

        assert _holding_lines(caplog, logging.ERROR) == [
            "BTC/USDT:USDT: the venue lists no contract for it, yet 1 order(s) rest "
            "on it; nothing is sent on it"
        ]
        assert cancelled_ids(exchange) == []

    def test_a_protection_left_on_a_listed_symbol_is_still_swept(
        self, account, exchange, make_listing_lacking, make_protection, caplog
    ):
        make_listing_lacking(BTC)
        exchange.get_open_orders.return_value = [
            make_protection("sl-btc", symbol=BTC),
            make_protection("sl-eth", symbol=ETH),
        ]

        run(booking(account, lambda bookkeeper: None))

        assert cancelled_ids(exchange) == ["sl-eth"]
        assert _holding_lines(caplog, logging.ERROR) == [
            "BTC/USDT:USDT: the venue lists no contract for it, yet 1 order(s) rest "
            "on it; nothing is sent on it"
        ]

    def test_a_position_on_a_listed_symbol_is_not_named(
        self, account, exchange, make_listing_lacking, caplog
    ):
        make_listing_lacking(BTC)
        exchange.get_open_positions.return_value = {
            BTC: long_position(BTC, 1.0),
            ETH: long_position(ETH, 3.0),
        }

        run(booking(account, lambda bookkeeper: None))

        assert _holding_lines(caplog, logging.ERROR) == [
            "BTC/USDT:USDT: the venue lists no contract for it, yet the account "
            "holds long 1; its position is not settled and nothing is sent on it"
        ]

    def test_the_cancels_of_exit_only_mode_are_dropped(
        self, account, exchange, make_listing_lacking, make_resting
    ):
        make_listing_lacking(BTC)
        exchange.get_open_orders.return_value = [
            make_resting(order_id="rung-1", side=OrderSide.BUY)
        ]

        run(booking(account, lambda bookkeeper: None), BTC)

        assert cancelled_ids(exchange) == []

    def test_the_symbols_closed_to_every_order_stay_closed(
        self, account, exchange, make_listing_lacking, make_protection
    ):
        make_listing_lacking(BTC, closures={ETH: EVERY_ORDER_CLOSED})
        exchange.get_open_orders.return_value = [
            make_protection("sl-btc", symbol=BTC),
            make_protection("sl-eth", symbol=ETH),
        ]

        run(
            booking(
                account,
                lambda bookkeeper: bookkeeper.add(account.long_entry(ETH, 2.0)),
            )
        )

        assert sent(exchange) == []

    async def test_a_dry_run_names_them_at_debug_and_logs_no_error(
        self, account, exchange, make_listing_lacking, caplog
    ):
        make_listing_lacking(BTC)
        exchange.get_open_positions.return_value = {BTC: long_position(BTC, 1.0)}

        await dry_run_at_debug(
            caplog, booking(account, lambda bookkeeper: None), make_provider()
        )

        assert _holding_lines(caplog, logging.ERROR) == []
        assert len(_holding_lines(caplog, logging.DEBUG)) == 1


class TestAnAccountDeclaringNeitherPositionsNorOrders:
    def test_the_position_and_the_orders_are_named_and_the_booked_cancel_is_not_sent(
        self, account, exchange, make_listing_lacking, make_resting, caplog
    ):
        make_listing_lacking(BTC)
        exchange.get_open_positions.return_value = {BTC: long_position(BTC, 1.0)}
        exchange.get_open_orders.return_value = [
            make_resting(order_id="rung-1"),
            make_resting(order_id="rung-2"),
        ]
        ohlcv_provider = make_provider({BTC: SymbolNotListedError("BTC is not listed")})
        cancelling = Booking(
            account,
            lambda bookkeeper: bookkeeper.add(account.cancel_order(BTC, "rung-1")),
            balances=True,
        )

        _run_cycle(cancelling, ohlcv_provider, utc_now)

        assert sent(exchange) == []
        assert _holding_lines(caplog, logging.ERROR) == [
            "BTC/USDT:USDT: the venue lists no contract for it, yet the account "
            "holds long 1; its position is not settled and nothing is sent on it",
            "BTC/USDT:USDT: the venue lists no contract for it, yet 2 order(s) rest "
            "on it; nothing is sent on it",
        ]
        exchange.get_open_positions.assert_awaited_once()
        exchange.get_open_orders.assert_awaited_once()

    def test_a_read_the_venue_refuses_names_the_account_and_the_cycle_goes_on(
        self, account, exchange, make_listing_lacking, caplog
    ):
        make_listing_lacking(BTC)
        exchange.get_open_positions.side_effect = ExchangeRecoverableError("refused")
        ohlcv_provider = make_provider({BTC: SymbolNotListedError("BTC is not listed")})

        def book(bookkeeper):
            bookkeeper.add(account.cancel_order(BTC, "rung-1"))
            bookkeeper.add(account.long_entry(ETH, 2.0))

        _run_cycle(Booking(account, book, balances=True), ohlcv_provider, utc_now)

        assert (
            f"{account.name}: the holdings on the symbols the listing lacks could not "
            "be read, nothing is reported this cycle: refused"
        ) in messages_at(caplog, logging.WARNING)
        assert "skipping cycle" not in caplog.text
        assert market_orders(exchange) == [(ETH, OrderSide.BUY, 2.0)]
        assert cancelled_ids(exchange) == []

    async def test_a_dry_run_returns_the_refused_read_as_a_recoverable_failure(
        self, account, exchange, make_listing_lacking, caplog
    ):
        make_listing_lacking(BTC)
        exchange.get_open_positions.side_effect = ExchangeRecoverableError("refused")
        ohlcv_provider = make_provider({BTC: SymbolNotListedError("BTC is not listed")})

        failures = await dry_run_at_debug(
            caplog,
            Booking(account, lambda bookkeeper: None, balances=True),
            ohlcv_provider,
        )

        refused_read = (
            f"{account.name}: the holdings on the symbols the listing lacks could not "
            "be read, nothing is reported this cycle: refused"
        )
        refused = [failure for failure in failures if failure.message == refused_read]
        assert [failure.recoverable for failure in refused] == [True]
        assert refused_read in messages_at(caplog, logging.DEBUG)
        assert refused_read not in messages_at(caplog, logging.WARNING)


class TestASymbolTheListingLacksAndTheAccountHoldsNothingOn:
    def test_the_orders_booked_on_it_are_not_sent(
        self, account, exchange, make_listing_lacking
    ):
        make_listing_lacking(BTC)

        run(
            booking(
                account,
                lambda bookkeeper: bookkeeper.add(account.cancel_order(BTC, "rung-1")),
            )
        )

        assert sent(exchange) == []


class TestTheReadsAnUnlistedSymbolAdds:
    def test_an_account_reading_all_three_is_named_and_not_read_twice(
        self, account, exchange, make_listing_lacking, caplog
    ):
        make_listing_lacking(BTC)
        exchange.get_open_positions.return_value = {
            BTC: long_position(BTC, 1.0),
            ETH: long_position(ETH, 3.0),
        }

        run(booking(account, lambda bookkeeper: None, triggers=True))

        exchange.get_open_positions.assert_awaited_once()
        exchange.get_open_orders.assert_awaited_once()
        assert _holding_lines(caplog, logging.ERROR) == [
            "BTC/USDT:USDT: the venue lists no contract for it, yet the account "
            "holds long 1; its position is not settled and nothing is sent on it"
        ]

    def test_an_account_not_reading_trigger_entries_is_read_again_for_them(
        self, account, exchange, make_listing_lacking, make_resting, caplog
    ):
        make_listing_lacking(BTC)

        async def get_open_orders(symbols, *, with_triggers=False):
            if with_triggers:
                return [
                    make_resting(
                        order_id="trigger-1", kind=OrderType.TRIGGER, side=OrderSide.BUY
                    )
                ]
            return []

        exchange.get_open_orders.side_effect = get_open_orders

        run(booking(account, lambda bookkeeper: None))

        assert "yet 1 order(s) rest on it" in _holding_lines(caplog, logging.ERROR)[0]

    def test_a_cycle_with_every_symbol_listed_adds_no_read(
        self, account, exchange, make_listing_lacking
    ):
        make_listing_lacking()

        run(Booking(account, lambda bookkeeper: None, balances=True))

        exchange.get_open_positions.assert_not_awaited()
        exchange.get_open_orders.assert_not_awaited()


class TestACycleThatEndsBeforeTheAccountIsRead:
    def test_no_pair_fetched_still_names_the_position_with_the_two_reads_alone(
        self, account, exchange, make_listing_lacking, caplog
    ):
        make_listing_lacking(BTC)
        exchange.get_open_positions.return_value = {BTC: long_position(BTC, 1.0)}
        ohlcv_provider = make_provider({BTC: SymbolNotListedError("BTC is not listed")})

        _run_cycle(_Declaring(account, {BTC: "1d"}), ohlcv_provider, utc_now)

        assert len(_holding_lines(caplog, logging.ERROR)) == 1
        exchange.get_open_positions.assert_awaited_once()
        exchange.get_open_orders.assert_awaited_once()
        assert sent(exchange) == []

    def test_no_timeframe_held_still_names_the_orders(
        self, account, exchange, make_listing_lacking, make_resting, caplog
    ):
        make_listing_lacking(BTC)
        exchange.get_open_orders.return_value = [make_resting(order_id="rung-1")]
        ohlcv_provider = make_provider({BTC: SymbolNotListedError("BTC is not listed")})

        _run_cycle(
            _Declaring(account, {BTC: "1h", ETH: "4h"}),
            ohlcv_provider,
            lambda: ONE_O_CLOCK,
        )

        assert len(_holding_lines(caplog, logging.ERROR)) == 1

    def test_a_failed_signal_generation_still_names_the_position(
        self, account, exchange, make_listing_lacking, caplog
    ):
        make_listing_lacking(BTC)
        exchange.get_open_positions.return_value = {BTC: long_position(BTC, 1.0)}
        ohlcv_provider = make_provider({BTC: SymbolNotListedError("BTC is not listed")})

        _run_cycle(
            _FailingSignals(account, {BTC: "1d", ETH: "1d"}), ohlcv_provider, utc_now
        )

        assert len(_holding_lines(caplog, logging.ERROR)) == 1
        exchange.get_open_positions.assert_awaited_once()
        exchange.get_open_orders.assert_awaited_once()
        assert sent(exchange) == []

    def test_only_the_positions_and_the_orders_of_the_declaring_account_are_read(
        self, account, exchange, make_live_venue, make_listing_lacking
    ):
        make_listing_lacking(BTC)
        bystander_exchange = make_live_venue()
        bystander = FuturesAccount(bystander_exchange, name="bystander")
        ohlcv_provider = make_provider({BTC: SymbolNotListedError("BTC is not listed")})

        _run_cycle(
            _DeclaringBeside(
                account,
                bystander,
                {BTC: "1d"},
                balances=True,
                executions=True,
                margin_settings=True,
            ),
            ohlcv_provider,
            utc_now,
        )

        exchange.get_open_positions.assert_awaited_once()
        exchange.get_open_orders.assert_awaited_once()
        exchange.get_balances.assert_not_awaited()
        exchange.get_executions_since.assert_not_awaited()
        exchange.get_margin_settings.assert_not_awaited()
        bystander_exchange.get_open_positions.assert_not_awaited()
        bystander_exchange.get_open_orders.assert_not_awaited()

    def test_the_moment_of_the_cycle_is_read_once(
        self, account, exchange, make_listing_lacking, caplog
    ):
        make_listing_lacking(BTC)
        exchange.get_open_positions.return_value = {BTC: long_position(BTC, 1.0)}
        ohlcv_provider = make_provider({BTC: SymbolNotListedError("BTC is not listed")})
        moments = []

        def moving_clock() -> datetime:
            moments.append(ONE_O_CLOCK + timedelta(seconds=len(moments)))
            return moments[-1]

        _run_cycle(_Declaring(account, {BTC: "1h"}), ohlcv_provider, moving_clock)

        assert len(_holding_lines(caplog, logging.ERROR)) == 1
        assert len(moments) == 1

    def test_a_symbol_only_the_candle_source_refuses_is_not_named(
        self, account, exchange, make_listing_lacking, caplog
    ):
        make_listing_lacking()
        exchange.get_open_positions.return_value = {BTC: long_position(BTC, 1.0)}
        ohlcv_provider = make_provider({BTC: SymbolNotListedError("BTC is not listed")})

        _run_cycle(_Declaring(account, {BTC: "1d"}), ohlcv_provider, utc_now)

        assert _holding_lines(caplog, logging.ERROR) == []
        exchange.get_open_positions.assert_not_awaited()

    def test_a_fetch_failing_for_another_reason_reads_nothing(
        self, account, exchange, make_listing_lacking, caplog
    ):
        make_listing_lacking(BTC)
        ohlcv_provider = make_provider(
            {BTC: ExchangeRecoverableError("the venue is down")}
        )

        _run_cycle(_Declaring(account, {BTC: "1d"}), ohlcv_provider, utc_now)

        exchange.get_open_positions.assert_not_awaited()
        exchange.get_open_orders.assert_not_awaited()
        assert _holding_lines(caplog, logging.ERROR) == []

    def test_a_symbol_the_account_holds_nothing_on_is_named_no_more(
        self, account, exchange, make_listing_lacking, caplog
    ):
        make_listing_lacking(BTC)
        ohlcv_provider = make_provider({BTC: SymbolNotListedError("BTC is not listed")})

        _run_cycle(_Declaring(account, {BTC: "1d"}), ohlcv_provider, utc_now)

        exchange.get_open_positions.assert_awaited_once()
        exchange.get_open_orders.assert_awaited_once()
        assert _holding_lines(caplog, logging.ERROR) == []

    async def test_a_dry_run_names_the_position_at_debug_and_logs_no_error(
        self, account, exchange, make_listing_lacking, caplog
    ):
        make_listing_lacking(BTC)
        exchange.get_open_positions.return_value = {BTC: long_position(BTC, 1.0)}
        ohlcv_provider = make_provider({BTC: SymbolNotListedError("BTC is not listed")})

        await dry_run_at_debug(caplog, _Declaring(account, {BTC: "1d"}), ohlcv_provider)

        assert _holding_lines(caplog, logging.ERROR) == []
        assert len(_holding_lines(caplog, logging.DEBUG)) == 1

    def test_a_read_the_venue_refuses_is_a_warning_and_no_error(
        self, account, exchange, make_listing_lacking, caplog
    ):
        make_listing_lacking(BTC)
        exchange.get_open_positions.side_effect = ExchangeRecoverableError("refused")
        ohlcv_provider = make_provider({BTC: SymbolNotListedError("BTC is not listed")})

        _run_cycle(_Declaring(account, {BTC: "1d"}), ohlcv_provider, utc_now)

        assert "Account state fetch failed, skipping cycle: refused" in messages_at(
            caplog, logging.WARNING
        )
        assert _holding_lines(caplog, logging.ERROR) == []


@pytest.fixture
def make_listing_lacking(exchange: Mock) -> Callable[..., None]:
    def make_listing_lacking(
        *unlisted: Symbol, closures: Mapping[Symbol, Closure] = MappingProxyType({})
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

        exchange.get_instrument.side_effect = get_instrument

    return make_listing_lacking


def _holding_lines(caplog: pytest.LogCaptureFixture, level: int) -> list[str]:
    return [
        message
        for message in messages_at(caplog, level)
        if "the venue lists no contract" in message
    ]


def _run_cycle(
    strategy: StrategyProtocol, ohlcv_provider: Mock, clock: Callable[[], datetime]
) -> None:
    asyncio.run(LiveBot(strategy, ohlcv_provider, clock=clock).run())


class _Declaring(Booking):
    def __init__(
        self,
        account: FuturesAccount,
        timeframes: Mapping[Symbol, TimeFrame],
        **declared: bool,
    ) -> None:
        super().__init__(
            account,
            lambda bookkeeper: None,
            **{"positions": True, "open_orders": True, **declared},
        )
        self._timeframes = timeframes

    async def setup(self, requirements: StrategyRequirements) -> None:
        for symbol, timeframe in self._timeframes.items():
            requirements.ohlcv.add(symbol, timeframe)
        requirements.account.add(
            self._account, symbols=list(self._timeframes), **self._declared
        )


class _DeclaringBeside(_Declaring):
    def __init__(
        self,
        account: FuturesAccount,
        bystander: FuturesAccount,
        timeframes: Mapping[Symbol, TimeFrame],
        **declared: bool,
    ) -> None:
        super().__init__(account, timeframes, **declared)
        self._bystander = bystander

    async def setup(self, requirements: StrategyRequirements) -> None:
        await super().setup(requirements)
        requirements.account.add(
            self._bystander, symbols=[ETH], positions=True, open_orders=True
        )


class _FailingSignals(_Declaring):
    def generate_trading_signals(self, ohlcvs: OHLCVs) -> None:
        raise KeyError("the candles of a dropped symbol are missing")
