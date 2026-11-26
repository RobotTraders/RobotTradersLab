import logging
from collections.abc import Iterable
from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from robottraderslab import Symbol
from robottraderslab._internal import (
    listing_lacks,
    listing_lacks_among,
    report_unlisted_holdings,
    split_by_read,
)
from robottraderslab.exchanges import PositionSide
from robottraderslab.strategies import (
    AccountSnapshot,
    AccountSnapshots,
    OrderProtocol,
    StrategyRequirements,
)
from robottraderslab.strategies.futures import PositionSnapshot

ACCOUNT_NAME = "test"
BTC = Symbol.create("BTC/USDT:USDT")
ETH = Symbol.create("ETH/USDT:USDT")
DOGE = Symbol.create("DOGE/USDT:USDT")


class TestAPositionTheListingLacks:
    def test_the_position_is_named_with_its_side_and_its_quantity(self, caplog):
        fake_account = _Account()
        strategy_requirements = _declared(fake_account, positions=True)
        account_snapshots = _account_snapshots(
            positions={BTC: _position_snapshot(BTC, PositionSide.SHORT, 0.25)}
        )

        with caplog.at_level(logging.ERROR):
            report_unlisted_holdings(
                strategy_requirements.account, account_snapshots, logging.ERROR
            )

        assert _lines(caplog, logging.ERROR) == [
            "BTC/USDT:USDT: the venue lists no contract for it, yet the account "
            "holds short 0.25; its position is not settled and nothing is sent on it"
        ]

    @pytest.mark.parametrize(
        ("quantity", "named"),
        [
            (1234567.0, "1234567"),
            (123456.789, "123456.789"),
            (1500000.0, "1500000"),
            (0.00001234, "0.00001234"),
        ],
    )
    def test_the_quantity_is_named_exactly(self, caplog, quantity, named):
        fake_account = _Account()
        strategy_requirements = _declared(fake_account, positions=True)
        account_snapshots = _account_snapshots(
            positions={BTC: _position_snapshot(BTC, PositionSide.LONG, quantity)}
        )

        with caplog.at_level(logging.ERROR):
            report_unlisted_holdings(
                strategy_requirements.account, account_snapshots, logging.ERROR
            )

        assert f"holds long {named};" in caplog.text

    def test_a_position_on_a_symbol_the_declarations_leave_out_is_not_named(
        self, caplog
    ):
        fake_account = _Account()
        strategy_requirements = _declared(fake_account, positions=True)
        account_snapshots = _account_snapshots(
            positions={DOGE: _position_snapshot(DOGE, PositionSide.LONG, 1.0)}
        )

        with caplog.at_level(logging.DEBUG):
            report_unlisted_holdings(
                strategy_requirements.account, account_snapshots, logging.ERROR
            )

        assert caplog.records == []


class TestOrdersTheListingLacks:
    def test_the_orders_are_counted_under_the_symbol(self, caplog):
        fake_account = _Account()
        strategy_requirements = _declared(fake_account, open_orders=True)
        account_snapshots = _account_snapshots(
            open_orders=[_fake_order(BTC, "one"), _fake_order(BTC, "two")]
        )

        with caplog.at_level(logging.ERROR):
            report_unlisted_holdings(
                strategy_requirements.account, account_snapshots, logging.ERROR
            )

        assert _lines(caplog, logging.ERROR) == [
            "BTC/USDT:USDT: the venue lists no contract for it, yet 2 order(s) rest "
            "on it; nothing is sent on it"
        ]

    def test_the_orders_of_another_symbol_are_not_counted(self, caplog):
        fake_account = _Account()
        strategy_requirements = _declared(fake_account, open_orders=True)
        account_snapshots = _account_snapshots(
            open_orders=[_fake_order(BTC, "one"), _fake_order(DOGE, "two")]
        )

        with caplog.at_level(logging.ERROR):
            report_unlisted_holdings(
                strategy_requirements.account, account_snapshots, logging.ERROR
            )

        assert _lines(caplog, logging.ERROR) == [
            "BTC/USDT:USDT: the venue lists no contract for it, yet 1 order(s) rest "
            "on it; nothing is sent on it"
        ]

    def test_a_position_and_orders_on_one_symbol_make_two_lines(self, caplog):
        fake_account = _Account()
        strategy_requirements = _declared(
            fake_account, positions=True, open_orders=True
        )
        account_snapshots = _account_snapshots(
            positions={BTC: _position_snapshot(BTC, PositionSide.LONG, 1.0)},
            open_orders=[_fake_order(BTC, "one")],
        )

        with caplog.at_level(logging.ERROR):
            report_unlisted_holdings(
                strategy_requirements.account, account_snapshots, logging.ERROR
            )

        position_line, orders_line = _lines(caplog, logging.ERROR)
        assert "the account holds long 1;" in position_line
        assert "yet 1 order(s) rest on it" in orders_line


class TestWhatIsSaid:
    def test_a_symbol_holding_nothing_says_nothing(self, caplog):
        fake_account = _Account()
        strategy_requirements = _declared(
            fake_account, positions=True, open_orders=True
        )
        account_snapshots = _account_snapshots(positions={}, open_orders=[])

        with caplog.at_level(logging.DEBUG):
            report_unlisted_holdings(
                strategy_requirements.account, account_snapshots, logging.ERROR
            )

        assert caplog.records == []

    def test_an_account_reading_neither_positions_nor_orders_is_left_out(self, caplog):
        fake_account = _Account()
        strategy_requirements = _declared(fake_account, balances=True)

        with caplog.at_level(logging.DEBUG):
            report_unlisted_holdings(
                strategy_requirements.account, AccountSnapshots({}), logging.ERROR
            )

        assert caplog.records == []

    def test_the_level_is_the_callers(self, caplog):
        fake_account = _Account()
        strategy_requirements = _declared(fake_account, positions=True)
        account_snapshots = _account_snapshots(
            positions={BTC: _position_snapshot(BTC, PositionSide.LONG, 1.0)}
        )

        with caplog.at_level(logging.DEBUG):
            report_unlisted_holdings(
                strategy_requirements.account, account_snapshots, logging.DEBUG
            )

        assert _lines(caplog, logging.ERROR) == []
        assert len(_lines(caplog, logging.DEBUG)) == 1


class TestListingLacks:
    async def test_each_account_gets_the_declared_symbols_its_venue_does_not_list(
        self,
    ):
        lacking = _Account([BTC, DOGE])
        complete = _Account(name="complete")
        strategy_requirements = _declared(lacking)
        strategy_requirements.account.add(complete, symbols=[BTC, ETH])

        unlisted = await listing_lacks(strategy_requirements.account)

        assert unlisted == {lacking: frozenset({BTC})}


class TestListingLacksAmong:
    async def test_each_account_gets_the_given_symbols_it_declares_and_lacks(self):
        first = _Account([BTC])
        second = _Account([DOGE], name="second")
        strategy_requirements = StrategyRequirements()
        strategy_requirements.account.add(first, symbols=[BTC, ETH])
        strategy_requirements.account.add(second, symbols=[ETH, DOGE])

        found = await listing_lacks_among(strategy_requirements.account, {BTC, DOGE})

        assert found == {first: frozenset({BTC}), second: frozenset({DOGE})}

    async def test_a_symbol_the_accounts_own_listing_has_is_left_out(self):
        strategy_requirements = _declared(_Account())

        found = await listing_lacks_among(strategy_requirements.account, {BTC})

        assert found == {}

    async def test_symbols_nobody_declares_give_nothing(self):
        strategy_requirements = _declared(_Account([DOGE]))

        found = await listing_lacks_among(strategy_requirements.account, {DOGE})

        assert found == {}


class TestSplitByRead:
    def test_an_account_reading_its_positions_and_orders_is_read(self):
        reading = _Account()
        strategy_requirements = _declared(
            reading, positions=True, open_orders=True, triggers=True
        )

        read, unread = split_by_read(
            strategy_requirements.account, {reading: frozenset({BTC})}
        )

        assert read == {reading: frozenset({BTC})}
        assert unread == {}

    @pytest.mark.parametrize(
        "declared",
        [
            {"positions": True},
            {"open_orders": True},
            {"positions": True, "open_orders": True},
            {"balances": True},
        ],
    )
    def test_an_account_reading_less_is_not(self, declared):
        reading_less = _Account()
        strategy_requirements = _declared(reading_less, **declared)

        read, unread = split_by_read(
            strategy_requirements.account, {reading_less: frozenset({BTC})}
        )

        assert read == {}
        assert unread == {reading_less: frozenset({BTC})}


class _Account:
    def __init__(
        self, unlisted: Iterable[Symbol] = (), name: str = ACCOUNT_NAME
    ) -> None:
        self.name = name
        self._unlisted = frozenset(unlisted)

    async def _unlisted_among(self, symbols: Iterable[Symbol]) -> frozenset[Symbol]:
        return self._unlisted & frozenset(symbols)


def _declared(fake_account: _Account, **declared: bool) -> StrategyRequirements:
    strategy_requirements = StrategyRequirements()
    strategy_requirements.account.add(fake_account, symbols=[BTC, ETH], **declared)
    return strategy_requirements


def _position_snapshot(
    symbol: Symbol, side: PositionSide, quantity: float
) -> PositionSnapshot:
    return PositionSnapshot(
        symbol=symbol,
        side=side,
        quantity=quantity,
        average_entry_price=100.0,
        entry_time=datetime(2025, 1, 1, tzinfo=UTC),
        leverage=1.0,
        liquidation_price=0.0,
        stop_loss_price=None,
        take_profit_price=None,
    )


def _account_snapshots(
    *,
    positions: dict[Symbol, PositionSnapshot] | None = None,
    open_orders: list[Mock] | None = None,
) -> AccountSnapshots:
    return AccountSnapshots(
        {
            ACCOUNT_NAME: AccountSnapshot(
                account_name=ACCOUNT_NAME, positions=positions, open_orders=open_orders
            )
        }
    )


def _lines(caplog: pytest.LogCaptureFixture, level: int) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.levelno == level]


def _fake_order(symbol: Symbol, order_id: str) -> Mock:
    fake_order = Mock(spec=OrderProtocol)
    fake_order.symbol = symbol
    fake_order.order_id = order_id
    return fake_order
