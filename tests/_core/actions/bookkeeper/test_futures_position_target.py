import pytest

from robottraderslab import Symbol
from robottraderslab.exchanges import (
    Balance,
    OrderSide,
    PositionSide,
    PositionSnapshot,
)
from robottraderslab.futures.futures_limit_order import FuturesLimitOrderAction
from robottraderslab.strategies import AccountSnapshot
from robottraderslab.strategies.futures import (
    AvailableBalanceRatio,
    EquityRatio,
    TotalBalanceRatio,
)

BTC_USDT = Symbol.create("BTC/USDT:USDT")


def _holding(
    position: PositionSnapshot | None,
    total: float = 10_000.0,
    locked: float = 0.0,
    equity: float | None = None,
) -> AccountSnapshot:
    return AccountSnapshot(
        account_name="test",
        balances={"USDT": Balance(locked=locked, total=total)},
        positions={} if position is None else {BTC_USDT: position},
        equities=None if equity is None else {"USDT": equity},
        conversion_rates={BTC_USDT: 1.0},
    )


class TestPositionTarget:
    def test_opens_a_long_from_flat(self, account):
        order = account.long_target(BTC_USDT, _holding(None)).size(
            TotalBalanceRatio(0.3), 100.0
        )

        assert order.build().side == OrderSide.BUY
        assert order.build().quantity == 30.0
        assert order.build().reduce_only is False

    def test_grows_a_long_towards_the_target(self, account, make_position):
        holding = _holding(make_position(PositionSide.LONG, 10.0))

        order = account.long_target(BTC_USDT, holding).size(
            TotalBalanceRatio(0.3), 100.0
        )

        assert order.build().side == OrderSide.BUY
        assert order.build().quantity == 20.0
        assert order.build().reduce_only is False

    def test_shrinks_a_long_with_a_reduce_only_sell(self, account, make_position):
        holding = _holding(make_position(PositionSide.LONG, 30.0))

        order = account.long_target(BTC_USDT, holding).size(
            TotalBalanceRatio(0.1), 100.0
        )

        assert order.build().side == OrderSide.SELL
        assert order.build().quantity == 20.0
        assert order.build().reduce_only is True

    def test_quantity_of_zero_flattens_a_short_with_a_reduce_only_buy(
        self, account, make_position
    ):
        holding = _holding(make_position(PositionSide.SHORT, 30.0))

        order = account.short_target(BTC_USDT, holding).quantity(0.0, 100.0)

        assert order.build().side == OrderSide.BUY
        assert order.build().quantity == 30.0
        assert order.build().reduce_only is True

    def test_short_target_from_a_long_crosses_in_one_order(
        self, account, make_position
    ):
        holding = _holding(make_position(PositionSide.LONG, 10.0))

        order = account.short_target(BTC_USDT, holding).size(
            TotalBalanceRatio(0.1), 100.0
        )

        assert order.build().side == OrderSide.SELL
        assert order.build().quantity == 20.0
        assert order.build().reduce_only is False

    def test_smaller_short_target_from_a_long_crosses_in_one_order(
        self, account, make_position
    ):
        holding = _holding(make_position(PositionSide.LONG, 30.0))

        order = account.short_target(BTC_USDT, holding).size(
            TotalBalanceRatio(0.05), 100.0
        )

        assert order.build().side == OrderSide.SELL
        assert order.build().quantity == 35.0
        assert order.build().reduce_only is False

    def test_equity_target_is_capped_by_the_available_balance(self, account):
        holding = _holding(None, total=10_000.0, locked=8_000.0, equity=12_000.0)

        order = account.long_target(BTC_USDT, holding).size(EquityRatio(0.5), 100.0)

        assert order.build().quantity == 20.0

    def test_available_balance_target_leaves_the_locked_share_out(self, account):
        holding = _holding(None, total=10_000.0, locked=2_000.0)

        order = account.long_target(BTC_USDT, holding).size(
            AvailableBalanceRatio(0.5), 100.0
        )

        assert order.build().quantity == 40.0

    def test_quantity_target_shrinks_a_long_to_it(self, account, make_position):
        holding = _holding(make_position(PositionSide.LONG, 30.0))

        order = account.long_target(BTC_USDT, holding).quantity(25.0, 100.0)

        assert order.build().side == OrderSide.SELL
        assert order.build().quantity == 5.0
        assert order.build().reduce_only is True

    def test_target_keeps_the_builder_chain(self, account):
        order = account.long_target(BTC_USDT, _holding(None)).size(
            TotalBalanceRatio(0.3), 100.0
        )

        limit_order = order.limit(price=99.0).reason("rebalance").build()

        assert isinstance(limit_order, FuturesLimitOrderAction)
        assert limit_order.reason == "rebalance"

    def test_position_already_at_the_target(self, account, make_position):
        holding = _holding(make_position(PositionSide.LONG, 30.0))

        order = account.long_target(BTC_USDT, holding).size(
            TotalBalanceRatio(0.3), 100.0
        )

        assert order is None

    def test_difference_under_the_minimum_order(self, account, make_position):
        holding = _holding(make_position(PositionSide.LONG, 30.0))

        left_alone = account.long_target(BTC_USDT, holding).size(
            TotalBalanceRatio(0.3005), 100.0
        )
        placed = account.long_target(BTC_USDT, holding, minimum_order_ratio=0.0).size(
            TotalBalanceRatio(0.3005), 100.0
        )

        assert left_alone is None
        assert placed.build().quantity == pytest.approx(0.05)

    def test_target_sets_the_placement_reserve_aside(self, reserving_account):
        order = reserving_account.long_target(BTC_USDT, _holding(None)).size(
            TotalBalanceRatio(0.5), 100.0
        )

        assert order.build().quantity == 40.0

    def test_minimum_order_takes_no_reserve(self, reserving_account, make_position):
        holding = _holding(make_position(PositionSide.LONG, 40.0))

        left_alone = reserving_account.long_target(BTC_USDT, holding).size(
            TotalBalanceRatio(0.501125), 100.0
        )

        assert left_alone is None

    def test_negative_quantity(self, account):
        with pytest.raises(ValueError, match="`quantity` must be zero or above"):
            account.long_target(BTC_USDT, _holding(None)).quantity(-1.0, 100.0)

    def test_symbol_without_a_margin_currency(self, account):
        with pytest.raises(ValueError, match="A sizing rule sizes from the margin"):
            account.long_target(Symbol.create("BTC/USDT"), _holding(None)).size(
                TotalBalanceRatio(0.3), 100.0
            )

    def test_minimum_order_ratio_below_zero(self, account):
        with pytest.raises(ValueError, match="`minimum_order_ratio` must be between"):
            account.long_target(BTC_USDT, _holding(None), minimum_order_ratio=-0.1)

    def test_shrinks_a_short_with_a_reduce_only_buy(self, account, make_position):
        holding = _holding(make_position(PositionSide.SHORT, 30.0))

        order = account.short_target(BTC_USDT, holding).size(
            TotalBalanceRatio(0.1), 100.0
        )

        assert order.build().side == OrderSide.BUY
        assert order.build().quantity == 20.0
        assert order.build().reduce_only is True

    @pytest.mark.parametrize(
        "rule", [AvailableBalanceRatio, EquityRatio, TotalBalanceRatio]
    )
    def test_price_not_above_zero_on_every_sizing(self, account, rule):
        target = account.long_target(BTC_USDT, _holding(None, equity=10_000.0))

        with pytest.raises(ValueError, match="`price` must be greater than 0"):
            target.size(rule(0.3), 0.0)

    def test_quantity_target_price_not_above_zero(self, account):
        with pytest.raises(ValueError, match="`current_price` must be greater than 0"):
            account.long_target(BTC_USDT, _holding(None)).quantity(25.0, 0.0)

    def test_nan_quantity(self, account):
        with pytest.raises(ValueError, match="`quantity` must be zero or above"):
            account.long_target(BTC_USDT, _holding(None)).quantity(float("nan"), 100.0)
