import pytest

from robottraderslab import Symbol
from robottraderslab.exceptions import StrategyCriticalError
from robottraderslab.exchanges import (
    Balance,
    Currency,
    OrderSide,
    StopLoss,
    TakeProfit,
    TimeInForce,
)
from robottraderslab.futures.futures_close_position import ClosePositionAction
from robottraderslab.futures.futures_limit_order import FuturesLimitOrderAction
from robottraderslab.futures.futures_market_order import FuturesMarketOrderAction
from robottraderslab.strategies import (
    AccountSnapshot,
    BookKeeper,
    tag_of,
)
from robottraderslab.strategies.futures import (
    AvailableBalanceRatio,
    EquityRatio,
    RiskRatio,
    TotalBalanceRatio,
)

BTC_USDT = Symbol.create("BTC/USDT:USDT")
AUD_CAD = Symbol.create("AUD/CAD:USD")


ANY_SNAPSHOT = AccountSnapshot(
    account_name="test",
    balances={"USDT": Balance(locked=0.0, total=10_000.0)},
    conversion_rates={BTC_USDT: 1.0, AUD_CAD: 1.0},
)


def _snapshot(
    balances: dict[Currency, Balance],
    equities: dict[Currency, float] | None = None,
    *,
    rate: float = 1.0,
) -> AccountSnapshot:
    return AccountSnapshot(
        account_name="test",
        balances=balances,
        equities=equities,
        conversion_rates={BTC_USDT: rate, AUD_CAD: rate},
    )


class TestLongPositions:
    """Test long position entry and exit functionality."""

    def test_long_entry_market(self, account):
        market_order = account.long_entry(BTC_USDT, 1.0).build()

        assert isinstance(market_order, FuturesMarketOrderAction)
        assert market_order.quantity == 1.0
        assert market_order.exchange is account._exchange
        assert market_order.side == OrderSide.BUY
        assert market_order.stop_loss is None
        assert market_order.symbol == BTC_USDT
        assert market_order.take_profit is None
        assert market_order.trigger_price is None

    def test_long_exit_market(self, account):
        market_order = account.long_exit(BTC_USDT, 1.0).build()

        assert isinstance(market_order, FuturesMarketOrderAction)
        assert market_order.quantity == 1.0
        assert market_order.exchange is account._exchange
        assert market_order.side == OrderSide.SELL
        assert market_order.stop_loss is None
        assert market_order.symbol == BTC_USDT
        assert market_order.take_profit is None
        assert market_order.trigger_price is None

    def test_long_entry_limit(self, account):
        limit_order = account.long_entry(BTC_USDT, 1.0).limit(100.0).build()

        assert isinstance(limit_order, FuturesLimitOrderAction)
        assert limit_order.quantity == 1.0
        assert limit_order.exchange is account._exchange
        assert limit_order.side == OrderSide.BUY
        assert limit_order.stop_loss is None
        assert limit_order.symbol == BTC_USDT
        assert limit_order.take_profit is None
        assert limit_order.trigger_price is None

    def test_long_exit_limit(self, account):
        limit_order = account.long_exit(BTC_USDT, 1.0).limit(100.0).build()

        assert isinstance(limit_order, FuturesLimitOrderAction)
        assert limit_order.quantity == 1.0
        assert limit_order.exchange is account._exchange
        assert limit_order.side == OrderSide.SELL
        assert limit_order.stop_loss is None
        assert limit_order.symbol == BTC_USDT
        assert limit_order.take_profit is None
        assert limit_order.trigger_price is None


class TestShortPositions:
    """Test short position entry and exit functionality."""

    def test_short_entry_market(self, account):
        market_order = account.short_entry(BTC_USDT, 1.0).build()

        assert isinstance(market_order, FuturesMarketOrderAction)
        assert market_order.quantity == 1.0
        assert market_order.exchange is account._exchange
        assert market_order.side == OrderSide.SELL
        assert market_order.stop_loss is None
        assert market_order.symbol == BTC_USDT
        assert market_order.take_profit is None
        assert market_order.trigger_price is None

    def test_short_exit_market(self, account):
        market_order = account.short_exit(BTC_USDT, 1.0).build()

        assert isinstance(market_order, FuturesMarketOrderAction)
        assert market_order.quantity == 1.0
        assert market_order.exchange is account._exchange
        assert market_order.side == OrderSide.BUY
        assert market_order.stop_loss is None
        assert market_order.symbol == BTC_USDT
        assert market_order.take_profit is None
        assert market_order.trigger_price is None

    def test_short_entry_limit(self, account):
        limit_order = account.short_entry(BTC_USDT, 1.0).limit(100.0).build()

        assert isinstance(limit_order, FuturesLimitOrderAction)
        assert limit_order.quantity == 1.0
        assert limit_order.exchange is account._exchange
        assert limit_order.side == OrderSide.SELL
        assert limit_order.stop_loss is None
        assert limit_order.symbol == BTC_USDT
        assert limit_order.take_profit is None
        assert limit_order.trigger_price is None

    def test_short_exit_limit(self, account):
        limit_order = account.short_exit(BTC_USDT, 1.0).limit(100.0).build()

        assert isinstance(limit_order, FuturesLimitOrderAction)
        assert limit_order.quantity == 1.0
        assert limit_order.exchange is account._exchange
        assert limit_order.side == OrderSide.BUY
        assert limit_order.stop_loss is None
        assert limit_order.symbol == BTC_USDT
        assert limit_order.take_profit is None
        assert limit_order.trigger_price is None


class TestPositionClose:
    def test_closes_the_whole_position_at_market_by_default(self, account):
        close = account.close_position(BTC_USDT).build()

        assert isinstance(close, ClosePositionAction)
        assert close.symbol == BTC_USDT
        assert close.closing_ratio == 1.0
        assert close.limit_price is None

    def test_limit_rests_the_close_at_its_price_and_time_in_force(self, account):
        close = (
            account.close_position(BTC_USDT)
            .limit(120.0, time_in_force=TimeInForce.IOC)
            .build()
        )

        assert close.limit_price == 120.0
        assert close.time_in_force == TimeInForce.IOC

    def test_a_closing_ratio_travels_onto_the_close(self, account):
        close = account.close_position(BTC_USDT, closing_ratio=0.25).build()

        assert close.closing_ratio == 0.25

    def test_reason_travels_onto_the_close(self, account):
        close = account.close_position(BTC_USDT).reason("time exit").build()

        assert close.reason == "time exit"

    def test_tag_travels_onto_the_close_client_order_id(self, account):
        close = account.close_position(BTC_USDT).tag("a").build()

        assert tag_of(close.client_order_id) == "a"

    def test_a_blank_tag_on_a_close(self, account):
        with pytest.raises(ValueError, match="Tag cannot be empty"):
            account.close_position(BTC_USDT).tag(" ")

    def test_extra_fields_accumulate_on_the_close(self, account):
        close = (
            account.close_position(BTC_USDT)
            .extra_fields(signal=1)
            .extra_fields(regime="trend")
            .build()
        )

        assert close.extra_fields == {"signal": 1, "regime": "trend"}

    def test_when_filled_registers_the_close_callback(self, account):
        async def on_filled(placed_order, fill):
            pass

        close = account.close_position(BTC_USDT).when_filled(on_filled).build()

        assert close.on_filled is on_filled

    def test_a_close_is_booked_through_the_bookkeeper(self, account):
        bookkeeper = BookKeeper()

        booked = bookkeeper.add(account.close_position(BTC_USDT))

        assert isinstance(booked, ClosePositionAction)


class TestOrderValidation:
    """Test validation of order parameters."""

    def test_amount_is_not_specified(self, account):
        with pytest.raises(StrategyCriticalError, match="has no size"):
            account.long_entry(BTC_USDT).build()

    def test_long_entry_validates_amount_gt_0(self, account):
        with pytest.raises(ValueError, match="Quantity must be greater than 0"):
            account.long_entry(BTC_USDT, 0.0)

    def test_long_exit_validates_amount_gt_0(self, account):
        with pytest.raises(ValueError, match="Quantity must be greater than 0"):
            account.long_exit(BTC_USDT, 0.0)

    def test_limit_order_validates_limit_price_gt_0(self, account):
        with pytest.raises(ValueError, match="`price` must be greater than 0"):
            account.long_exit(BTC_USDT, 1.0).limit(0.0)

    def test_long_entry_rejects_nan_quantity(self, account):
        with pytest.raises(ValueError, match="Quantity must be greater than 0"):
            account.long_entry(BTC_USDT, float("nan"))

    def test_limit_order_rejects_nan_limit_price(self, account):
        with pytest.raises(ValueError, match="`price` must be greater than 0"):
            account.long_exit(BTC_USDT, 1.0).limit(float("nan"))


class TestQuoteConversion:
    def test_a_balance_share_with_a_cross_quoted_symbol(self, account):
        snapshot = _snapshot({"USD": Balance(locked=0.0, total=10_000.0)}, rate=0.8)

        order = (
            account.long_entry(AUD_CAD)
            .size(AvailableBalanceRatio(0.1), 0.9, snapshot)
            .build()
        )

        # 1000 USD buys 1000 / (0.9 CAD * 0.8 USD per CAD) AUD
        assert order.quantity == pytest.approx(1000.0 / 0.72)

    def test_a_risk_share_with_a_cross_quoted_symbol(self, account):
        snapshot = _snapshot({"USD": Balance(locked=0.0, total=10_000.0)}, rate=0.8)

        order = (
            account.long_entry(AUD_CAD)
            .size(RiskRatio(0.01, of="total_balance"), 0.9, snapshot)
            .stop_loss(price=0.88)
            .build()
        )

        # risking 100 USD over a 0.02 CAD stop distance worth 0.016 USD
        assert order.quantity == pytest.approx(100.0 / 0.016)


class TestOrderModification:
    """Test order modification functionality."""

    def test_last_limit_wins(self, account):
        limit_order = account.long_exit(BTC_USDT, 1.0).limit(50.0).limit(100.0).build()

        assert limit_order.price == 100.0


class TestBuildQuantityGuard:
    def test_build_with_zero_balance_creates_zero_quantity_order(self, account):
        snapshot = _snapshot({"USDT": Balance(locked=10_000.0, total=10_000.0)})

        action = (
            account.long_entry(BTC_USDT)
            .size(AvailableBalanceRatio(0.1), 1.0, snapshot)
            .build()
        )

        assert action.quantity == 0.0


class TestStopLoss:
    """Test stop loss functionality."""

    def test_market_order_with_stop_loss(self, account):
        market_order = account.long_entry(BTC_USDT, 1.0).stop_loss(4.0).build()

        assert market_order.stop_loss == StopLoss(trigger_price=4.0)

    def test_limit_order_with_stop_loss(self, account):
        limit_order = (
            account.long_entry(BTC_USDT, 1.0).limit(2.0).stop_loss(4.0).build()
        )

        assert limit_order.stop_loss == StopLoss(trigger_price=4.0)

    def test_stop_loss_validates_trigger_price_gt_0(self, account):
        with pytest.raises(ValueError, match="`trigger_price` must be greater than 0"):
            account.long_entry(BTC_USDT, 1.0).stop_loss(0.0)

    def test_stop_loss_rejects_nan_trigger_price(self, account):
        with pytest.raises(ValueError, match="`trigger_price` must be greater than 0"):
            account.long_entry(BTC_USDT, 1.0).stop_loss(float("nan"))

    def test_last_stop_loss_wins(self, account):
        market_order = (
            account.long_entry(BTC_USDT, 1.0).stop_loss(2.0).stop_loss(4.0).build()
        )

        assert market_order.stop_loss == StopLoss(trigger_price=4.0)

    def test_reason_travels_onto_stop_loss(self, account):
        market_order = (
            account.long_entry(BTC_USDT, 1.0)
            .stop_loss(4.0, reason="tight stop")
            .build()
        )

        assert market_order.stop_loss == StopLoss(
            trigger_price=4.0, reason="tight stop"
        )

    def test_reason_is_keyword_only(self, account):
        with pytest.raises(TypeError):
            account.long_entry(BTC_USDT, 1.0).stop_loss(4.0, "tight stop")  # type: ignore[misc]


class TestTakeProfit:
    """Test take profit functionality."""

    def test_market_order_with_take_profit(self, account):
        market_order = account.long_entry(BTC_USDT, 1.0).take_profit(8.0).build()

        assert market_order.take_profit == TakeProfit(trigger_price=8.0)

    def test_limit_order_with_take_profit(self, account):
        limit_order = (
            account.long_entry(BTC_USDT, 1.0).limit(2.0).take_profit(8.0).build()
        )

        assert limit_order.take_profit == TakeProfit(trigger_price=8.0)

    def test_take_profit_validates_trigger_price_gt_0(self, account):
        with pytest.raises(ValueError, match="`trigger_price` must be greater than 0"):
            account.long_entry(BTC_USDT, 1.0).take_profit(0.0)

    def test_take_profit_rejects_nan_trigger_price(self, account):
        with pytest.raises(ValueError, match="`trigger_price` must be greater than 0"):
            account.long_entry(BTC_USDT, 1.0).take_profit(float("nan"))

    def test_last_take_profit_wins(self, account):
        market_order = (
            account.long_entry(BTC_USDT, 1.0).take_profit(4.0).take_profit(8.0).build()
        )

        assert market_order.take_profit == TakeProfit(trigger_price=8.0)

    def test_reason_travels_onto_take_profit(self, account):
        market_order = (
            account.long_entry(BTC_USDT, 1.0)
            .take_profit(8.0, reason="target 50pct")
            .build()
        )

        assert market_order.take_profit == TakeProfit(
            trigger_price=8.0, reason="target 50pct"
        )

    def test_reason_is_keyword_only(self, account):
        with pytest.raises(TypeError):
            account.long_entry(BTC_USDT, 1.0).take_profit(8.0, "target 50pct")  # type: ignore[misc]


class TestTrigger:
    def test_market_order_with_trigger(self, account):
        market_order = account.long_entry(BTC_USDT, 1.0).trigger(8.0).build()

        assert market_order.trigger_price == 8.0

    def test_limit_order_with_trigger(self, account):
        limit_order = account.long_entry(BTC_USDT, 1.0).limit(2.0).trigger(8.0).build()

        assert limit_order.trigger_price == 8.0

    def test_last_trigger_wins(self, account):
        market_order = (
            account.long_entry(BTC_USDT, 1.0).trigger(4.0).trigger(8.0).build()
        )

        assert market_order.trigger_price == 8.0

    def test_rejects_nan_price(self, account):
        with pytest.raises(ValueError, match="`price` must be greater than 0"):
            account.long_entry(BTC_USDT, 1.0).trigger(float("nan"))

    def test_rejects_zero_price(self, account):
        with pytest.raises(ValueError, match="`price` must be greater than 0"):
            account.long_entry(BTC_USDT, 1.0).trigger(0.0)


class TestTimeInForce:
    def test_limit_order_rests_until_cancelled_by_default(self, account):
        limit_order = account.long_entry(BTC_USDT, 1.0).limit(100.0).build()

        assert limit_order.time_in_force is TimeInForce.GTC

    @pytest.mark.parametrize("time_in_force", [TimeInForce.IOC, TimeInForce.POST_ONLY])
    def test_limit_order_carries_its_time_in_force(self, account, time_in_force):
        limit_order = (
            account.long_entry(BTC_USDT, 1.0)
            .limit(100.0, time_in_force=time_in_force)
            .build()
        )

        assert limit_order.time_in_force is time_in_force

    def test_triggered_order_cannot_wait(self, account):
        with pytest.raises(StrategyCriticalError, match="on a triggered order"):
            (
                account.long_entry(BTC_USDT, 1.0)
                .limit(100.0, time_in_force=TimeInForce.POST_ONLY)
                .trigger(price=90.0)
                .build()
            )


class TestExtraFields:
    def test_extra_fields_adds_single_field_to_market_order(self, account):
        market_order = (
            account.long_entry(BTC_USDT, 1.0).extra_fields(spread=0.05).build()
        )

        assert market_order.extra_fields == {"spread": 0.05}

    def test_extra_fields_adds_single_field_to_limit_order(self, account):
        limit_order = (
            account.long_entry(BTC_USDT, 1.0)
            .limit(100.0)
            .extra_fields(rsi=65.2)
            .build()
        )

        assert limit_order.extra_fields == {"rsi": 65.2}

    def test_extra_fields_adds_multiple_fields_at_once(self, account):
        market_order = (
            account.long_entry(BTC_USDT, 1.0)
            .extra_fields(spread=0.05, rsi=65.2, volume=1500)
            .build()
        )

        assert market_order.extra_fields == {
            "spread": 0.05,
            "rsi": 65.2,
            "volume": 1500,
        }

    def test_extra_fields_unpacks_dictionary(self, account):
        custom_fields = {"spread": 0.05, "rsi": 65.2}
        market_order = (
            account.long_entry(BTC_USDT, 1.0).extra_fields(**custom_fields).build()
        )

        assert market_order.extra_fields == {"spread": 0.05, "rsi": 65.2}

    def test_extra_fields_unpacks_dictionary_for_limit_order(self, account):
        custom_fields = {"volatility": 0.23, "volume_ratio": 1.8}
        limit_order = (
            account.long_entry(BTC_USDT, 1.0)
            .limit(100.0)
            .extra_fields(**custom_fields)
            .build()
        )

        assert limit_order.extra_fields == {"volatility": 0.23, "volume_ratio": 1.8}

    def test_extra_fields_hybrid_usage(self, account):
        custom_fields = {"spread": 0.05}
        market_order = (
            account.long_entry(BTC_USDT, 1.0)
            .extra_fields(rsi=65.2, **custom_fields)
            .build()
        )

        assert market_order.extra_fields == {"spread": 0.05, "rsi": 65.2}

    def test_extra_fields_without_any_fields(self, account):
        market_order = account.long_entry(BTC_USDT, 1.0).build()

        assert market_order.extra_fields is None

    def test_extra_fields_accumulate(self, account):
        market_order = (
            account.long_entry(BTC_USDT, 1.0)
            .extra_fields(spread=0.05)
            .extra_fields(rsi=65.2)
            .build()
        )

        assert market_order.extra_fields == {"spread": 0.05, "rsi": 65.2}


class TestRiskRatio:
    """A risk rule on the chain sizes from the stop-loss attached anywhere on it."""

    def test_calculates_amount_for_long_position(self, account):
        snapshot = _snapshot({"USDT": Balance(locked=0.0, total=10000.0)})

        order = (
            account.long_entry(BTC_USDT)
            .stop_loss(48000.0)
            .size(RiskRatio(0.02, of="total_balance"), 50000.0, snapshot)
            .build()
        )

        assert order.quantity == 0.1

    def test_calculates_amount_for_short_position(self, account):
        snapshot = _snapshot({"USDT": Balance(locked=0.0, total=10000.0)})

        order = (
            account.short_entry(BTC_USDT)
            .stop_loss(52000.0)
            .size(RiskRatio(0.02, of="total_balance"), 50000.0, snapshot)
            .build()
        )

        assert order.quantity == 0.1

    def test_uses_total_balance_not_available(self, account):
        snapshot = _snapshot({"USDT": Balance(locked=5000.0, total=10000.0)})

        order = (
            account.long_entry(BTC_USDT)
            .stop_loss(48000.0)
            .size(RiskRatio(0.02, of="total_balance"), 50000.0, snapshot)
            .build()
        )

        assert order.quantity == 0.1

    def test_calculates_with_tight_stop_loss(self, account):
        snapshot = _snapshot({"USDT": Balance(locked=0.0, total=10000.0)})

        order = (
            account.long_entry(BTC_USDT)
            .stop_loss(49500.0)
            .size(RiskRatio(0.02, of="total_balance"), 50000.0, snapshot)
            .build()
        )

        assert order.quantity == 0.4

    def test_calculates_with_wide_stop_loss(self, account):
        snapshot = _snapshot({"USDT": Balance(locked=0.0, total=10000.0)})

        order = (
            account.long_entry(BTC_USDT)
            .stop_loss(40000.0)
            .size(RiskRatio(0.02, of="total_balance"), 50000.0, snapshot)
            .build()
        )

        assert order.quantity == 0.02

    def test_works_with_limit_orders(self, account):
        snapshot = _snapshot({"USDT": Balance(locked=0.0, total=10000.0)})

        order = (
            account.long_entry(BTC_USDT)
            .size(RiskRatio(0.02, of="total_balance"), 50000.0, snapshot)
            .limit(49000.0)
            .stop_loss(48000.0)
            .build()
        )

        assert isinstance(order, FuturesLimitOrderAction)
        assert order.quantity == 0.1
        assert order.price == 49000.0

    def test_stop_loss_is_preserved_in_order(self, account):
        snapshot = _snapshot({"USDT": Balance(locked=0.0, total=10000.0)})

        order = (
            account.long_entry(BTC_USDT)
            .stop_loss(48000.0)
            .size(RiskRatio(0.02, of="total_balance"), 50000.0, snapshot)
            .build()
        )

        assert order.stop_loss == StopLoss(trigger_price=48000.0)


class TestTag:
    def test_market_order_carries_the_tag_in_its_client_order_id(self, account):
        market_order = account.long_entry(BTC_USDT, 1.0).tag("r2").build()

        assert tag_of(market_order.client_order_id) == "r2"

    def test_limit_order_carries_the_tag_in_its_client_order_id(self, account):
        limit_order = account.long_entry(BTC_USDT, 1.0).limit(100.0).tag("r3").build()

        assert tag_of(limit_order.client_order_id) == "r3"

    def test_untagged_order(self, account):
        market_order = account.long_entry(BTC_USDT, 1.0).build()

        assert tag_of(market_order.client_order_id) is None

    def test_last_tag_wins(self, account):
        market_order = account.long_entry(BTC_USDT, 1.0).tag("r1").tag("r2").build()

        assert tag_of(market_order.client_order_id) == "r2"

    def test_rejects_empty_tag(self, account):
        with pytest.raises(ValueError, match="Tag cannot be empty"):
            account.long_entry(BTC_USDT, 1.0).tag("")


class TestSymbolValidation:
    def test_rejects_spot_symbol(self, account):
        spot_symbol = Symbol.create("BTC/USDT")
        with pytest.raises(ValueError, match="futures symbol"):
            account.long_entry(spot_symbol, 1.0)


class TestActionIdentity:
    def test_each_action_gets_its_own_id(self, account):
        first = account.long_entry(BTC_USDT, 1.0).build()
        second = account.long_entry(BTC_USDT, 1.0).build()

        assert first.id != second.id

    def test_actions_can_be_booked_together(self, account):
        bookkeeper = BookKeeper()

        bookkeeper.add(account.long_entry(BTC_USDT, 1.0).build())
        bookkeeper.add(account.short_entry(BTC_USDT, 1.0).build())

        assert len(bookkeeper.list_actions()) == 2


class TestPlacementReserve:
    """An account whose venue holds back a share of an order at placement sizes
    a balance or an equity ratio against what is left once that share is set
    aside, so an order sized from a whole balance is one the venue accepts; a
    risk ratio is sized off a stop and sets nothing aside.
    """

    def test_available_balance_ratio_sets_the_reserve_aside(self, reserving_account):
        snapshot = _snapshot({"USDT": Balance(locked=0.0, total=10_000.0)})

        order = (
            reserving_account.long_entry(BTC_USDT)
            .size(AvailableBalanceRatio(1.0), 2.0, snapshot)
            .build()
        )

        assert order.quantity == 4_000.0

    def test_total_balance_ratio_sets_the_reserve_aside(self, reserving_account):
        snapshot = _snapshot({"USDT": Balance(locked=2_000.0, total=10_000.0)})

        order = (
            reserving_account.long_entry(BTC_USDT)
            .size(TotalBalanceRatio(1.0), 2.0, snapshot)
            .build()
        )

        assert order.quantity == 4_000.0

    def test_equity_ratio_sets_the_reserve_aside(self, reserving_account):
        snapshot = _snapshot(
            {"USDT": Balance(locked=0.0, total=10_000.0)}, {"USDT": 10_000.0}
        )

        order = (
            reserving_account.long_entry(BTC_USDT)
            .size(EquityRatio(1.0), 2.0, snapshot)
            .build()
        )

        assert order.quantity == 4_000.0

    def test_risk_ratio_sizes_against_a_stop_and_sets_nothing_aside(
        self, reserving_account
    ):
        snapshot = _snapshot({"USDT": Balance(locked=0.0, total=10_000.0)})

        order = (
            reserving_account.long_entry(BTC_USDT)
            .size(RiskRatio(0.1, of="total_balance"), 100.0, snapshot)
            .stop_loss(98.0)
            .build()
        )

        assert order.quantity == 500.0
