from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from robottraderslab import Symbol
from robottraderslab.exceptions import StrategyCriticalError
from robottraderslab.exchanges import Balance, MarginMode, MarginSettings, PositionSide
from robottraderslab.futures import FuturesAccount, FuturesOrderBuilder
from robottraderslab.strategies import (
    AccountSnapshot,
    BookKeeper,
    Profile,
    ProfileStrategy,
    TradingMode,
    TradingSystem,
)
from robottraderslab.strategies.futures import (
    AvailableBalanceRatio,
    EquityRatio,
    Margin,
    Notional,
    Quantity,
    RiskRatio,
    SizingRule,
    TotalBalanceRatio,
)

BTC_USDT = Symbol.create("BTC/USDT:USDT")
PRICE = 100.0
STOP = 95.0
SNAPSHOT = AccountSnapshot(
    account_name="test",
    balances={"USDT": Balance(locked=2_000.0, total=10_000.0)},
    equities={"USDT": 12_000.0},
    positions={},
    conversion_rates={BTC_USDT: 1.0},
)


def _at_leverage(leverage: float | None) -> AccountSnapshot:
    return AccountSnapshot(
        account_name="test",
        balances={"USDT": Balance(locked=0.0, total=10_000.0)},
        positions={},
        margin_settings={
            BTC_USDT: MarginSettings(leverage=leverage, margin_mode=MarginMode.CROSS)
        },
        conversion_rates={BTC_USDT: 1.0},
    )


def _consumed_at_placement(quantity: float, *, leverage: float) -> float:
    """The margin locked and the reserving account's reserve on the worth."""
    worth = quantity * PRICE
    return worth / leverage + worth * 0.25


@dataclass(frozen=True, kw_only=True)
class _SizedProfile(Profile):
    sizing: SizingRule


@dataclass(frozen=True, slots=True)
class _ShareOfLockedMargin(SizingRule):
    ratio: float

    def quantity(
        self,
        symbol: Symbol,
        price: float,
        account_snapshot: AccountSnapshot,
        *,
        placement_reserve_rate: float,
        placement_requirement_rate: Callable[[float], float],
        stop_loss_price: float | None,
    ) -> float:
        return account_snapshot.balance("USDT").locked * self.ratio / price


class _SizedStrategy(ProfileStrategy[_SizedProfile]):
    sizing_rules = {"locked_margin_ratio": _ShareOfLockedMargin}


def _rule_of(key: str, value: Any) -> SizingRule:
    strategy = _SizedStrategy(
        account=object(),
        trading_system=TradingSystem(trading_mode=TradingMode.BACKTEST),
        config_dir=None,
        profiles=[{"symbol": str(BTC_USDT), "timeframe": "4h", key: value}],
    )
    (profile,) = strategy.profiles
    return profile.sizing


def _entry(account: FuturesAccount) -> FuturesOrderBuilder:
    return account.long_entry(BTC_USDT)


ENTRY_PAIRS: list[tuple[str, Any, Callable[[FuturesAccount], FuturesOrderBuilder]]] = [
    (
        "total_balance_ratio",
        0.15,
        lambda account: _entry(account).size(TotalBalanceRatio(0.15), PRICE, SNAPSHOT),
    ),
    (
        "available_balance_ratio",
        0.15,
        lambda account: _entry(account).size(
            AvailableBalanceRatio(0.15), PRICE, SNAPSHOT
        ),
    ),
    (
        "equity_ratio",
        0.15,
        lambda account: _entry(account).size(EquityRatio(0.15), PRICE, SNAPSHOT),
    ),
    (
        "risk_ratio",
        {"ratio": 0.02, "of": "total_balance"},
        lambda account: _entry(account).size(
            RiskRatio(0.02, of="total_balance"), PRICE, SNAPSHOT
        ),
    ),
    ("quantity", 3.0, lambda account: account.long_entry(BTC_USDT, 3.0)),
]


class TestEntrySizing:
    @pytest.mark.parametrize(("key", "value", "builder_method"), ENTRY_PAIRS)
    def test_a_rule_parsed_from_its_key_sizes_as_the_rule_built_in_code(
        self, reserving_account, key, value, builder_method
    ):
        by_rule = (
            _entry(reserving_account)
            .size(_rule_of(key, value), PRICE, SNAPSHOT)
            .stop_loss(STOP)
            .build()
        )
        by_method = builder_method(reserving_account).stop_loss(STOP).build()

        assert by_rule.quantity == by_method.quantity

    def test_a_rule_the_strategy_registers(self, account):
        rule = _rule_of("locked_margin_ratio", 0.5)

        order = _entry(account).size(rule, PRICE, SNAPSHOT).build()

        assert order.quantity == pytest.approx(10.0)

    def test_notional_is_the_positions_worth_in_the_margin_currency(self, account):
        order = _entry(account).size(Notional(200.0), PRICE, SNAPSHOT).build()

        assert order.quantity == pytest.approx(2.0)

    def test_notional_sets_the_placement_reserve_aside(self, reserving_account):
        order = _entry(reserving_account).size(Notional(200.0), PRICE, SNAPSHOT).build()

        assert order.quantity == pytest.approx(2.0 / 1.25)

    def test_margin_at_leverage_one_sizes_as_notional(self, reserving_account):
        snapshot = _at_leverage(1.0)

        by_margin = _entry(reserving_account).size(Margin(40.0), PRICE, snapshot)
        by_notional = _entry(reserving_account).size(Notional(40.0), PRICE, snapshot)

        assert by_margin.build().quantity == by_notional.build().quantity

    def test_margin_is_the_positions_worth_over_the_leverage(self, account):
        order = _entry(account).size(Margin(40.0), PRICE, _at_leverage(5.0)).build()

        assert order.quantity == pytest.approx(2.0)

    def test_margin_covers_the_reserve_on_the_positions_worth(self, reserving_account):
        order = (
            _entry(reserving_account)
            .size(Margin(40.0), PRICE, _at_leverage(5.0))
            .build()
        )

        assert _consumed_at_placement(order.quantity, leverage=5.0) == pytest.approx(
            40.0
        )

    def test_margin_on_a_symbol_without_leverage(self, account):
        with pytest.raises(
            StrategyCriticalError,
            match=r"reports none in cross margin mode: set a leverage on the symbol",
        ):
            _entry(account).size(Margin(40.0), PRICE, _at_leverage(None)).build()

    def test_a_rule_reads_a_stop_loss_attached_after_it(self, account):
        order = (
            _entry(account)
            .size(RiskRatio(0.02), PRICE, SNAPSHOT)
            .stop_loss(STOP)
            .build()
        )

        assert order.quantity == pytest.approx(12_000.0 * 0.02 / 5.0)

    @pytest.mark.parametrize("price", [0.0, float("nan")])
    def test_a_price_not_above_zero(self, account, price):
        with pytest.raises(ValueError, match="`price` must be greater than 0"):
            _entry(account).size(Notional(200.0), price, SNAPSHOT)

    def test_an_order_sized_twice(self, account):
        entry = _entry(account).size(TotalBalanceRatio(0.1), PRICE, SNAPSHOT)

        with pytest.raises(StrategyCriticalError, match="already sized"):
            entry.size(Notional(200.0), PRICE, SNAPSHOT)

    def test_a_rule_on_an_order_given_a_quantity(self, account):
        entry = account.long_entry(BTC_USDT, 1.0)

        with pytest.raises(StrategyCriticalError, match="already sized"):
            entry.size(Notional(200.0), PRICE, SNAPSHOT)


class TestRiskRatio:
    def test_the_stop_loss_costs_the_share_of_equity(self, account):
        snapshot = AccountSnapshot(
            account_name="test",
            balances={"USDT": Balance(locked=0.0, total=8_000.0)},
            equities={"USDT": 10_000.0},
            conversion_rates={BTC_USDT: 1.0},
        )

        order = (
            _entry(account)
            .stop_loss(STOP)
            .size(_rule_of("risk_ratio", 0.02), PRICE, snapshot)
            .build()
        )

        assert order.quantity == pytest.approx(40.0)

    def test_the_share_of_the_available_balance(self, account):
        order = (
            _entry(account)
            .stop_loss(STOP)
            .size(RiskRatio(0.02, of="available_balance"), PRICE, SNAPSHOT)
            .build()
        )

        assert order.quantity == pytest.approx(8_000.0 * 0.02 / 5.0)

    def test_an_order_without_a_stop_loss_is_refused_at_booking(self, account):
        entry = _entry(account).size(RiskRatio(0.02), PRICE, SNAPSHOT)

        with pytest.raises(
            StrategyCriticalError, match=r"an entry takes one with stop_loss\(\)"
        ):
            BookKeeper().add(entry)

    def test_a_reference_it_does_not_know(self):
        with pytest.raises(ValueError, match="`of` must be one of"):
            RiskRatio(0.02, of="margin")

    def test_reads_equity_only_when_equity_is_its_reference(self):
        assert RiskRatio(0.02).reads_equity is True
        assert RiskRatio(0.02, of="total_balance").reads_equity is False


TARGET_PAIRS = [
    ("total_balance_ratio", 0.15, TotalBalanceRatio(0.15)),
    ("available_balance_ratio", 0.15, AvailableBalanceRatio(0.15)),
    ("equity_ratio", 0.15, EquityRatio(0.15)),
    ("quantity", 3.0, Quantity(3.0)),
]


class TestTargetSizing:
    @pytest.mark.parametrize(("key", "value", "rule"), TARGET_PAIRS)
    def test_a_rule_parsed_from_its_key_sizes_as_the_rule_built_in_code(
        self, reserving_account, key, value, rule
    ):
        target = reserving_account.long_target(BTC_USDT, SNAPSHOT)

        by_key = target.size(_rule_of(key, value), PRICE).build()
        by_rule = target.size(rule, PRICE).build()

        assert by_key.quantity == by_rule.quantity

    def test_a_quantity_target_sizes_as_the_quantity_rule(self, reserving_account):
        target = reserving_account.long_target(BTC_USDT, SNAPSHOT)

        by_call = target.quantity(3.0, PRICE).build()
        by_rule = target.size(Quantity(3.0), PRICE).build()

        assert by_call.quantity == by_rule.quantity

    def test_the_same_quantity_as_an_entry_sized_by_the_rule(self, reserving_account):
        rule = Notional(200.0)

        target = reserving_account.long_target(BTC_USDT, SNAPSHOT).size(rule, PRICE)
        entry = _entry(reserving_account).size(rule, PRICE, SNAPSHOT)

        assert target.build().quantity == entry.build().quantity

    def test_margin_sizes_a_target_as_an_entry(self, reserving_account):
        snapshot = _at_leverage(5.0)

        target = reserving_account.long_target(BTC_USDT, snapshot)
        entry = _entry(reserving_account).size(Margin(40.0), PRICE, snapshot)

        sized = target.size(Margin(40.0), PRICE).build()

        assert sized.quantity == entry.build().quantity
        assert _consumed_at_placement(sized.quantity, leverage=5.0) == pytest.approx(
            40.0
        )

    def test_a_position_standing_at_the_target(self, account, make_position):
        snapshot = AccountSnapshot(
            account_name="test",
            balances={"USDT": Balance(locked=0.0, total=10_000.0)},
            positions={BTC_USDT: make_position(PositionSide.LONG, 2.0)},
            conversion_rates={BTC_USDT: 1.0},
        )

        order = account.long_target(BTC_USDT, snapshot).size(Notional(200.0), PRICE)

        assert order is None

    @pytest.mark.parametrize("price", [0.0, float("nan")])
    def test_a_price_not_above_zero(self, account, price):
        target = account.long_target(BTC_USDT, SNAPSHOT)

        with pytest.raises(ValueError, match="`price` must be greater than 0"):
            target.size(Notional(200.0), price)

    def test_a_risk_rule_has_no_stop_loss_to_size_from(self, account):
        target = account.long_target(BTC_USDT, SNAPSHOT)

        with pytest.raises(
            StrategyCriticalError,
            match="a position target has no stop-loss to size from",
        ):
            target.size(RiskRatio(0.02), PRICE)


class TestRuleValues:
    @pytest.mark.parametrize(
        "rule_class",
        [TotalBalanceRatio, AvailableBalanceRatio, EquityRatio, RiskRatio],
    )
    @pytest.mark.parametrize("ratio", [-0.1, 0.0, 1.5, float("nan")])
    def test_a_share_outside_zero_to_one(self, rule_class, ratio):
        with pytest.raises(ValueError, match="`ratio` must be above 0.0"):
            rule_class(ratio)

    @pytest.mark.parametrize("rule_class", [TotalBalanceRatio, RiskRatio])
    def test_the_whole_share(self, rule_class):
        assert rule_class(1.0).ratio == 1.0

    @pytest.mark.parametrize("rule_class", [Margin, Notional, Quantity])
    @pytest.mark.parametrize("amount", [-1.0, 0.0, float("nan")])
    def test_an_amount_that_is_not_positive(self, rule_class, amount):
        with pytest.raises(ValueError, match="must be greater than 0"):
            rule_class(amount)
