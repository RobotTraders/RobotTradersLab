import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest
from rung_ids import PROCESS_TAG, rung_client_order_id

from robottraderslab import Symbol
from robottraderslab.exchanges import OrderType
from robottraderslab.strategies import (
    AccountSnapshots,
    BookKeeper,
    Execution,
    OHLCVs,
    OrderProtocol,
    OrderSide,
    PositionSide,
    StrategyRequirements,
)
from robottraderslab.strategies.futures import (
    BatchableOrderAction,
    CancelOrderByIdAction,
    CancelOrdersForSymbolAction,
    FuturesLimitOrderAction,
    FuturesMarketOrderAction,
    FuturesOrderBatchAction,
    FuturesOrderModifyAction,
    MarginMode,
    MarginSettings,
    OrderModification,
    PositionSnapshot,
    SetLeverageAction,
    SetMarginModeAction,
    UpdatePositionStopLossAction,
    UpdatePositionTakeProfitAction,
)
from robottraderslab_envelope import EnvelopeStrategy

BTC = Symbol.create("BTC/USDT:USDT")
TIMEFRAME = "1d"
TIMESTAMP = datetime(2024, 2, 19)
SETUP_ID = "BTC/USDT:USDT"


def _stand_on_first_candle(ohlcvs: OHLCVs) -> None:
    """Set the iteration cursor so `current` and `signal` read the first candle."""
    next(ohlcvs._iter_timeframes())


def _make_ohlcvs(
    *,
    close: float = 100.0,
    reference: float = 100.0,
    band_low_1: float = 95.0,
    band_high_1: float = 105.0,
) -> OHLCVs:
    dates = pd.date_range("2024-02-19", periods=1, freq="1D")
    df = pd.DataFrame(
        {
            "open": [close],
            "high": [close + 1],
            "low": [close - 1],
            "close": [close],
            "volume": [1000.0],
        },
        index=dates,
    )
    ohlcvs = OHLCVs({TIMEFRAME: {BTC: df}})
    ohlcvs.add_column(BTC, TIMEFRAME, f"{SETUP_ID}_reference", [reference])
    ohlcvs.add_column(BTC, TIMEFRAME, f"{SETUP_ID}_band_low_1", [band_low_1])
    ohlcvs.add_column(BTC, TIMEFRAME, f"{SETUP_ID}_band_high_1", [band_high_1])

    _stand_on_first_candle(ohlcvs)

    return ohlcvs


PROFILE = {
    "symbol": "BTC/USDT:USDT",
    "timeframe": TIMEFRAME,
    "average_type": "SMA",
    "average_period": 100,
    "envelopes": [0.05],
    "stop_loss_pct": 0.04,
    "total_balance_ratio": 0.5,
    "leverage": 5.0,
    "margin_mode": "isolated",
}


def _directional_profile(**envelope_keys: object) -> dict[str, object]:
    return {
        key: value for key, value in PROFILE.items() if key != "envelopes"
    } | envelope_keys


PROFILE_LONG_ONLY = _directional_profile(long_envelopes=[0.05])
PROFILE_SHORT_ONLY = _directional_profile(short_envelopes=[0.05])
PROFILE_UNEQUAL_SIDES = _directional_profile(
    long_envelopes=[0.05, 0.10, 0.15], short_envelopes=[0.05], stop_loss_pct=0.30
)


_RUNG_ACTION = {"trigger": FuturesMarketOrderAction, "limit": FuturesLimitOrderAction}
_OTHER_ENTRY_ORDER = {"trigger": "limit", "limit": "trigger"}


@pytest.fixture
def make_strategy(
    make_strategy: Callable[..., EnvelopeStrategy], entry_order: str
) -> Callable[..., EnvelopeStrategy]:
    def _make(
        profiles: list[dict[str, Any]], **strategy_kwargs: Any
    ) -> EnvelopeStrategy:
        return make_strategy(profiles, entry_order=entry_order, **strategy_kwargs)

    return _make


@pytest.fixture
def strategy(
    make_strategy: Callable[..., EnvelopeStrategy], requirements: StrategyRequirements
) -> EnvelopeStrategy:
    strategy = make_strategy([PROFILE])
    asyncio.run(strategy.setup(requirements))
    return strategy


def _rung_orders(
    bookkeeper: BookKeeper, entry_order: str
) -> list[BatchableOrderAction]:
    return [
        order
        for action in bookkeeper.list_actions()
        if isinstance(action, FuturesOrderBatchAction)
        for order in action.orders
        if isinstance(order, _RUNG_ACTION[entry_order])
    ]


def _level(order: BatchableOrderAction, entry_order: str) -> float:
    if not isinstance(order, _RUNG_ACTION[entry_order]):
        pytest.fail(f"{order!r} does not rest as a {entry_order} entry")
    level = (
        order.price
        if isinstance(order, FuturesLimitOrderAction)
        else order.trigger_price
    )
    if level is None:
        pytest.fail(f"{order!r} carries no level")
    return level


def _modifications(bookkeeper: BookKeeper) -> list[OrderModification]:
    return [
        modification
        for action in bookkeeper.list_actions()
        if isinstance(action, FuturesOrderModifyAction)
        for modification in action.modifications
    ]


def _cancelled_order_ids(bookkeeper: BookKeeper) -> list[str]:
    return sorted(
        action.order_id
        for action in bookkeeper.list_actions()
        if isinstance(action, CancelOrderByIdAction)
    )


def _reshaped_levels(bookkeeper: BookKeeper, entry_order: str) -> dict[str, float]:
    return {
        modification.order_id: _level(modification.order, entry_order)
        for modification in _modifications(bookkeeper)
    }


def _placed_levels(bookkeeper: BookKeeper, entry_order: str) -> list[float]:
    return [
        _level(order, entry_order)
        for order in _rung_orders(bookkeeper, entry_order)
        if not order.reduce_only
    ]


class TestMarginTargets:
    def test_each_symbol_declares_its_profiles_settings(self, make_strategy):
        requirements = StrategyRequirements()

        asyncio.run(make_strategy([PROFILE]).setup(requirements))

        assert requirements.account._get_all()[0].margin_targets == {
            BTC: MarginSettings(leverage=5.0, margin_mode=MarginMode.ISOLATED)
        }

    def test_a_drifted_symbol_books_no_margin_action(
        self, strategy, make_account_state
    ):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(), make_account_state(), TIMESTAMP, bookkeeper, [TIMEFRAME]
        )

        setters = [
            action
            for action in bookkeeper.list_actions()
            if isinstance(action, (SetLeverageAction, SetMarginModeAction))
        ]
        assert setters == []


class TestReconciliation:
    def test_flat_book_without_resting_orders(self, strategy, make_account_state):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(), make_account_state(), TIMESTAMP, bookkeeper, [TIMEFRAME]
        )

        cancels = [
            a
            for a in bookkeeper.list_actions()
            if isinstance(a, (CancelOrdersForSymbolAction, CancelOrderByIdAction))
        ]
        assert cancels == []
        assert _modifications(bookkeeper) == []

    def test_resting_rungs_are_reshaped_to_the_new_bands(
        self, strategy, entry_order, make_account_state
    ):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(),
            make_account_state(
                open_orders=[
                    _fake_order(
                        "long-1", entry_order, OrderSide.BUY, level=96.0, rung=1
                    ),
                    _fake_order(
                        "short-1",
                        entry_order,
                        OrderSide.SELL,
                        level=104.0,
                        rung=1,
                    ),
                ]
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        reshaped = {
            modification.order_id: _level(modification.order, entry_order)
            for modification in _modifications(bookkeeper)
        }
        assert reshaped == {"long-1": 95.0, "short-1": 105.0}
        assert _rung_orders(bookkeeper, entry_order) == []

    def test_extra_resting_rungs_are_cancelled(
        self, strategy, entry_order, make_account_state
    ):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(),
            make_account_state(
                open_orders=[
                    _fake_order(
                        "long-1", entry_order, OrderSide.BUY, level=96.0, rung=1
                    ),
                    _fake_order(
                        "long-2", entry_order, OrderSide.BUY, level=93.0, rung=2
                    ),
                    _fake_order(
                        "short-1",
                        entry_order,
                        OrderSide.SELL,
                        level=104.0,
                        rung=1,
                    ),
                ]
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        cancelled_ids = _cancelled_order_ids(bookkeeper)
        reshaped_ids = {m.order_id for m in _modifications(bookkeeper)}
        assert cancelled_ids == ["long-2"]
        assert reshaped_ids == {"long-1", "short-1"}

    def test_second_rung_on_the_same_band_is_cancelled(
        self, strategy, entry_order, make_account_state
    ):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(),
            make_account_state(
                open_orders=[
                    _fake_order(
                        "long-1a", entry_order, OrderSide.BUY, level=96.0, rung=1
                    ),
                    _fake_order(
                        "long-1b", entry_order, OrderSide.BUY, level=94.0, rung=1
                    ),
                ]
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        cancelled_ids = _cancelled_order_ids(bookkeeper)
        reshaped_ids = {m.order_id for m in _modifications(bookkeeper)}
        assert cancelled_ids == ["long-1b"]
        assert reshaped_ids == {"long-1a"}

    def test_surviving_rung_keeps_its_own_band_while_the_others_refill(
        self, make_strategy, entry_order, make_account_state
    ):
        strategy = make_strategy([PROFILE_THREE_RUNGS])
        asyncio.run(strategy.setup(StrategyRequirements()))
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_three_rung_ohlcvs(),
            make_account_state(
                open_orders=[
                    _fake_order(
                        "rung-3", entry_order, OrderSide.BUY, level=86.0, rung=3
                    ),
                ]
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        reshaped = {
            modification.order_id: _level(modification.order, entry_order)
            for modification in _modifications(bookkeeper)
        }
        placed_long_levels = sorted(
            _level(order, entry_order)
            for order in _rung_orders(bookkeeper, entry_order)
            if order.side == OrderSide.BUY
        )
        assert reshaped == {"rung-3": 85.0}
        assert placed_long_levels == [90.0, 95.0]

    def test_resting_rung_of_another_program_is_cancelled(
        self, strategy, entry_order, make_account_state
    ):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(),
            make_account_state(
                open_orders=[
                    _fake_order("manual-1", entry_order, OrderSide.BUY, level=96.0)
                ]
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        cancelled_ids = _cancelled_order_ids(bookkeeper)
        assert cancelled_ids == ["manual-1"]
        assert _modifications(bookkeeper) == []

    def test_a_rung_resting_under_a_tag_of_no_band_is_cancelled_and_placed_anew(
        self, strategy, entry_order, make_account_state
    ):
        foreign = _fake_order("foreign-1", entry_order, OrderSide.BUY, level=96.0)
        foreign.client_order_id = f"{PROCESS_TAG}-r1"
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(),
            make_account_state(open_orders=[foreign]),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        placed = {
            order.side: _level(order, entry_order)
            for order in _rung_orders(bookkeeper, entry_order)
        }
        assert _cancelled_order_ids(bookkeeper) == ["foreign-1"]
        assert _modifications(bookkeeper) == []
        assert placed == {OrderSide.BUY: 95.0, OrderSide.SELL: 105.0}

    def test_stray_resting_order_is_cancelled(
        self, strategy, entry_order, make_account_state
    ):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(),
            make_account_state(
                open_orders=[
                    _fake_order(
                        "ghost-1", _OTHER_ENTRY_ORDER[entry_order], OrderSide.BUY
                    ),
                    _fake_order("sl-1", "stop-loss", OrderSide.SELL),
                    _fake_order("tp-1", "take-profit", OrderSide.SELL),
                ]
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        cancelled_ids = _cancelled_order_ids(bookkeeper)
        assert cancelled_ids == ["ghost-1"]


class TestRungShape:
    def test_long_rung_rests_at_low_band(
        self, strategy, entry_order, make_account_state
    ):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(band_low_1=95.0),
            make_account_state(),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        long_orders = [
            o for o in _rung_orders(bookkeeper, entry_order) if o.side == OrderSide.BUY
        ]
        assert len(long_orders) == 1
        assert _level(long_orders[0], entry_order) == 95.0

    def test_short_rung_rests_at_high_band(
        self, strategy, entry_order, make_account_state
    ):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(band_high_1=105.0),
            make_account_state(),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        short_orders = [
            o for o in _rung_orders(bookkeeper, entry_order) if o.side == OrderSide.SELL
        ]
        assert len(short_orders) == 1
        assert _level(short_orders[0], entry_order) == 105.0

    def test_long_rung_stop_loss_below_entry(
        self, strategy, entry_order, make_account_state
    ):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(band_low_1=100.0),
            make_account_state(),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        long_orders = [
            o for o in _rung_orders(bookkeeper, entry_order) if o.side == OrderSide.BUY
        ]
        assert long_orders[0].stop_loss.trigger_price == pytest.approx(
            100.0 * (1 - 0.04)
        )

    def test_short_rung_stop_loss_above_entry(
        self, strategy, entry_order, make_account_state
    ):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(band_high_1=100.0),
            make_account_state(),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        short_orders = [
            o for o in _rung_orders(bookkeeper, entry_order) if o.side == OrderSide.SELL
        ]
        assert short_orders[0].stop_loss.trigger_price == pytest.approx(
            100.0 * (1 + 0.04)
        )

    def test_rung_take_profit_at_reference(
        self, strategy, entry_order, make_account_state
    ):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(reference=110.0),
            make_account_state(),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        take_profit_by_side = {
            order.side: order.take_profit.trigger_price
            for order in _rung_orders(bookkeeper, entry_order)
        }
        assert take_profit_by_side == {
            OrderSide.BUY: pytest.approx(110.0),
            OrderSide.SELL: pytest.approx(110.0),
        }

    def test_no_rung_rests_as_the_other_entry_order(
        self, strategy, entry_order, make_account_state
    ):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(), make_account_state(), TIMESTAMP, bookkeeper, [TIMEFRAME]
        )

        assert _rung_orders(bookkeeper, _OTHER_ENTRY_ORDER[entry_order]) == []

    def test_rung_carries_its_tag(self, strategy, entry_order, make_account_state):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(), make_account_state(), TIMESTAMP, bookkeeper, [TIMEFRAME]
        )

        tags = {order.tag for order in _rung_orders(bookkeeper, entry_order)}
        assert tags == {"1d-l1", "1d-s1"}

    def test_rungs_are_sized_from_the_balance(
        self, strategy, entry_order, make_account_state
    ):
        total_balance = 10_000.0
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(),
            make_account_state(total_balance=total_balance),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        quantity_by_level = {
            _level(order, entry_order): order.quantity
            for order in _rung_orders(bookkeeper, entry_order)
        }
        notional = total_balance * PROFILE["total_balance_ratio"]
        assert quantity_by_level == {
            95.0: pytest.approx(notional / 95.0),
            105.0: pytest.approx(notional / 105.0),
        }


class TestDirectionalProfiles:
    def test_long_envelopes_alone_emits_only_long_rungs(
        self, make_strategy, entry_order, make_account_state
    ):
        strategy = make_strategy([PROFILE_LONG_ONLY])
        asyncio.run(strategy.setup(StrategyRequirements()))
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(), make_account_state(), TIMESTAMP, bookkeeper, [TIMEFRAME]
        )

        orders = _rung_orders(bookkeeper, entry_order)
        assert len(orders) == 1
        assert orders[0].side == OrderSide.BUY

    def test_short_envelopes_alone_emits_only_short_rungs(
        self, make_strategy, entry_order, make_account_state
    ):
        strategy = make_strategy([PROFILE_SHORT_ONLY])
        asyncio.run(strategy.setup(StrategyRequirements()))
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(), make_account_state(), TIMESTAMP, bookkeeper, [TIMEFRAME]
        )

        orders = _rung_orders(bookkeeper, entry_order)
        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL

    def test_unequal_side_counts_emit_their_own_number_of_rungs(
        self, make_strategy, entry_order, make_account_state
    ):
        strategy = make_strategy([PROFILE_UNEQUAL_SIDES])
        asyncio.run(strategy.setup(StrategyRequirements()))
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_three_rung_ohlcvs(),
            make_account_state(),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        orders = _rung_orders(bookkeeper, entry_order)
        long_orders = [o for o in orders if o.side == OrderSide.BUY]
        short_orders = [o for o in orders if o.side == OrderSide.SELL]
        assert len(long_orders) == 3
        assert len(short_orders) == 1

    def test_unequal_side_counts_size_rungs_from_their_own_side(
        self, make_strategy, entry_order, make_account_state
    ):
        strategy = make_strategy([PROFILE_UNEQUAL_SIDES])
        asyncio.run(strategy.setup(StrategyRequirements()))
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_three_rung_ohlcvs(),
            make_account_state(),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        quantity_by_level = {
            _level(order, entry_order): order.quantity
            for order in _rung_orders(bookkeeper, entry_order)
        }
        balance = 10_000.0
        long_ratio = PROFILE_UNEQUAL_SIDES["total_balance_ratio"] / 3
        short_ratio = PROFILE_UNEQUAL_SIDES["total_balance_ratio"] / 1
        assert quantity_by_level[95.0] == pytest.approx(balance * long_ratio / 95.0)
        assert quantity_by_level[90.0] == pytest.approx(balance * long_ratio / 90.0)
        assert quantity_by_level[85.0] == pytest.approx(balance * long_ratio / 85.0)
        assert quantity_by_level[105.0] == pytest.approx(balance * short_ratio / 105.0)

    def test_a_notional_splits_over_the_rungs_of_each_side(
        self, make_strategy, entry_order, make_account_state
    ):
        profile = {**PROFILE_UNEQUAL_SIDES, "notional": 300.0}
        del profile["total_balance_ratio"]
        strategy = make_strategy([profile])
        asyncio.run(strategy.setup(StrategyRequirements()))
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_three_rung_ohlcvs(),
            make_account_state(),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        quantity_by_level = {
            _level(order, entry_order): order.quantity
            for order in _rung_orders(bookkeeper, entry_order)
        }
        assert quantity_by_level[90.0] == pytest.approx(300.0 / 3 / 90.0)
        assert quantity_by_level[105.0] == pytest.approx(300.0 / 105.0)

    def test_an_empty_balance_books_rungs_sized_to_nothing(
        self, make_strategy, entry_order, make_account_state
    ):
        strategy = make_strategy([PROFILE_UNEQUAL_SIDES])
        asyncio.run(strategy.setup(StrategyRequirements()))
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_three_rung_ohlcvs(),
            make_account_state(total_balance=0.0),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        assert [order.quantity for order in _rung_orders(bookkeeper, entry_order)] == [
            0.0
        ] * 4


class TestSkipConditions:
    def test_untriggered_timeframe_skips_profile(
        self, strategy, entry_order, make_account_state
    ):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(), make_account_state(), TIMESTAMP, bookkeeper, ["4h"]
        )

        assert _rung_orders(bookkeeper, entry_order) == []


class TestNanLevels:
    def test_nan_reference_cancels_orders_and_emits_no_rungs(
        self, strategy, entry_order, make_account_state
    ):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(
                reference=float("nan"),
                band_low_1=float("nan"),
                band_high_1=float("nan"),
            ),
            make_account_state(),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        cancels = [
            a
            for a in bookkeeper.list_actions()
            if isinstance(a, CancelOrdersForSymbolAction)
        ]
        assert len(cancels) == 1
        assert cancels[0].symbol == BTC
        assert _rung_orders(bookkeeper, entry_order) == []

    def test_nan_long_band_alone_emits_no_rungs(
        self, strategy, entry_order, make_account_state
    ):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(band_low_1=float("nan")),
            make_account_state(),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        assert _rung_orders(bookkeeper, entry_order) == []

    def test_nan_short_band_alone_emits_no_rungs(
        self, strategy, entry_order, make_account_state
    ):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(band_high_1=float("nan")),
            make_account_state(),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        assert _rung_orders(bookkeeper, entry_order) == []


def _fake_order(
    order_id: str,
    order_type: str,
    side: OrderSide,
    symbol: Symbol = BTC,
    level: float | None = None,
    rung: int | None = None,
) -> Mock:
    order = Mock(spec=OrderProtocol)
    order.order_id = order_id
    order.kind = order_type
    order.symbol = symbol
    order.side = side
    order.trigger_price = level if order_type == "trigger" else None
    order.price = level if order_type == "limit" else None
    order.client_order_id = (
        None if rung is None else rung_client_order_id(TIMEFRAME, side, rung)
    )
    return order


PROFILE_THREE_RUNGS = {
    **PROFILE,
    "envelopes": [0.05, 0.10, 0.15],
    "stop_loss_pct": 0.30,
}
PROFILE_THREE_SHORT_ONE_LONG = _directional_profile(
    long_envelopes=[0.05], short_envelopes=[0.05, 0.10, 0.15], stop_loss_pct=0.30
)


def _make_three_rung_ohlcvs(
    *,
    close: float = 100.0,
    reference: float = 100.0,
    band_lows: tuple[float, float, float] = (95.0, 90.0, 85.0),
    band_highs: tuple[float, float, float] = (105.0, 110.0, 115.0),
) -> OHLCVs:
    dates = pd.date_range("2024-02-19", periods=1, freq="1D")
    df = pd.DataFrame(
        {
            "open": [close],
            "high": [close + 1],
            "low": [close - 1],
            "close": [close],
            "volume": [1000.0],
        },
        index=dates,
    )
    ohlcvs = OHLCVs({TIMEFRAME: {BTC: df}})
    ohlcvs.add_column(BTC, TIMEFRAME, f"{SETUP_ID}_reference", [reference])
    for index, (low, high) in enumerate(zip(band_lows, band_highs), start=1):
        ohlcvs.add_column(BTC, TIMEFRAME, f"{SETUP_ID}_band_low_{index}", [low])
        ohlcvs.add_column(BTC, TIMEFRAME, f"{SETUP_ID}_band_high_{index}", [high])
    _stand_on_first_candle(ohlcvs)
    return ohlcvs


class TestOpenPosition:
    def test_opposite_side_rung_is_cancelled(
        self, strategy, hold_bands, entry_order, make_account_state, make_position
    ):
        hold_bands(strategy, PositionSide.LONG, 1)
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(),
            make_account_state(
                holding=make_position(PositionSide.LONG),
                open_orders=[
                    _fake_order(
                        "short-1",
                        entry_order,
                        OrderSide.SELL,
                        level=104.0,
                        rung=1,
                    ),
                ],
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        cancelled_ids = _cancelled_order_ids(bookkeeper)
        assert cancelled_ids == ["short-1"]

    def test_emits_update_take_profit_at_reference(
        self, strategy, make_account_state, make_position
    ):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(reference=110.0),
            make_account_state(holding=make_position(PositionSide.LONG)),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        tp_updates = [
            a
            for a in bookkeeper.list_actions()
            if isinstance(a, UpdatePositionTakeProfitAction)
        ]
        assert len(tp_updates) == 1
        assert tp_updates[0].symbol == BTC
        assert tp_updates[0].trigger_price == pytest.approx(110.0)

    def test_emits_update_stop_loss_from_average_entry_for_long(
        self, strategy, make_account_state, make_position
    ):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(),
            make_account_state(
                holding=make_position(PositionSide.LONG, entry_price=98.0)
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        sl_updates = [
            a
            for a in bookkeeper.list_actions()
            if isinstance(a, UpdatePositionStopLossAction)
        ]
        assert len(sl_updates) == 1
        assert sl_updates[0].trigger_price == pytest.approx(98.0 * (1 - 0.04))

    def test_emits_update_stop_loss_from_average_entry_for_short(
        self, strategy, make_account_state, make_position
    ):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(),
            make_account_state(
                holding=make_position(PositionSide.SHORT, entry_price=102.0)
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        sl_updates = [
            a
            for a in bookkeeper.list_actions()
            if isinstance(a, UpdatePositionStopLossAction)
        ]
        assert len(sl_updates) == 1
        assert sl_updates[0].trigger_price == pytest.approx(102.0 * (1 + 0.04))

    def test_long_position_reshapes_only_unfilled_long_rungs(
        self,
        make_strategy,
        requirements,
        hold_bands,
        entry_order,
        make_account_state,
        make_position,
    ):
        strategy = make_strategy([PROFILE_THREE_RUNGS])
        asyncio.run(strategy.setup(requirements))
        hold_bands(strategy, PositionSide.LONG, 1)
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_three_rung_ohlcvs(),
            make_account_state(
                holding=make_position(PositionSide.LONG),
                open_orders=[
                    _fake_order(
                        "rung-2", entry_order, OrderSide.BUY, level=91.0, rung=2
                    ),
                    _fake_order(
                        "rung-3", entry_order, OrderSide.BUY, level=86.0, rung=3
                    ),
                ],
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        reshaped = {
            modification.order_id: _level(modification.order, entry_order)
            for modification in _modifications(bookkeeper)
        }
        assert reshaped == {"rung-2": 90.0, "rung-3": 85.0}
        assert _rung_orders(bookkeeper, entry_order) == []

    def test_short_position_reshapes_only_unfilled_short_rungs(
        self,
        make_strategy,
        requirements,
        hold_bands,
        entry_order,
        make_account_state,
        make_position,
    ):
        strategy = make_strategy([PROFILE_THREE_RUNGS])
        asyncio.run(strategy.setup(requirements))
        hold_bands(strategy, PositionSide.SHORT, 1)
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_three_rung_ohlcvs(),
            make_account_state(
                holding=make_position(PositionSide.SHORT),
                open_orders=[
                    _fake_order(
                        "rung-2", entry_order, OrderSide.SELL, level=109.0, rung=2
                    ),
                    _fake_order(
                        "rung-3", entry_order, OrderSide.SELL, level=114.0, rung=3
                    ),
                ],
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        reshaped = {
            modification.order_id: _level(modification.order, entry_order)
            for modification in _modifications(bookkeeper)
        }
        assert reshaped == {"rung-2": 110.0, "rung-3": 115.0}
        assert _rung_orders(bookkeeper, entry_order) == []

    def test_a_long_position_keeps_rungs_two_and_three_when_the_short_side_has_one_band(
        self,
        make_strategy,
        requirements,
        hold_bands,
        entry_order,
        make_account_state,
        make_position,
    ):
        strategy = make_strategy([PROFILE_UNEQUAL_SIDES])
        asyncio.run(strategy.setup(requirements))
        hold_bands(strategy, PositionSide.LONG, 1)
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_three_rung_ohlcvs(),
            make_account_state(
                holding=make_position(PositionSide.LONG),
                open_orders=[
                    _fake_order(
                        "rung-2", entry_order, OrderSide.BUY, level=91.0, rung=2
                    ),
                    _fake_order(
                        "rung-3", entry_order, OrderSide.BUY, level=86.0, rung=3
                    ),
                ],
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        reshaped = {
            modification.order_id: _level(modification.order, entry_order)
            for modification in _modifications(bookkeeper)
        }
        assert reshaped == {"rung-2": 90.0, "rung-3": 85.0}
        assert _rung_orders(bookkeeper, entry_order) == []

    def test_a_short_position_keeps_rungs_two_and_three_when_the_long_side_has_one_band(
        self,
        make_strategy,
        requirements,
        hold_bands,
        entry_order,
        make_account_state,
        make_position,
    ):
        strategy = make_strategy([PROFILE_THREE_SHORT_ONE_LONG])
        asyncio.run(strategy.setup(requirements))
        hold_bands(strategy, PositionSide.SHORT, 1)
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_three_rung_ohlcvs(),
            make_account_state(
                holding=make_position(PositionSide.SHORT),
                open_orders=[
                    _fake_order(
                        "rung-2", entry_order, OrderSide.SELL, level=109.0, rung=2
                    ),
                    _fake_order(
                        "rung-3", entry_order, OrderSide.SELL, level=114.0, rung=3
                    ),
                ],
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        reshaped = {
            modification.order_id: _level(modification.order, entry_order)
            for modification in _modifications(bookkeeper)
        }
        assert reshaped == {"rung-2": 110.0, "rung-3": 115.0}
        assert _rung_orders(bookkeeper, entry_order) == []

    def test_nan_levels_cancel_only_rungs_and_skip_updates(
        self, strategy, hold_bands, entry_order, make_account_state, make_position
    ):
        hold_bands(strategy, PositionSide.LONG, 1)
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(
                reference=float("nan"),
                band_low_1=float("nan"),
                band_high_1=float("nan"),
            ),
            make_account_state(
                holding=make_position(PositionSide.LONG),
                open_orders=[
                    _fake_order(
                        "short-1", entry_order, OrderSide.SELL, level=104.0, rung=1
                    ),
                    _fake_order("sl-1", "stop-loss", OrderSide.SELL),
                ],
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        cancelled_ids = _cancelled_order_ids(bookkeeper)
        updates = [
            a
            for a in bookkeeper.list_actions()
            if isinstance(
                a, (UpdatePositionStopLossAction, UpdatePositionTakeProfitAction)
            )
        ]
        assert cancelled_ids == ["short-1"]
        assert updates == []
        assert _rung_orders(bookkeeper, entry_order) == []

    def test_does_not_emit_cancel_all_for_symbol(
        self, strategy, hold_bands, make_account_state, make_position
    ):
        hold_bands(strategy, PositionSide.LONG, 1)
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(),
            make_account_state(holding=make_position(PositionSide.LONG)),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        cancel_all = [
            a
            for a in bookkeeper.list_actions()
            if isinstance(a, CancelOrdersForSymbolAction)
        ]
        assert cancel_all == []


class TestCancelReplaceMode:
    @pytest.fixture
    def strategy(
        self,
        make_strategy: Callable[..., EnvelopeStrategy],
        requirements: StrategyRequirements,
    ) -> EnvelopeStrategy:
        strategy = make_strategy([PROFILE], update_mode="cancel_replace")
        asyncio.run(strategy.setup(requirements))
        return strategy

    def test_flat_book_cancels_all_orders_for_the_symbol(
        self, strategy, make_account_state
    ):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(), make_account_state(), TIMESTAMP, bookkeeper, [TIMEFRAME]
        )

        cancels = [
            a
            for a in bookkeeper.list_actions()
            if isinstance(a, CancelOrdersForSymbolAction)
        ]
        assert len(cancels) == 1
        assert cancels[0].symbol == BTC

    def test_resting_rungs_are_replaced_not_reshaped(
        self, strategy, entry_order, make_account_state
    ):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(),
            make_account_state(
                open_orders=[
                    _fake_order(
                        "long-1", entry_order, OrderSide.BUY, level=96.0, rung=1
                    ),
                ]
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        orders = _rung_orders(bookkeeper, entry_order)
        assert _modifications(bookkeeper) == []
        assert {o.side: _level(o, entry_order) for o in orders} == {
            OrderSide.BUY: 95.0,
            OrderSide.SELL: 105.0,
        }

    def test_open_position_cancels_pending_rungs_by_id(
        self, strategy, hold_bands, entry_order, make_account_state, make_position
    ):
        hold_bands(strategy, PositionSide.LONG, 1)
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(),
            make_account_state(
                holding=make_position(PositionSide.LONG),
                open_orders=[
                    _fake_order("rung-1", entry_order, OrderSide.BUY),
                    _fake_order("tp-1", "take-profit", OrderSide.SELL),
                    _fake_order("sl-1", "stop-loss", OrderSide.SELL),
                ],
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        cancelled_ids = _cancelled_order_ids(bookkeeper)
        assert cancelled_ids == ["rung-1"]

    def test_long_position_re_emits_only_unfilled_long_rungs(
        self,
        make_strategy,
        requirements,
        hold_bands,
        entry_order,
        make_account_state,
        make_position,
    ):
        strategy = make_strategy([PROFILE_THREE_RUNGS], update_mode="cancel_replace")
        asyncio.run(strategy.setup(requirements))
        hold_bands(strategy, PositionSide.LONG, 1)
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_three_rung_ohlcvs(),
            make_account_state(
                holding=make_position(PositionSide.LONG),
                open_orders=[
                    _fake_order("rung-2", entry_order, OrderSide.BUY),
                    _fake_order("rung-3", entry_order, OrderSide.BUY),
                ],
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        entry_orders = [
            o for o in _rung_orders(bookkeeper, entry_order) if not o.reduce_only
        ]
        assert len(entry_orders) == 2
        assert {o.side for o in entry_orders} == {OrderSide.BUY}
        assert {_level(o, entry_order) for o in entry_orders} == {90.0, 85.0}

    def test_open_position_does_not_emit_cancel_all_for_symbol(
        self, strategy, hold_bands, make_account_state, make_position
    ):
        hold_bands(strategy, PositionSide.LONG, 1)
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(),
            make_account_state(holding=make_position(PositionSide.LONG)),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        cancel_all = [
            a
            for a in bookkeeper.list_actions()
            if isinstance(a, CancelOrdersForSymbolAction)
        ]
        assert cancel_all == []


_LONG_BAND_STOP = 95.0 * (1 - 0.04)
_SHORT_BAND_STOP = 105.0 * (1 + 0.04)

_CLOSES_CROSSING_A_BAND_SHORT_OF_ITS_STOP_LOSS = [
    pytest.param(94.0, OrderSide.BUY, 95.0, id="long-close-below-band"),
    pytest.param(95.0, OrderSide.BUY, 95.0, id="long-close-at-band"),
    pytest.param(
        _LONG_BAND_STOP + 0.1, OrderSide.BUY, 95.0, id="long-close-above-its-stop-loss"
    ),
    pytest.param(106.0, OrderSide.SELL, 105.0, id="short-close-above-band"),
    pytest.param(105.0, OrderSide.SELL, 105.0, id="short-close-at-band"),
    pytest.param(
        _SHORT_BAND_STOP - 0.1,
        OrderSide.SELL,
        105.0,
        id="short-close-below-its-stop-loss",
    ),
]
_PAST_STOP_LOSS_CLOSES = [
    pytest.param(_LONG_BAND_STOP, OrderSide.BUY, id="long-close-at-its-stop-loss"),
    pytest.param(
        _LONG_BAND_STOP - 1.0, OrderSide.BUY, id="long-close-below-its-stop-loss"
    ),
    pytest.param(_SHORT_BAND_STOP, OrderSide.SELL, id="short-close-at-its-stop-loss"),
    pytest.param(
        _SHORT_BAND_STOP + 1.0, OrderSide.SELL, id="short-close-above-its-stop-loss"
    ),
]

_HELD_WITH_RUNG_TWO_SKIPPED = [
    pytest.param(
        PositionSide.LONG,
        OrderSide.BUY,
        61.0,
        {"rung-2": 91.0, "rung-3": 86.0},
        85.0,
        id="long",
    ),
    pytest.param(
        PositionSide.SHORT,
        OrderSide.SELL,
        145.0,
        {"rung-2": 109.0, "rung-3": 114.0},
        115.0,
        id="short",
    ),
]


def _resting_rungs(
    entry_order: str, order_side: OrderSide, resting_levels: dict[str, float]
) -> list[OrderProtocol]:
    return [
        _fake_order(order_id, entry_order, order_side, level=level, rung=rung)
        for rung, (order_id, level) in enumerate(resting_levels.items(), start=2)
    ]


@pytest.fixture
def book_while_holding(
    make_strategy: Callable[..., EnvelopeStrategy],
    requirements: StrategyRequirements,
    hold_bands: Callable[..., None],
    make_position: Callable[..., PositionSnapshot],
    make_account_state: Callable[..., AccountSnapshots],
) -> Callable[..., BookKeeper]:
    def _book(
        update_mode: str,
        held_side: PositionSide,
        close: float,
        open_orders: list[OrderProtocol],
    ) -> BookKeeper:
        strategy = make_strategy([PROFILE_THREE_RUNGS], update_mode=update_mode)
        asyncio.run(strategy.setup(requirements))
        hold_bands(strategy, held_side, 1)
        bookkeeper = BookKeeper()
        strategy.book_trading_actions(
            _make_three_rung_ohlcvs(close=close),
            make_account_state(
                holding=make_position(held_side), open_orders=open_orders
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )
        return bookkeeper

    return _book


class TestRungsPastTheirStopLoss:
    @pytest.mark.parametrize(
        ("close", "order_side", "level"), _CLOSES_CROSSING_A_BAND_SHORT_OF_ITS_STOP_LOSS
    )
    def test_a_crossed_band_the_stop_loss_is_not_reached_takes_a_rung(
        self, strategy, entry_order, close, order_side, level, make_account_state
    ):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(close=close),
            make_account_state(),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        orders = _rung_orders(bookkeeper, entry_order)
        assert {o.side: _level(o, entry_order) for o in orders}[order_side] == level

    @pytest.mark.parametrize(("close", "order_side"), _PAST_STOP_LOSS_CLOSES)
    def test_a_band_whose_stop_loss_the_close_reached_takes_no_rung(
        self, strategy, entry_order, close, order_side, make_account_state
    ):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(close=close),
            make_account_state(),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        orders = _rung_orders(bookkeeper, entry_order)
        assert order_side not in {o.side for o in orders}

    @pytest.mark.parametrize(("close", "order_side"), _PAST_STOP_LOSS_CLOSES)
    def test_a_rung_skipped_for_its_stop_loss_is_booked_once_the_close_is_back(
        self, strategy, entry_order, close, order_side, make_account_state
    ):
        skipped_bookkeeper = BookKeeper()
        booked_bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(close=close),
            make_account_state(),
            TIMESTAMP,
            skipped_bookkeeper,
            [TIMEFRAME],
        )
        strategy.book_trading_actions(
            _make_ohlcvs(close=100.0),
            make_account_state(),
            TIMESTAMP,
            booked_bookkeeper,
            [TIMEFRAME],
        )

        skipped = _rung_orders(skipped_bookkeeper, entry_order)
        booked = _rung_orders(booked_bookkeeper, entry_order)
        assert order_side not in {o.side for o in skipped}
        assert {o.side: _level(o, entry_order) for o in booked} == {
            OrderSide.BUY: 95.0,
            OrderSide.SELL: 105.0,
        }

    def test_a_rung_resting_on_a_band_past_its_stop_loss_is_cancelled_while_flat(
        self, strategy, entry_order, make_account_state
    ):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(close=_LONG_BAND_STOP),
            make_account_state(
                open_orders=[
                    _fake_order(
                        "long-1", entry_order, OrderSide.BUY, level=96.0, rung=1
                    )
                ]
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        assert _cancelled_order_ids(bookkeeper) == ["long-1"]
        assert _modifications(bookkeeper) == []

    @pytest.mark.parametrize(
        ("held_side", "order_side", "close", "resting_levels", "reshaped_level"),
        _HELD_WITH_RUNG_TWO_SKIPPED,
    )
    def test_a_held_positions_rung_on_a_skipped_band_is_cancelled(
        self,
        book_while_holding,
        entry_order,
        held_side,
        order_side,
        close,
        resting_levels,
        reshaped_level,
    ):
        bookkeeper = book_while_holding(
            "modify",
            held_side,
            close,
            _resting_rungs(entry_order, order_side, resting_levels),
        )

        reshaped = {
            modification.order_id: _level(modification.order, entry_order)
            for modification in _modifications(bookkeeper)
        }
        assert _cancelled_order_ids(bookkeeper) == ["rung-2"]
        assert reshaped == {"rung-3": reshaped_level}
        assert _rung_orders(bookkeeper, entry_order) == []

    @pytest.mark.parametrize(
        ("held_side", "order_side", "close", "resting_levels", "placed_level"),
        _HELD_WITH_RUNG_TWO_SKIPPED,
    )
    def test_a_held_positions_rung_on_a_skipped_band_is_cancelled_when_replacing(
        self,
        book_while_holding,
        entry_order,
        held_side,
        order_side,
        close,
        resting_levels,
        placed_level,
    ):
        bookkeeper = book_while_holding(
            "cancel_replace",
            held_side,
            close,
            _resting_rungs(entry_order, order_side, resting_levels),
        )

        entry_orders = [
            o for o in _rung_orders(bookkeeper, entry_order) if not o.reduce_only
        ]
        assert _cancelled_order_ids(bookkeeper) == ["rung-2", "rung-3"]
        assert [_level(o, entry_order) for o in entry_orders] == [placed_level]

    @pytest.mark.parametrize(
        ("update_mode", "cancelled_ids", "reshaped", "placed"),
        [
            pytest.param(
                "modify",
                ["short-2"],
                {"rung-2": 90.0, "rung-3": 85.0},
                [],
                id="modify",
            ),
            pytest.param(
                "cancel_replace",
                ["rung-2", "rung-3", "short-2"],
                {},
                [90.0, 85.0],
                id="cancel-replace",
            ),
        ],
    )
    def test_an_opposite_side_rung_is_cancelled_while_holding(
        self,
        book_while_holding,
        entry_order,
        update_mode,
        cancelled_ids,
        reshaped,
        placed,
    ):
        bookkeeper = book_while_holding(
            update_mode,
            PositionSide.LONG,
            100.0,
            [
                *_resting_rungs(
                    entry_order, OrderSide.BUY, {"rung-2": 91.0, "rung-3": 86.0}
                ),
                _fake_order(
                    "short-2", entry_order, OrderSide.SELL, level=110.0, rung=2
                ),
            ],
        )

        assert _cancelled_order_ids(bookkeeper) == cancelled_ids
        assert _reshaped_levels(bookkeeper, entry_order) == reshaped
        assert _placed_levels(bookkeeper, entry_order) == placed

    @pytest.mark.parametrize(
        ("update_mode", "cancelled_rungs", "reshaped", "placed"),
        [
            pytest.param("modify", [], {"rung-2": 90.0}, [85.0], id="modify"),
            pytest.param(
                "cancel_replace", ["rung-2"], {}, [90.0, 85.0], id="cancel-replace"
            ),
        ],
    )
    @pytest.mark.parametrize(
        "foreign_rung",
        [
            pytest.param(None, id="untagged"),
            pytest.param(0, id="band-zero"),
            pytest.param(4, id="band-beyond-the-ladder"),
        ],
    )
    def test_a_rung_on_no_band_of_the_ladder_is_cancelled_while_holding(
        self,
        book_while_holding,
        entry_order,
        foreign_rung,
        update_mode,
        cancelled_rungs,
        reshaped,
        placed,
    ):
        bookkeeper = book_while_holding(
            update_mode,
            PositionSide.LONG,
            100.0,
            [
                *_resting_rungs(entry_order, OrderSide.BUY, {"rung-2": 91.0}),
                _fake_order(
                    "foreign", entry_order, OrderSide.BUY, level=91.0, rung=foreign_rung
                ),
            ],
        )

        assert _cancelled_order_ids(bookkeeper) == sorted(["foreign", *cancelled_rungs])
        assert _reshaped_levels(bookkeeper, entry_order) == reshaped
        assert _placed_levels(bookkeeper, entry_order) == placed

    def test_a_rung_resting_on_a_filled_band_is_cancelled(
        self, book_while_holding, entry_order
    ):
        bookkeeper = book_while_holding(
            "modify",
            PositionSide.LONG,
            100.0,
            [
                _fake_order("rung-1", entry_order, OrderSide.BUY, level=96.0, rung=1),
                _fake_order("rung-2", entry_order, OrderSide.BUY, level=91.0, rung=2),
            ],
        )

        reshaped = {
            modification.order_id: _level(modification.order, entry_order)
            for modification in _modifications(bookkeeper)
        }
        assert _cancelled_order_ids(bookkeeper) == ["rung-1"]
        assert reshaped == {"rung-2": 90.0}

    def test_a_rung_resting_on_a_filled_band_is_cancelled_when_replacing(
        self, book_while_holding, entry_order
    ):
        bookkeeper = book_while_holding(
            "cancel_replace",
            PositionSide.LONG,
            100.0,
            [
                _fake_order("rung-1", entry_order, OrderSide.BUY, level=96.0, rung=1),
                _fake_order("rung-2", entry_order, OrderSide.BUY, level=91.0, rung=2),
            ],
        )

        assert _cancelled_order_ids(bookkeeper) == ["rung-1", "rung-2"]
        assert _placed_levels(bookkeeper, entry_order) == [90.0, 85.0]


class TestBandsHeldByTheTracker:
    @pytest.fixture
    def strategy(
        self,
        make_strategy: Callable[..., EnvelopeStrategy],
        requirements: StrategyRequirements,
    ) -> EnvelopeStrategy:
        strategy = make_strategy([PROFILE_THREE_RUNGS])
        asyncio.run(strategy.setup(requirements))
        return strategy

    @pytest.fixture
    def wanted_levels(
        self,
        make_position: Callable[..., PositionSnapshot],
        make_account_state: Callable[..., AccountSnapshots],
    ) -> Callable[..., list[float | None]]:
        def _wanted(
            strategy: EnvelopeStrategy,
            entry_order: str,
            close: float,
            held_side: PositionSide = PositionSide.LONG,
        ) -> list[float | None]:
            bookkeeper = BookKeeper()
            strategy.book_trading_actions(
                _make_three_rung_ohlcvs(close=close),
                make_account_state(holding=make_position(held_side)),
                TIMESTAMP,
                bookkeeper,
                [TIMEFRAME],
            )
            return [
                _level(o, entry_order) for o in _rung_orders(bookkeeper, entry_order)
            ]

        return _wanted

    def test_band_three_filled_after_bands_one_and_two_were_skipped(
        self, strategy, hold_bands, wanted_levels, entry_order
    ):
        hold_bands(strategy, PositionSide.LONG, 3)

        wanted = wanted_levels(strategy, entry_order, close=100.0)

        assert wanted == [95.0, 90.0]

    def test_band_three_filled_while_the_close_is_below_the_stop_loss_of_bands_one_and_two(
        self, strategy, hold_bands, wanted_levels, entry_order
    ):
        hold_bands(strategy, PositionSide.LONG, 3)

        wanted = wanted_levels(strategy, entry_order, close=61.0)

        assert wanted == []

    def test_band_one_filled_then_band_two(
        self, strategy, hold_bands, wanted_levels, entry_order
    ):
        hold_bands(strategy, PositionSide.LONG, 1, 2)

        wanted = wanted_levels(strategy, entry_order, close=100.0)

        assert wanted == [85.0]

    def test_band_one_filled_before_the_candle_window_and_band_two_within_it(
        self,
        strategy,
        fold_fills,
        make_band_fill,
        make_position,
        wanted_levels,
        entry_order,
    ):
        profile = strategy.profiles[0]
        window_start = TIMESTAMP + timedelta(minutes=1, seconds=30)
        fills = [make_band_fill(profile, PositionSide.LONG, index) for index in (1, 2)]

        fold_fills(
            fills, make_position(PositionSide.LONG, 2.0), window_start=window_start
        )
        wanted = wanted_levels(strategy, entry_order, close=100.0)

        assert [fill.timestamp < window_start for fill in fills] == [True, False]
        assert wanted == [85.0]


def _long_sl_fill(timestamp: datetime) -> Execution:
    return Execution(
        execution_id="sl-long",
        order_id="sl-long",
        symbol=BTC,
        side=OrderSide.SELL,
        price=100.0,
        quantity=1.0,
        timestamp=timestamp,
        kind="stop-loss",
    )


def _short_sl_fill(timestamp: datetime) -> Execution:
    return Execution(
        execution_id="sl-short",
        order_id="sl-short",
        symbol=BTC,
        side=OrderSide.BUY,
        price=100.0,
        quantity=1.0,
        timestamp=timestamp,
        kind="stop-loss",
    )


class TestReentryGating:
    def test_long_locked_suppresses_long_rungs(
        self, strategy, entry_order, make_account_state
    ):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(close=95.0, reference=100.0),
            make_account_state(executions=[_long_sl_fill(TIMESTAMP)]),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        orders = _rung_orders(bookkeeper, entry_order)
        assert len(orders) == 1
        assert orders[0].side == OrderSide.SELL

    def test_short_locked_suppresses_short_rungs(
        self, strategy, entry_order, make_account_state
    ):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(close=105.0, reference=100.0),
            make_account_state(executions=[_short_sl_fill(TIMESTAMP)]),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        orders = _rung_orders(bookkeeper, entry_order)
        assert len(orders) == 1
        assert orders[0].side == OrderSide.BUY

    def test_both_locked_suppresses_all_rungs(
        self, strategy, entry_order, make_account_state
    ):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(close=100.0, reference=100.0),
            make_account_state(
                executions=[_long_sl_fill(TIMESTAMP), _short_sl_fill(TIMESTAMP)]
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        assert _rung_orders(bookkeeper, entry_order) == []

    def test_open_position_with_own_side_locked_skips_re_emission(
        self, make_strategy, entry_order, make_account_state, make_position
    ):
        strategy = make_strategy([PROFILE_THREE_RUNGS])
        asyncio.run(strategy.setup(StrategyRequirements()))
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_three_rung_ohlcvs(reference=100.0),
            make_account_state(
                holding=make_position(PositionSide.LONG),
                open_orders=[
                    _fake_order(
                        "rung-2", entry_order, OrderSide.BUY, level=91.0, rung=2
                    ),
                    _fake_order(
                        "rung-3", entry_order, OrderSide.BUY, level=86.0, rung=3
                    ),
                ],
                executions=[_long_sl_fill(TIMESTAMP)],
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        entry_orders = [
            o for o in _rung_orders(bookkeeper, entry_order) if not o.reduce_only
        ]
        assert entry_orders == []
        assert _modifications(bookkeeper) == []

    def test_open_position_with_own_side_locked_still_updates_sl_tp(
        self, make_strategy, make_account_state, make_position
    ):
        strategy = make_strategy([PROFILE_THREE_RUNGS])
        asyncio.run(strategy.setup(StrategyRequirements()))
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_three_rung_ohlcvs(reference=100.0),
            make_account_state(
                holding=make_position(PositionSide.LONG),
                executions=[_long_sl_fill(TIMESTAMP)],
            ),
            TIMESTAMP,
            bookkeeper,
            [TIMEFRAME],
        )

        sl_updates = [
            a
            for a in bookkeeper.list_actions()
            if isinstance(a, UpdatePositionStopLossAction)
        ]
        tp_updates = [
            a
            for a in bookkeeper.list_actions()
            if isinstance(a, UpdatePositionTakeProfitAction)
        ]
        assert len(sl_updates) == 1
        assert len(tp_updates) == 1


class TestFillReasons:
    @staticmethod
    def _declared_describer(
        strategy: EnvelopeStrategy,
    ) -> Callable[[Execution], str | None]:
        requirements = StrategyRequirements()
        asyncio.run(strategy.setup(requirements))
        describer = requirements.account._get_all()[0].describe_fills
        assert describer is not None
        return describer

    def test_names_the_rung_its_tag_claims(
        self, make_strategy, entry_order, make_entry_fill
    ):
        strategy = make_strategy([PROFILE_THREE_RUNGS])
        fill = make_entry_fill(
            rung_client_order_id(TIMEFRAME, OrderSide.BUY, 2), kind=entry_order
        )

        reason = self._declared_describer(strategy)(fill)

        assert reason == "entry rung 2 of 3"

    def test_counts_the_fills_own_side_when_sides_are_unequal(
        self, make_strategy, entry_order, make_entry_fill
    ):
        strategy = make_strategy([PROFILE_UNEQUAL_SIDES])
        long_fill = make_entry_fill(
            rung_client_order_id(TIMEFRAME, OrderSide.BUY, 2),
            side=OrderSide.BUY,
            kind=entry_order,
        )
        short_fill = make_entry_fill(
            rung_client_order_id(TIMEFRAME, OrderSide.SELL, 1),
            side=OrderSide.SELL,
            kind=entry_order,
        )
        describer = self._declared_describer(strategy)

        assert describer(long_fill) == "entry rung 2 of 3"
        assert describer(short_fill) == "entry rung 1 of 1"

    def test_fill_under_the_tag_of_the_other_side(
        self, strategy, entry_order, make_entry_fill
    ):
        fill = make_entry_fill(
            rung_client_order_id(TIMEFRAME, OrderSide.SELL, 1),
            side=OrderSide.BUY,
            kind=entry_order,
        )

        reason = self._declared_describer(strategy)(fill)

        assert reason is None

    def test_fill_of_another_program(self, strategy, entry_order, make_entry_fill):
        fill = make_entry_fill("t1785908301620", kind=entry_order)

        reason = self._declared_describer(strategy)(fill)

        assert reason is None

    def test_fill_without_a_client_order_id(
        self, strategy, entry_order, make_entry_fill
    ):
        fill = make_entry_fill(None, kind=entry_order)

        reason = self._declared_describer(strategy)(fill)

        assert reason is None

    def test_take_profit_means_the_moving_average_was_reached(self, strategy):
        fill = _venue_fired_exit("take-profit")

        reason = self._declared_describer(strategy)(fill)

        assert reason == "moving average reached"

    def test_stop_loss_is_left_to_its_own_name(self, strategy):
        fill = _venue_fired_exit("stop-loss")

        reason = self._declared_describer(strategy)(fill)

        assert reason is None

    def test_fill_of_the_other_entry_order(
        self, strategy, entry_order, make_entry_fill
    ):
        fill = make_entry_fill(
            rung_client_order_id(TIMEFRAME, OrderSide.BUY, 1),
            kind=_OTHER_ENTRY_ORDER[entry_order],
        )

        reason = self._declared_describer(strategy)(fill)

        assert reason is None


def _venue_fired_exit(kind: OrderType) -> Execution:
    return Execution(
        execution_id="fired-2",
        order_id="fired-2",
        symbol=BTC,
        side=OrderSide.SELL,
        price=105.0,
        quantity=0.5,
        timestamp=TIMESTAMP + timedelta(hours=6),
        kind=kind,
    )
