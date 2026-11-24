import asyncio
from datetime import datetime
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from impulse_helpers import write_signal_columns

from robottraderslab import Symbol
from robottraderslab.strategies import (
    AccountSnapshot,
    AccountSnapshots,
    Balance,
    BookKeeper,
    OHLCVs,
    OrderProtocol,
    OrderSide,
    PositionSide,
    StrategyRequirements,
    TrackedPosition,
    TrackingId,
    tag_of,
)
from robottraderslab.strategies.futures import (
    CancelOrderByIdAction,
    FuturesMarketOrderAction,
    MarginMode,
    MarginSettings,
    PositionSnapshot,
    SetLeverageAction,
    SetMarginModeAction,
    UpdatePositionStopLossAction,
)
from robottraderslab_impulse import ImpulseStrategy

BTC = Symbol.create("BTC/USDT:USDT")
ACCOUNT_NAME = "test"
TIMEFRAME = "1d"
TIMESTAMP = datetime(2024, 2, 19)
SETUP_ID = "BTC/USDT:USDT@1d"


def _open_position(
    side: PositionSide,
    quantity: float,
    average_entry_price: float,
    symbol: Symbol = BTC,
) -> PositionSnapshot:
    return PositionSnapshot(
        symbol=symbol,
        side=side,
        quantity=quantity,
        average_entry_price=average_entry_price,
        entry_time=TIMESTAMP,
        leverage=5.0,
        liquidation_price=0.0,
        stop_loss_price=None,
        take_profit_price=None,
    )


def _account_state(
    *,
    positions: dict[Symbol, PositionSnapshot] | None = None,
    open_orders: list[OrderProtocol] | None = None,
    symbols: tuple[Symbol, ...] = (BTC,),
) -> AccountSnapshot:
    snapshot = AccountSnapshot(
        account_name=ACCOUNT_NAME,
        balances={"USDT": Balance(locked=0.0, total=10_000.0)},
        positions=positions or {},
        open_orders=open_orders or [],
        margin_settings={
            symbol: MarginSettings(leverage=None, margin_mode=MarginMode.CROSS)
            for symbol in symbols
        },
        conversion_rates={symbol: 1.0 for symbol in symbols},
    )
    return AccountSnapshots({ACCOUNT_NAME: snapshot})


def _stand_on_first_candle(ohlcvs: OHLCVs) -> None:
    """Set the iteration cursor so `current` and `signal` read the first candle."""
    next(ohlcvs._iter_timeframes())


def _make_ohlcvs(
    *,
    close: float = 100.0,
    long_entry: bool = False,
    long_exit: bool = False,
    short_entry: bool = False,
    short_exit: bool = False,
    profile_id: str = SETUP_ID,
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

    ohlcvs.add_column(BTC, TIMEFRAME, f"{profile_id}_long_entry", [float(long_entry)])
    ohlcvs.add_column(BTC, TIMEFRAME, f"{profile_id}_long_exit", [float(long_exit)])
    ohlcvs.add_column(BTC, TIMEFRAME, f"{profile_id}_short_entry", [float(short_entry)])
    ohlcvs.add_column(BTC, TIMEFRAME, f"{profile_id}_short_exit", [float(short_exit)])

    _stand_on_first_candle(ohlcvs)

    return ohlcvs


def _make_nan_ohlcvs(profile_id: str = SETUP_ID) -> OHLCVs:
    dates = pd.date_range("2024-02-19", periods=1, freq="1D")
    df = pd.DataFrame(
        {
            "open": [np.nan],
            "high": [np.nan],
            "low": [np.nan],
            "close": [np.nan],
            "volume": [0.0],
        },
        index=dates,
    )
    ohlcvs = OHLCVs({TIMEFRAME: {BTC: df}})

    ohlcvs.add_column(BTC, TIMEFRAME, f"{profile_id}_long_entry", [0.0])
    ohlcvs.add_column(BTC, TIMEFRAME, f"{profile_id}_long_exit", [0.0])
    ohlcvs.add_column(BTC, TIMEFRAME, f"{profile_id}_short_entry", [0.0])
    ohlcvs.add_column(BTC, TIMEFRAME, f"{profile_id}_short_exit", [0.0])

    _stand_on_first_candle(ohlcvs)

    return ohlcvs


PROFILE = {
    "symbol": "BTC/USDT:USDT",
    "timeframe": TIMEFRAME,
    "trix_length": 9,
    "signal_length": 21,
    "trend_length": 200,
    "total_balance_ratio": 0.5,
    "leverage": 5.0,
    "margin_mode": "isolated",
}


@pytest.fixture
def strategy(make_strategy) -> ImpulseStrategy:
    return _loaded_strategy(make_strategy, [PROFILE])


class TestMarginTargets:
    def test_a_drifted_symbol_books_no_margin_action(self, strategy):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(long_entry=True),
            _account_state(),
            TIMESTAMP,
            bookkeeper,
            {TIMEFRAME},
        )

        setters = [
            action
            for action in bookkeeper.list_actions()
            if isinstance(action, (SetLeverageAction, SetMarginModeAction))
        ]
        assert setters == []


class TestEntrySignals:
    def test_long_entry_signal(self, strategy):
        bookkeeper = BookKeeper()
        ohlcvs = _make_ohlcvs(long_entry=True, close=100.0)
        state = _account_state()

        strategy.book_trading_actions(ohlcvs, state, TIMESTAMP, bookkeeper, {TIMEFRAME})

        market_orders = [
            a
            for a in bookkeeper.list_actions()
            if isinstance(a, FuturesMarketOrderAction)
        ]
        assert len(market_orders) == 1
        assert market_orders[0].side == OrderSide.BUY
        assert market_orders[0].reduce_only is False

    def test_short_entry_signal(self, strategy):
        bookkeeper = BookKeeper()
        ohlcvs = _make_ohlcvs(short_entry=True, close=100.0)
        state = _account_state()

        strategy.book_trading_actions(ohlcvs, state, TIMESTAMP, bookkeeper, {TIMEFRAME})

        market_orders = [
            a
            for a in bookkeeper.list_actions()
            if isinstance(a, FuturesMarketOrderAction)
        ]
        assert len(market_orders) == 1
        assert market_orders[0].side == OrderSide.SELL
        assert market_orders[0].reduce_only is False

    def test_no_signal_produces_no_order(self, strategy):
        bookkeeper = BookKeeper()
        ohlcvs = _make_ohlcvs()
        state = _account_state()

        strategy.book_trading_actions(ohlcvs, state, TIMESTAMP, bookkeeper, {TIMEFRAME})

        market_orders = [
            a
            for a in bookkeeper.list_actions()
            if isinstance(a, FuturesMarketOrderAction)
        ]
        assert len(market_orders) == 0

    def test_nan_close_skips_profile(self, strategy):
        bookkeeper = BookKeeper()
        ohlcvs = _make_nan_ohlcvs()
        state = _account_state()

        strategy.book_trading_actions(ohlcvs, state, TIMESTAMP, bookkeeper, {TIMEFRAME})

        market_orders = [
            a
            for a in bookkeeper.list_actions()
            if isinstance(a, FuturesMarketOrderAction)
        ]
        assert len(market_orders) == 0

    def test_untriggered_timeframe_skips_profile(self, strategy):
        bookkeeper = BookKeeper()
        ohlcvs = _make_ohlcvs(long_entry=True)
        state = _account_state()

        strategy.book_trading_actions(ohlcvs, state, TIMESTAMP, bookkeeper, {"4h"})

        market_orders = [
            a
            for a in bookkeeper.list_actions()
            if isinstance(a, FuturesMarketOrderAction)
        ]
        assert len(market_orders) == 0


class TestExitSignals:
    def test_a_long_exit_signal_on_a_held_long(self, strategy):
        _hold_on_every_profile(strategy, PositionSide.LONG, 0.5)
        exit_state = _account_state(
            positions={BTC: _open_position(PositionSide.LONG, 0.5, 100.0)}
        )

        exit_ohlcvs = _make_ohlcvs(long_exit=True, close=110.0)
        exit_bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            exit_ohlcvs, exit_state, TIMESTAMP, exit_bookkeeper, {TIMEFRAME}
        )

        exit_orders = [
            a
            for a in exit_bookkeeper.list_actions()
            if isinstance(a, FuturesMarketOrderAction)
        ]
        assert len(exit_orders) == 1
        assert exit_orders[0].side == OrderSide.SELL
        assert exit_orders[0].reduce_only is False
        assert exit_orders[0].quantity == 0.5
        assert exit_orders[0].reason == "impulse long exit"

    def test_a_short_exit_signal_on_a_held_short(self, strategy):
        _hold_on_every_profile(strategy, PositionSide.SHORT, 0.5)
        exit_state = _account_state(
            positions={BTC: _open_position(PositionSide.SHORT, 0.5, 100.0)}
        )

        exit_ohlcvs = _make_ohlcvs(short_exit=True, close=90.0)
        exit_bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            exit_ohlcvs, exit_state, TIMESTAMP, exit_bookkeeper, {TIMEFRAME}
        )

        exit_orders = [
            a
            for a in exit_bookkeeper.list_actions()
            if isinstance(a, FuturesMarketOrderAction)
        ]
        assert len(exit_orders) == 1
        assert exit_orders[0].side == OrderSide.BUY
        assert exit_orders[0].reduce_only is False
        assert exit_orders[0].reason == "impulse short exit"

    def test_wrong_exit_signal_does_not_close(self, strategy):
        _hold_on_every_profile(strategy, PositionSide.LONG, 0.5)
        exit_state = _account_state(
            positions={BTC: _open_position(PositionSide.LONG, 0.5, 100.0)}
        )

        exit_ohlcvs = _make_ohlcvs(short_exit=True, close=110.0)
        exit_bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            exit_ohlcvs, exit_state, TIMESTAMP, exit_bookkeeper, {TIMEFRAME}
        )

        exit_orders = [
            a
            for a in exit_bookkeeper.list_actions()
            if isinstance(a, FuturesMarketOrderAction)
        ]
        assert len(exit_orders) == 0


class TestDirectionFlags:
    def test_long_only_blocks_short_entry(self, make_strategy):
        profile = {**PROFILE, "long_only": True}
        strategy = make_strategy([profile])
        asyncio.run(strategy.setup(StrategyRequirements()))
        bookkeeper = BookKeeper()
        ohlcvs = _make_ohlcvs(short_entry=True, close=100.0)
        state = _account_state()

        strategy.book_trading_actions(ohlcvs, state, TIMESTAMP, bookkeeper, {TIMEFRAME})

        market_orders = [
            a
            for a in bookkeeper.list_actions()
            if isinstance(a, FuturesMarketOrderAction)
        ]
        assert len(market_orders) == 0

    def test_short_only_blocks_long_entry(self, make_strategy):
        profile = {**PROFILE, "short_only": True}
        strategy = make_strategy([profile])
        asyncio.run(strategy.setup(StrategyRequirements()))
        bookkeeper = BookKeeper()
        ohlcvs = _make_ohlcvs(long_entry=True, close=100.0)
        state = _account_state()

        strategy.book_trading_actions(ohlcvs, state, TIMESTAMP, bookkeeper, {TIMEFRAME})

        market_orders = [
            a
            for a in bookkeeper.list_actions()
            if isinstance(a, FuturesMarketOrderAction)
        ]
        assert len(market_orders) == 0

    def test_long_only_allows_long_entry(self, make_strategy):
        profile = {**PROFILE, "long_only": True}
        strategy = make_strategy([profile])
        asyncio.run(strategy.setup(StrategyRequirements()))
        bookkeeper = BookKeeper()
        ohlcvs = _make_ohlcvs(long_entry=True, close=100.0)
        state = _account_state()

        strategy.book_trading_actions(ohlcvs, state, TIMESTAMP, bookkeeper, {TIMEFRAME})

        market_orders = [
            a
            for a in bookkeeper.list_actions()
            if isinstance(a, FuturesMarketOrderAction)
        ]
        assert len(market_orders) == 1
        assert market_orders[0].side == OrderSide.BUY


def _sl_profile(stop_loss_pct: float | None = 0.1, tag: str = "") -> dict:
    profile = {**PROFILE, "stop_loss_pct": stop_loss_pct}
    if tag:
        profile["tag"] = tag
    return profile


def _loaded_strategy(make_strategy, profiles: list[dict]) -> ImpulseStrategy:
    strategy = make_strategy(profiles)
    asyncio.run(strategy.setup(StrategyRequirements()))
    strategy._position_tracker = _HeldPositions()
    return strategy


def _flat_ohlcvs(profile_ids: list[str], close: float = 100.0) -> OHLCVs:
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
    for profile_id in profile_ids:
        write_signal_columns(ohlcvs, BTC, TIMEFRAME, profile_id, {})
    _stand_on_first_candle(ohlcvs)
    return ohlcvs


def _long_exit_ohlcvs(
    profile_ids: list[str], exiting_profile_ids: set[str], close: float = 100.0
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
    for profile_id in profile_ids:
        exiting = 1.0 if profile_id in exiting_profile_ids else 0.0
        ohlcvs.add_column(BTC, TIMEFRAME, f"{profile_id}_long_entry", [0.0])
        ohlcvs.add_column(BTC, TIMEFRAME, f"{profile_id}_long_exit", [exiting])
        ohlcvs.add_column(BTC, TIMEFRAME, f"{profile_id}_short_entry", [0.0])
        ohlcvs.add_column(BTC, TIMEFRAME, f"{profile_id}_short_exit", [0.0])
    _stand_on_first_candle(ohlcvs)
    return ohlcvs


def _signalled_ohlcvs(
    signals_by_profile_id: dict[str, str | None], close: float = 100.0
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
    for profile_id, raised in signals_by_profile_id.items():
        write_signal_columns(
            ohlcvs, BTC, TIMEFRAME, profile_id, {} if raised is None else {0: raised}
        )
    _stand_on_first_candle(ohlcvs)
    return ohlcvs


class _HeldPositions:
    """A tracker holding what a test says each profile holds."""

    def __init__(self) -> None:
        self._held: dict[TrackingId, TrackedPosition] = {}

    def hold(self, tracking_id: TrackingId, held: TrackedPosition) -> None:
        self._held[tracking_id] = held

    def get(self, tracking_id: TrackingId) -> TrackedPosition | None:
        return self._held.get(tracking_id)


def _hold(
    strategy: ImpulseStrategy,
    tracking_id: TrackingId,
    side: PositionSide,
    quantity: float,
) -> None:
    strategy._position_tracker.hold(tracking_id, TrackedPosition(side, quantity))


def _mark_long_open(strategy: ImpulseStrategy, quantity: float = 1.0) -> None:
    _hold_on_every_profile(strategy, PositionSide.LONG, quantity)


def _hold_on_every_profile(
    strategy: ImpulseStrategy, side: PositionSide, quantity: float
) -> None:
    for profile in strategy.profiles:
        _hold(strategy, profile.profile_id, side, quantity)


def _resting_stop(order_id: str = "sl-1", symbol: Symbol = BTC) -> Mock:
    order = Mock()
    order.order_id = order_id
    order.kind = "stop-loss"
    order.symbol = symbol
    return order


def _stop_loss_updates(bookkeeper: BookKeeper) -> list[UpdatePositionStopLossAction]:
    return [
        action
        for action in bookkeeper.list_actions()
        if isinstance(action, UpdatePositionStopLossAction)
    ]


def _cancel_orders(bookkeeper: BookKeeper) -> list[CancelOrderByIdAction]:
    return [
        action
        for action in bookkeeper.list_actions()
        if isinstance(action, CancelOrderByIdAction)
    ]


def _entry_orders(bookkeeper: BookKeeper) -> list[FuturesMarketOrderAction]:
    return [
        action
        for action in bookkeeper.list_actions()
        if isinstance(action, FuturesMarketOrderAction)
    ]


class TestEntryStopLoss:
    def test_long_entry_attaches_stop_loss_below_close(self, make_strategy):
        strategy = _loaded_strategy(make_strategy, [_sl_profile(stop_loss_pct=0.1)])
        bookkeeper = BookKeeper()
        ohlcvs = _make_ohlcvs(long_entry=True, close=100.0)
        state = _account_state()

        strategy.book_trading_actions(ohlcvs, state, TIMESTAMP, bookkeeper, {TIMEFRAME})

        entry = _entry_orders(bookkeeper)[0]
        assert entry.stop_loss.trigger_price == pytest.approx(90.0)

    def test_short_entry_attaches_stop_loss_above_close(self, make_strategy):
        strategy = _loaded_strategy(make_strategy, [_sl_profile(stop_loss_pct=0.1)])
        bookkeeper = BookKeeper()
        ohlcvs = _make_ohlcvs(short_entry=True, close=100.0)
        state = _account_state()

        strategy.book_trading_actions(ohlcvs, state, TIMESTAMP, bookkeeper, {TIMEFRAME})

        entry = _entry_orders(bookkeeper)[0]
        assert entry.stop_loss.trigger_price == pytest.approx(110.0)

    def test_entry_without_stop_loss_pct_has_no_bracket(self, make_strategy):
        strategy = _loaded_strategy(make_strategy, [_sl_profile(stop_loss_pct=None)])
        bookkeeper = BookKeeper()
        ohlcvs = _make_ohlcvs(long_entry=True, close=100.0)
        state = _account_state()

        strategy.book_trading_actions(ohlcvs, state, TIMESTAMP, bookkeeper, {TIMEFRAME})

        entry = _entry_orders(bookkeeper)[0]
        assert entry.stop_loss is None


class TestNetStopLoss:
    def test_updates_stop_loss_from_average_entry_for_long(self, make_strategy):
        strategy = _loaded_strategy(make_strategy, [_sl_profile(stop_loss_pct=0.1)])
        _mark_long_open(strategy)
        state = _account_state(
            positions={BTC: _open_position(PositionSide.LONG, 1.0, 98.0)}
        )
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _flat_ohlcvs([SETUP_ID]), state, TIMESTAMP, bookkeeper, {TIMEFRAME}
        )

        updates = _stop_loss_updates(bookkeeper)
        assert len(updates) == 1
        assert updates[0].symbol == BTC
        assert updates[0].trigger_price == pytest.approx(98.0 * (1 - 0.1))

    def test_updates_stop_loss_from_average_entry_for_short(self, make_strategy):
        strategy = _loaded_strategy(make_strategy, [_sl_profile(stop_loss_pct=0.1)])
        for profile in strategy.profiles:
            _hold(strategy, profile.profile_id, PositionSide.SHORT, 1.0)
        state = _account_state(
            positions={BTC: _open_position(PositionSide.SHORT, 1.0, 102.0)}
        )
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _flat_ohlcvs([SETUP_ID]), state, TIMESTAMP, bookkeeper, {TIMEFRAME}
        )

        updates = _stop_loss_updates(bookkeeper)
        assert len(updates) == 1
        assert updates[0].trigger_price == pytest.approx(102.0 * (1 + 0.1))

    def test_stacked_profiles_share_one_stop(self, make_strategy):
        profiles = [
            _sl_profile(stop_loss_pct=0.05, tag="a"),
            _sl_profile(stop_loss_pct=0.05, tag="b"),
        ]
        strategy = _loaded_strategy(make_strategy, profiles)
        _mark_long_open(strategy)
        state = _account_state(
            positions={BTC: _open_position(PositionSide.LONG, 2.0, 100.0)}
        )
        bookkeeper = BookKeeper()
        profile_ids = [profile.profile_id for profile in strategy.profiles]

        strategy.book_trading_actions(
            _flat_ohlcvs(profile_ids), state, TIMESTAMP, bookkeeper, {TIMEFRAME}
        )

        updates = _stop_loss_updates(bookkeeper)
        assert len(updates) == 1
        assert updates[0].trigger_price == pytest.approx(95.0)

    def test_reissues_stop_each_candle(self, make_strategy):
        strategy = _loaded_strategy(make_strategy, [_sl_profile(stop_loss_pct=0.1)])
        _mark_long_open(strategy)
        state = _account_state(
            positions={BTC: _open_position(PositionSide.LONG, 1.0, 100.0)}
        )
        first_bookkeeper = BookKeeper()
        strategy.book_trading_actions(
            _flat_ohlcvs([SETUP_ID]), state, TIMESTAMP, first_bookkeeper, {TIMEFRAME}
        )

        second_bookkeeper = BookKeeper()
        strategy.book_trading_actions(
            _flat_ohlcvs([SETUP_ID]), state, TIMESTAMP, second_bookkeeper, {TIMEFRAME}
        )

        second_updates = _stop_loss_updates(second_bookkeeper)
        assert len(_stop_loss_updates(first_bookkeeper)) == 1
        assert len(second_updates) == 1
        assert second_updates[0].trigger_price == pytest.approx(90.0)

    def test_stop_follows_average_entry_when_it_moves(self, make_strategy):
        strategy = _loaded_strategy(make_strategy, [_sl_profile(stop_loss_pct=0.1)])
        _mark_long_open(strategy)
        strategy.book_trading_actions(
            _flat_ohlcvs([SETUP_ID]),
            _account_state(
                positions={BTC: _open_position(PositionSide.LONG, 1.0, 100.0)}
            ),
            TIMESTAMP,
            BookKeeper(),
            {TIMEFRAME},
        )
        state = _account_state(
            positions={BTC: _open_position(PositionSide.LONG, 2.0, 110.0)}
        )
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _flat_ohlcvs([SETUP_ID]), state, TIMESTAMP, bookkeeper, {TIMEFRAME}
        )

        updates = _stop_loss_updates(bookkeeper)
        assert len(updates) == 1
        assert updates[0].trigger_price == pytest.approx(110.0 * (1 - 0.1))

    def test_no_update_when_no_profile_has_stop_loss_pct(self, make_strategy):
        strategy = _loaded_strategy(make_strategy, [_sl_profile(stop_loss_pct=None)])
        _mark_long_open(strategy)
        state = _account_state(
            positions={BTC: _open_position(PositionSide.LONG, 1.0, 100.0)}
        )
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _flat_ohlcvs([SETUP_ID]), state, TIMESTAMP, bookkeeper, {TIMEFRAME}
        )

        assert len(_stop_loss_updates(bookkeeper)) == 0


class TestTrackerDeclaration:
    """The engine reconciles the tracker; the strategy only declares it."""

    def test_declares_the_tracker_it_keeps(self, make_strategy):
        strategy = make_strategy([_sl_profile(stop_loss_pct=0.1)])
        requirements = StrategyRequirements()

        asyncio.run(strategy.setup(requirements))

        declared = requirements.tracker._get_all()
        assert len(declared) == 1
        assert declared[0].tracker is strategy._position_tracker

    def test_declares_it_against_the_account_holding_the_positions(self, make_strategy):
        strategy = make_strategy([_sl_profile(stop_loss_pct=0.1)])
        requirements = StrategyRequirements()

        asyncio.run(strategy.setup(requirements))

        assert requirements.tracker._get_all()[0].account is strategy.account

    def test_declares_the_symbol_every_profile_trades(self, make_strategy):
        strategy = make_strategy(
            [
                _sl_profile(tag="alpha"),
                _sl_profile(tag="beta"),
            ]
        )
        requirements = StrategyRequirements()

        asyncio.run(strategy.setup(requirements))

        assert requirements.tracker._get_all()[0].ids_by_symbol == {
            BTC: tuple(profile.profile_id for profile in strategy.profiles)
        }


class TestStopLossCancelOnFlat:
    def test_closing_last_profile_cancels_resting_stop(self, make_strategy):
        strategy = _loaded_strategy(make_strategy, [_sl_profile(stop_loss_pct=0.1)])
        _mark_long_open(strategy)
        state = _account_state(
            positions={BTC: _open_position(PositionSide.LONG, 1.0, 100.0)},
            open_orders=[_resting_stop("sl-1")],
        )
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _long_exit_ohlcvs([SETUP_ID], {SETUP_ID}),
            state,
            TIMESTAMP,
            bookkeeper,
            {TIMEFRAME},
        )

        cancels = _cancel_orders(bookkeeper)
        assert len(cancels) == 1
        assert cancels[0].order_id == "sl-1"
        assert len(_stop_loss_updates(bookkeeper)) == 0

    def test_closing_one_of_two_stacked_keeps_stop(self, make_strategy):
        profiles = [
            _sl_profile(stop_loss_pct=0.05, tag="a"),
            _sl_profile(stop_loss_pct=0.05, tag="b"),
        ]
        strategy = _loaded_strategy(make_strategy, profiles)
        _mark_long_open(strategy)
        state = _account_state(
            positions={BTC: _open_position(PositionSide.LONG, 2.0, 100.0)},
            open_orders=[_resting_stop("sl-1")],
        )
        bookkeeper = BookKeeper()
        profile_ids = [profile.profile_id for profile in strategy.profiles]

        strategy.book_trading_actions(
            _long_exit_ohlcvs(profile_ids, {profile_ids[0]}),
            state,
            TIMESTAMP,
            bookkeeper,
            {TIMEFRAME},
        )

        assert len(_cancel_orders(bookkeeper)) == 0
        assert len(_stop_loss_updates(bookkeeper)) == 1


class TestStackedProfilesDisagreeing:
    """A symbol holds one net position, so stacked profiles must all reach it.

    The exchange nets whatever is booked, so the strategy's job is to emit every
    profile's own intent and let the venue combine them.
    """

    def _stacked(self, make_strategy, ratios: tuple[float, float]) -> ImpulseStrategy:
        return _loaded_strategy(
            make_strategy,
            [
                {**PROFILE, "tag": "alpha", "total_balance_ratio": ratios[0]},
                {**PROFILE, "tag": "beta", "total_balance_ratio": ratios[1]},
            ],
        )

    def _booked(self, bookkeeper: BookKeeper) -> list[tuple[OrderSide, float]]:
        return [(order.side, order.quantity) for order in _entry_orders(bookkeeper)]

    def test_opposite_signals_on_one_candle_both_reach_the_exchange(
        self, make_strategy
    ):
        strategy = self._stacked(make_strategy, (0.5, 0.25))
        alpha, beta = (profile.profile_id for profile in strategy.profiles)
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _signalled_ohlcvs({alpha: "long_entry", beta: "short_entry"}),
            _account_state(),
            TIMESTAMP,
            bookkeeper,
            {TIMEFRAME},
        )

        assert self._booked(bookkeeper) == [
            (OrderSide.BUY, 50.0),
            (OrderSide.SELL, 25.0),
        ]

    def test_offsetting_profiles_both_reach_the_exchange(self, make_strategy):
        strategy = self._stacked(make_strategy, (0.25, 0.25))
        alpha, beta = (profile.profile_id for profile in strategy.profiles)
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _signalled_ohlcvs({alpha: "long_entry", beta: "short_entry"}),
            _account_state(),
            TIMESTAMP,
            bookkeeper,
            {TIMEFRAME},
        )

        assert self._booked(bookkeeper) == [
            (OrderSide.BUY, 25.0),
            (OrderSide.SELL, 25.0),
        ]

    def test_a_profile_opening_against_a_held_position_still_books(self, make_strategy):
        strategy = self._stacked(make_strategy, (0.5, 0.25))
        alpha, beta = (profile.profile_id for profile in strategy.profiles)
        _hold(strategy, alpha, PositionSide.LONG, 50.0)
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _signalled_ohlcvs({alpha: "long_exit", beta: "short_entry"}),
            _account_state(
                positions={BTC: _open_position(PositionSide.LONG, 50.0, 100.0)}
            ),
            TIMESTAMP,
            bookkeeper,
            {TIMEFRAME},
        )

        assert self._booked(bookkeeper) == [
            (OrderSide.SELL, 50.0),
            (OrderSide.SELL, 25.0),
        ]

    def test_a_profile_opening_against_a_position_held_since_an_earlier_candle(
        self, make_strategy
    ):
        strategy = self._stacked(make_strategy, (0.5, 0.25))
        alpha, beta = (profile.profile_id for profile in strategy.profiles)
        _hold(strategy, alpha, PositionSide.LONG, 50.0)
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _signalled_ohlcvs({alpha: None, beta: "short_entry"}),
            _account_state(
                positions={BTC: _open_position(PositionSide.LONG, 50.0, 100.0)}
            ),
            TIMESTAMP,
            bookkeeper,
            {TIMEFRAME},
        )

        assert self._booked(bookkeeper) == [(OrderSide.SELL, 25.0)]

    def test_agreeing_profiles_both_book_their_own_size(self, make_strategy):
        strategy = self._stacked(make_strategy, (0.5, 0.25))
        alpha, beta = (profile.profile_id for profile in strategy.profiles)
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _signalled_ohlcvs({alpha: "long_entry", beta: "long_entry"}),
            _account_state(),
            TIMESTAMP,
            bookkeeper,
            {TIMEFRAME},
        )

        assert self._booked(bookkeeper) == [
            (OrderSide.BUY, 50.0),
            (OrderSide.BUY, 25.0),
        ]


def _book_exit_on_held(
    strategy: ImpulseStrategy, side: PositionSide
) -> FuturesMarketOrderAction:
    _hold_on_every_profile(strategy, side, 0.5)
    exiting_long = side == PositionSide.LONG
    exit_bookkeeper = BookKeeper()
    strategy.book_trading_actions(
        _make_ohlcvs(long_exit=exiting_long, short_exit=not exiting_long, close=110.0),
        _account_state(positions={BTC: _open_position(side, 0.5, 100.0)}),
        TIMESTAMP,
        exit_bookkeeper,
        {TIMEFRAME},
    )
    return _market_orders(exit_bookkeeper)[0]


def _market_orders(bookkeeper: BookKeeper) -> list[FuturesMarketOrderAction]:
    return [
        action
        for action in bookkeeper.list_actions()
        if isinstance(action, FuturesMarketOrderAction)
    ]


class TestTradeReasons:
    def test_a_long_entry_names_what_opened_it(self, strategy):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(long_entry=True, close=100.0),
            _account_state(),
            TIMESTAMP,
            bookkeeper,
            {TIMEFRAME},
        )

        assert _market_orders(bookkeeper)[0].reason == "impulse long entry"

    def test_a_short_entry_names_what_opened_it(self, strategy):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(short_entry=True, close=100.0),
            _account_state(),
            TIMESTAMP,
            bookkeeper,
            {TIMEFRAME},
        )

        assert _market_orders(bookkeeper)[0].reason == "impulse short entry"

    def test_a_long_exit_names_what_closed_it(self, strategy):
        closing = _book_exit_on_held(strategy, PositionSide.LONG)

        assert closing.reason == "impulse long exit"

    def test_a_short_exit_names_what_closed_it(self, strategy):
        closing = _book_exit_on_held(strategy, PositionSide.SHORT)

        assert closing.reason == "impulse short exit"


class TestOrderTags:
    def test_a_long_entry_uses_the_timeframe_without_a_tag(self, strategy):
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(long_entry=True, close=100.0),
            _account_state(),
            TIMESTAMP,
            bookkeeper,
            {TIMEFRAME},
        )

        order = _market_orders(bookkeeper)[0]
        assert tag_of(order.client_order_id) == "1d"

    def test_a_long_entry_composes_the_timeframe_with_the_profiles_tag(
        self, make_strategy
    ):
        strategy = _loaded_strategy(make_strategy, [{**PROFILE, "tag": "slow"}])
        bookkeeper = BookKeeper()

        strategy.book_trading_actions(
            _make_ohlcvs(
                long_entry=True, close=100.0, profile_id="BTC/USDT:USDT@1d-slow"
            ),
            _account_state(),
            TIMESTAMP,
            bookkeeper,
            {TIMEFRAME},
        )

        order = _market_orders(bookkeeper)[0]
        assert tag_of(order.client_order_id) == "1d-slow"

    def test_a_long_exit_carries_the_same_tag_as_its_entry(self, strategy):
        closing = _book_exit_on_held(strategy, PositionSide.LONG)

        assert tag_of(closing.client_order_id) == "1d"
