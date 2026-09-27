import logging
import math
from abc import ABC, abstractmethod
from datetime import datetime
from typing import cast

from robottraderslab._core import FillDescriber, OHLCVsBySymbol, Symbol
from robottraderslab.exceptions import ExchangeRecoverableError, StrategyCriticalError
from robottraderslab.exchanges import (
    Balance,
    Currency,
    MarketType,
    OrderSide,
    OrderType,
    PositionSide,
    PositionSnapshot,
    StopLoss,
    TakeProfit,
    TimeInForce,
    VenueFill,
)

from .currency_converter import CurrencyConverter, Prices
from .daily_equity_recorder import DailyEquityRecorder
from .equity_snapshot import EquitySnapshot
from .fees import FeeRates
from .fill_recorder import (
    FillRecorder,
    NullFillRecorder,
)
from .order_book import OrderBook
from .order_models import (
    LimitOrder,
    MarketOrder,
    Order,
    StopLossOrder,
    TakeProfitOrder,
    TriggerOrder,
    new_group_id,
)
from .simulated_position import SimulatedPosition
from .trade_equity_recorder import TradeEquityRecorder

logger = logging.getLogger(__name__)

_LIMIT = OrderType.LIMIT
_MARKET = OrderType.MARKET
_STOP_LOSS = OrderType.STOP_LOSS
_TAKE_PROFIT = OrderType.TAKE_PROFIT
_TRIGGER = OrderType.TRIGGER


class BaseSimulationEngine(ABC):
    """Matches orders and tracks equity for every simulated market; a
    subclass supplies only how a position enters and exits.
    """

    def __init__(
        self,
        balances: dict[Currency, Balance],
        fee_rates: FeeRates,
        fill_recorder: FillRecorder | None = None,
        market_type: MarketType = "futures",
        converter: CurrencyConverter | None = None,
    ) -> None:
        self._converter = converter
        self._fee_rates = fee_rates
        self._order_book = OrderBook()
        self._fill_recorder: FillRecorder = fill_recorder or NullFillRecorder()
        self._balances = balances
        self._market_type = market_type
        self._daily_equity_recorder = DailyEquityRecorder()
        self._trade_equity_recorder = TradeEquityRecorder()
        self._last_tick_snapshot: EquitySnapshot | None = None
        self._last_seen_prices: Prices = {}
        self._trades_occurred_this_tick = False
        self._last_fills: dict[str, VenueFill] = {}
        self._current_ohlcvs: OHLCVsBySymbol = {}

    @property
    def converter(self) -> CurrencyConverter:
        if self._converter is None:
            self._converter = CurrencyConverter(self.equity_currency)
        return self._converter

    @property
    @abstractmethod
    def equity_currency(self) -> Currency: ...

    @property
    @abstractmethod
    def open_positions(self) -> dict[Symbol, PositionSnapshot]: ...

    @property
    def open_orders(self) -> list[Order]:
        return list(self._order_book)

    def get_balances(self) -> dict[Currency, Balance]:
        return self._balances.copy()

    def get_equity(self, currency: Currency) -> float:
        if self._last_tick_snapshot is not None:
            return self._last_tick_snapshot.get_equity(currency)
        return self._initial_equity(currency)

    def get_daily_equity_snapshots(self) -> list[EquitySnapshot]:
        return self._daily_equity_recorder.snapshots

    def get_trade_equity_snapshots(self) -> list[EquitySnapshot]:
        return self._trade_equity_recorder.snapshots

    def get_last_fills(self) -> dict[str, VenueFill]:
        return self._last_fills

    def record_prices(self, prices: Prices) -> None:
        """Seed the last-seen prices used for valuation and currency conversion."""
        self._last_seen_prices.update(prices)

    def record_opening_equity(self, moment: datetime) -> None:
        """Record the equity the run starts with at its start, a moment no
        candle of the run closes on.
        """
        self._update_equity_snapshot(moment)
        self._daily_equity_recorder.record_bound(self._last_tick_snapshot)  # type: ignore[arg-type]

    def record_closing_equity(self) -> None:
        """Keep the last candle's equity in the daily curve, whatever moment
        that candle closed on, once the orders placed at its close have filled.
        """
        self._record_trade_equity()
        self._daily_equity_recorder.record_bound(self._last_tick_snapshot)  # type: ignore[arg-type]

    def enable_execution_recording(self) -> None:
        """Hook called once, before the first candle, for a run that needs
        executions read back. Override to act on it; building an Execution
        on every fill is wasted work for a run that never asks.
        """
        pass

    def describe_fills_with(self, describer: FillDescriber) -> None:
        """Hook called once, before the first candle, with the strategy's
        describer. Override to consult it for every fill whose order states no
        reason.
        """
        pass

    def add_order(self, order: LimitOrder | MarketOrder | TriggerOrder) -> None:
        """A resting order meets only the candles that follow the one it was
        placed after. On a symbol with no candle closing then, or a gap row
        standing where one would, one that cannot rest is decided at that
        symbol's next published candle all the same.

        Raises:
            ExchangeRecoverableError: If a `POST_ONLY` order would take
                liquidity at the close it is placed at, or the placement
                reserve refuses a limit entry.
        """
        ohlcv = self._current_ohlcvs.get(order.symbol)
        if ohlcv is None or math.isnan(ohlcv.close):
            self._rest_order(order)
            return
        if _crosses_as_post_only(order, ohlcv.close):
            raise ExchangeRecoverableError(
                f"{order} would take liquidity at {ohlcv.close}: a post-only "
                "order is refused rather than filled as a taker"
            )
        self._rest_order(order)
        if order.kind == _MARKET:
            self._execute_market_order(
                ohlcv.timestamp, cast(MarketOrder, order), ohlcv.close
            )
        elif (
            order.kind == _LIMIT
            and cast(LimitOrder, order).time_in_force is not TimeInForce.GTC
        ):
            self._settled_at_placement(
                ohlcv.timestamp, cast(LimitOrder, order), ohlcv.close
            )

    def cancel_order_by_id(self, symbol: Symbol, order_id: str) -> bool:
        """Cancel a single open order by id. Returns False if no order matched."""
        cancelled = self._order_book.cancel_order_by_id(symbol, order_id)
        if cancelled is None:
            return False
        if _is_limit_entry(cancelled):
            self._on_release_limit_entry(cast(LimitOrder, cancelled))
        return True

    def cancel_orders_for_symbol(self, symbol: Symbol) -> None:
        """Cancel all open orders for the symbol."""
        for cancelled in self._order_book.cancel_orders_for_symbol(symbol):
            if _is_limit_entry(cancelled):
                self._on_release_limit_entry(cast(LimitOrder, cancelled))

    def simulate_on_current_ohlcvs(
        self, timestamp_arg: datetime, ohlcvs: OHLCVsBySymbol
    ) -> None:
        """Play the candles that closed at `timestamp_arg` against the orders
        resting before they opened, then record the figures of that close.

        An order placed after this call and before the next one is placed at
        this close, so it meets the next candle and never one it was decided
        on.
        """
        self._record_trade_equity()
        self._current_ohlcvs = ohlcvs
        self._last_fills = {}
        self._record_current_prices(ohlcvs)

        taker_fee_rate = self._fee_rates.taker
        logger.debug("Pending orders:")
        for symbol in ohlcvs:
            ohlcv = ohlcvs[symbol]
            timestamp = ohlcv.timestamp
            for order in self._order_book.orders_for(symbol):
                logger.debug("  %s", order)

                try:
                    if order.kind == _LIMIT:
                        limit_order = cast(LimitOrder, order)
                        if (
                            limit_order.time_in_force is not TimeInForce.GTC
                            and not limit_order.settled
                            and not math.isnan(ohlcv.close)
                            and self._settled_at_placement(
                                timestamp, limit_order, ohlcv.close
                            )
                        ):
                            continue
                        low, high = ohlcv.low, ohlcv.high
                        fee_rate = self._fee_rates.maker
                        if low <= limit_order.limit_price <= high:
                            if limit_order.reduce_only:
                                self._exit_position(
                                    timestamp,
                                    limit_order,
                                    limit_order.limit_price,
                                    fee_rate,
                                )
                            else:
                                self._on_release_limit_entry(limit_order)
                                self._enter_position(
                                    timestamp,
                                    limit_order,
                                    limit_order.limit_price,
                                    fee_rate,
                                    taker_fee_rate,
                                )

                    elif order.kind == _MARKET and not math.isnan(ohlcv.close):
                        price = ohlcv.close
                        self._execute_market_order(
                            timestamp, cast(MarketOrder, order), price
                        )

                    elif order.kind == _STOP_LOSS:
                        stop_loss_order = cast(StopLossOrder, order)
                        trigger_price = stop_loss_order.trigger_price
                        fee_rate = self._fee_rates.taker
                        position_side = self._get_position_side(order.symbol)
                        exit_long = (
                            position_side == PositionSide.LONG
                            and ohlcv.low <= trigger_price
                        )
                        exit_short = (
                            position_side == PositionSide.SHORT
                            and ohlcv.high >= trigger_price
                        )

                        if exit_long or exit_short:
                            self._exit_position(
                                timestamp,
                                stop_loss_order,
                                trigger_price,
                                fee_rate,
                            )

                    elif order.kind == _TAKE_PROFIT:
                        take_profit_order = cast(TakeProfitOrder, order)
                        trigger_price = take_profit_order.trigger_price
                        fee_rate = self._fee_rates.taker
                        position_side = self._get_position_side(order.symbol)
                        exit_long = (
                            position_side == PositionSide.LONG
                            and ohlcv.high >= trigger_price
                        )
                        exit_short = (
                            position_side == PositionSide.SHORT
                            and ohlcv.low <= trigger_price
                        )

                        if exit_long or exit_short:
                            self._exit_position(
                                timestamp,
                                take_profit_order,
                                trigger_price,
                                fee_rate,
                            )

                    elif order.kind == _TRIGGER:
                        trigger_order = cast(TriggerOrder, order)
                        low, high = ohlcv.low, ohlcv.high
                        if low <= trigger_order.trigger_price <= high:
                            trigger_order.order.triggered = True
                            self._rest_order(trigger_order.order)
                            if trigger_order.order.kind == _MARKET:
                                self._execute_market_order(
                                    timestamp,
                                    cast(MarketOrder, trigger_order.order),
                                    trigger_order.trigger_price,
                                )
                            self._order_book.remove_order(trigger_order)

                except ExchangeRecoverableError as e:
                    logger.warning(e)

        self._after_tick(timestamp_arg, ohlcvs)
        self._update_equity_snapshot(timestamp_arg)
        self._daily_equity_recorder.record_snapshot(self._last_tick_snapshot)  # type: ignore[arg-type]

    def _settled_at_placement(
        self, timestamp: datetime, limit_order: LimitOrder, close: float
    ) -> bool:
        """An `IOC` order fills or is cancelled at the close it is settled
        against, and never rests. A `POST_ONLY` order rests like a GTC order,
        except one placed with no close to judge it by, which is cancelled at
        WARNING when the first close it meets shows it would have taken
        liquidity.

        Returns:
            Whether the order is done with, filled or cancelled.
        """
        limit_order.settled = True
        if limit_order.time_in_force is TimeInForce.POST_ONLY:
            if _takes_liquidity(limit_order, close):
                logger.warning(
                    "%s would take liquidity at %s and is cancelled", limit_order, close
                )
                self._cancel_at_placement(limit_order)
                return True
            return False
        if _takes_liquidity(limit_order, close):
            self._order_book.remove_order(limit_order)
            self._fill_taking_liquidity(timestamp, limit_order, close)
            return True
        logger.info("%s is not filled at %s and is cancelled", limit_order, close)
        self._cancel_at_placement(limit_order)
        return True

    def _rest_order(self, order: LimitOrder | MarketOrder | TriggerOrder) -> None:
        if _is_limit_entry(order):
            self._on_add_limit_entry(cast(LimitOrder, order))
        self._order_book.add_order(order)

    def _cancel_at_placement(self, limit_order: LimitOrder) -> None:
        self._order_book.remove_order(limit_order)
        if _is_limit_entry(limit_order):
            self._on_release_limit_entry(limit_order)

    def _fill_taking_liquidity(
        self, timestamp: datetime, limit_order: LimitOrder, price: float
    ) -> None:
        taker_fee_rate = self._fee_rates.taker
        if limit_order.reduce_only:
            self._exit_position(timestamp, limit_order, price, taker_fee_rate)
            return
        self._on_release_limit_entry(limit_order)
        self._enter_position(
            timestamp, limit_order, price, taker_fee_rate, taker_fee_rate
        )

    def _execute_market_order(
        self,
        timestamp: datetime,
        market_order: MarketOrder,
        price: float,
    ) -> None:
        taker_fee_rate = self._fee_rates.taker
        if not market_order.reduce_only:
            self._enter_position(
                timestamp, market_order, price, taker_fee_rate, taker_fee_rate
            )
        else:
            self._exit_position(timestamp, market_order, price, taker_fee_rate)

    def _get_position_side(self, _symbol: Symbol) -> PositionSide:
        """Defaults to LONG here, since spot trading has no short side."""
        return PositionSide.LONG

    def _initial_equity(self, currency: Currency) -> float:
        """Assumes a single initial balance per currency, since equity has no
        snapshot yet to read a truer figure from.
        """
        balance = self._balances.get(currency)
        if balance is None:
            return 0.0
        return balance.total

    def _record_current_prices(self, ohlcvs_by_symbol: OHLCVsBySymbol) -> None:
        for symbol, ohlcv in ohlcvs_by_symbol.items():
            if not math.isnan(ohlcv.close):
                self._last_seen_prices[symbol] = ohlcv.close

    def _record_trade_equity(self) -> None:
        """Value the account at the last simulated moment once every trade of
        that moment has settled, the candle's own fills and the orders placed
        at its close alike, so the moment holds one trade snapshot.
        """
        if not self._trades_occurred_this_tick:
            return
        self._update_equity_snapshot(self._last_tick_snapshot.timestamp)  # type: ignore[union-attr]
        self._trade_equity_recorder.record_snapshot(self._last_tick_snapshot)  # type: ignore[arg-type]
        self._trades_occurred_this_tick = False

    def _update_equity_snapshot(self, timestamp: datetime) -> None:
        self._last_tick_snapshot = EquitySnapshot(
            timestamp=timestamp,
            market_type=self._market_type,
            balances=self.get_balances(),
            positions=self._get_positions_for_snapshot(),
            prices=self._last_seen_prices,
            converter=self.converter,
        )

    def _get_positions_for_snapshot(self) -> dict[Symbol, SimulatedPosition]:
        """Return positions dict for equity snapshot. Override in subclasses."""
        return {}

    def _open_tp_sl(self, order: LimitOrder | MarketOrder) -> str | None:
        """Open the resting guards `order` requests, if any.

        Guards opened for one position across separate calls join the same
        contingency group, so the order book resolves them together. The
        group is read off the symbol, which holds while a symbol carries one
        position at a time.

        Returns:
            The group the symbol's guards belong to, `None` when `order`
            asked for none.
        """
        stop_loss: StopLoss | None = getattr(order, "stop_loss", None)
        take_profit: TakeProfit | None = getattr(order, "take_profit", None)
        if not (stop_loss or take_profit):
            return None

        existing_sl = self._order_book.find_guard(order.symbol, StopLossOrder)
        existing_tp = self._order_book.find_guard(order.symbol, TakeProfitOrder)
        group_id = (
            (existing_sl.group_id if existing_sl is not None else None)
            or (existing_tp.group_id if existing_tp is not None else None)
            or new_group_id()
        )

        if stop_loss is not None and existing_sl is None:
            self._order_book.add_order(
                StopLossOrder(
                    side=order.side.opposite,
                    symbol=order.symbol,
                    trigger_price=stop_loss.trigger_price,
                    reason=stop_loss.reason,
                    group_id=group_id,
                )
            )

        if take_profit is not None and existing_tp is None:
            self._order_book.add_order(
                TakeProfitOrder(
                    side=order.side.opposite,
                    symbol=order.symbol,
                    trigger_price=take_profit.trigger_price,
                    reason=take_profit.reason,
                    group_id=group_id,
                )
            )

        return group_id

    def _validate_tp_sl_prices(
        self, order: LimitOrder | MarketOrder, entry_price: float
    ) -> None:
        """A protection on the wrong side of its entry is a defect of the
        strategy that booked it, so it stops the run whether the entry fills
        at placement or later on a candle.

        Raises:
            StrategyCriticalError: If the stop-loss or the take-profit sits at
                or beyond the entry price on the side it would close.
        """
        sl = order.stop_loss
        tp = order.take_profit

        if sl is None and tp is None:
            return

        entry = f"{'long' if order.side == OrderSide.BUY else 'short'} entry on {order.symbol} at {entry_price}"
        if order.side == OrderSide.BUY:
            if sl and sl.trigger_price >= entry_price:
                raise StrategyCriticalError(
                    f"Invalid SL {sl.trigger_price} for {entry}: SL must be below entry price"
                )
            if tp and tp.trigger_price <= entry_price:
                raise StrategyCriticalError(
                    f"Invalid TP {tp.trigger_price} for {entry}: TP must be above entry price"
                )
        else:
            if sl and sl.trigger_price <= entry_price:
                raise StrategyCriticalError(
                    f"Invalid SL {sl.trigger_price} for {entry}: SL must be above entry price"
                )
            if tp and tp.trigger_price >= entry_price:
                raise StrategyCriticalError(
                    f"Invalid TP {tp.trigger_price} for {entry}: TP must be below entry price"
                )

    def _on_add_limit_entry(self, limit_order: LimitOrder) -> None:
        """Hook called when a limit entry is added, whatever its side. Override
        to lock the funds it would fill against.
        """
        pass

    def _on_release_limit_entry(self, limit_order: LimitOrder) -> None:
        """Hook called when a limit entry leaves the book, matched or cancelled.
        Override to release what `_on_add_limit_entry` locked.
        """
        pass

    def _after_tick(self, timestamp: datetime, ohlcvs: OHLCVsBySymbol) -> None:
        """Hook called after the order matching loop. Override for post-tick
        logic (e.g. liquidations).
        """
        pass

    @abstractmethod
    def _enter_position(
        self,
        timestamp: datetime,
        order: LimitOrder | MarketOrder,
        price: float,
        fee_rate: float,
        taker_fee_rate: float,
    ) -> None: ...

    @abstractmethod
    def _exit_position(
        self,
        timestamp: datetime,
        order: LimitOrder | MarketOrder | StopLossOrder | TakeProfitOrder,
        price: float,
        fee_rate: float,
    ) -> None: ...


def _is_limit_entry(order: Order) -> bool:
    return order.kind == _LIMIT and not cast(LimitOrder, order).reduce_only


def _crosses_as_post_only(
    order: LimitOrder | MarketOrder | TriggerOrder, close: float
) -> bool:
    return (
        order.kind == _LIMIT
        and cast(LimitOrder, order).time_in_force is TimeInForce.POST_ONLY
        and _takes_liquidity(cast(LimitOrder, order), close)
    )


def _takes_liquidity(limit_order: LimitOrder, close: float) -> bool:
    if limit_order.side == OrderSide.BUY:
        return limit_order.limit_price >= close
    return limit_order.limit_price <= close
