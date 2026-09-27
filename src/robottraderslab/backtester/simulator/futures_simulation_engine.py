import logging
from bisect import bisect_left
from datetime import datetime

from robottraderslab._core import (
    FillDescriber,
    OHLCVsBySymbol,
    OrderType,
    StepCount,
    Symbol,
    tag_of,
    to_quantity,
    to_step_count,
)
from robottraderslab.exceptions import ExchangeRecoverableError
from robottraderslab.exchanges import (
    Balance,
    Currency,
    Execution,
    OrderSide,
    PositionSide,
    PositionSnapshot,
    VenueFill,
)

from .base_simulation_engine import BaseSimulationEngine
from .currency_converter import CurrencyConverter
from .fees import FeeModel, FeeRates
from .fill_recorder import FillRecorder, FillType
from .order_models import (
    LimitOrder,
    MarketOrder,
    StopLossOrder,
    TakeProfitOrder,
    new_execution_id,
    new_order_id,
    reported_reason,
)
from .simulated_position import SimulatedPosition

logger = logging.getLogger(__name__)


class FuturesSimulationEngine(BaseSimulationEngine):
    """Adds margin, leverage and liquidation to the shared order matching."""

    def __init__(
        self,
        initial_balance: dict[Currency, float],
        fee_rates: FeeRates,
        margin_currency: Currency,
        fee_model: FeeModel,
        fill_recorder: FillRecorder,
    ) -> None:
        self._margin_currency = margin_currency
        self._fee_model = fee_model
        self._currency_converter = CurrencyConverter(margin_currency)
        self._positions: dict[Symbol, SimulatedPosition] = {}
        self._leverage_by_symbol: dict[Symbol, float] = {}
        self._executions: list[Execution] = []
        self._record_executions = False
        self._describe_fill: FillDescriber | None = None

        available = float(initial_balance.get(margin_currency, 0.0))
        balances = {margin_currency: Balance.compute(available=available, locked=0.0)}

        super().__init__(
            balances=balances,
            fee_rates=fee_rates,
            fill_recorder=fill_recorder,
            market_type="futures",
            converter=self._currency_converter,
        )

    @property
    def equity_currency(self) -> Currency:
        return self._margin_currency

    @property
    def margin_currency(self) -> Currency:
        return self._margin_currency

    @property
    def open_positions(self) -> dict[Symbol, PositionSnapshot]:
        return {symbol: pos.snapshot() for symbol, pos in self._positions.items()}

    @property
    def placement_reserve_rate(self) -> float:
        return self._fee_model.placement_reserve_rate(self._fee_rates.taker)

    def enable_execution_recording(self) -> None:
        self._record_executions = True

    def describe_fills_with(self, describer: FillDescriber) -> None:
        self._describe_fill = describer

    def get_executions_since(self, since: datetime) -> list[Execution]:
        """Return every execution booked at or after `since`, in booking order.

        A backtest replays candles in order, so executions are always
        appended in non-decreasing timestamp order.
        """
        start = bisect_left(
            self._executions, since, key=lambda execution: execution.timestamp
        )
        return self._executions[start:]

    def set_symbol_leverage(self, symbol: Symbol, leverage: float) -> None:
        """
        Args:
            leverage: Leverage value, must be >= 1.0.

        Raises:
            ValueError: If leverage < 1.0
        """
        if leverage < 1.0:
            raise ValueError(f"Leverage must be >= 1.0, got {leverage}")
        self._leverage_by_symbol[symbol] = leverage

    def get_symbol_leverage(self, symbol: Symbol) -> float:
        return self._leverage_by_symbol.get(symbol, 1.0)

    def order_steps(self, quantity: float) -> StepCount:
        """A placement counts an order the way its fill will, so it can refuse
        what would fill at nothing.
        """
        return self._fee_model.order_steps(quantity)

    def placement_requirement_rate(self, leverage: float) -> float:
        return self._fee_model.placement_requirement_rate(
            leverage, self._fee_rates.taker
        )

    def quote_conversion_rate(self, symbol: Symbol) -> float:
        """Rate converting the symbol's quote currency into the margin currency.

        Raises:
            MissingConversionRateError: If no conversion price has been seen yet.
        """
        return self._currency_converter.rate(symbol.quote, self._last_seen_prices)

    def _get_positions_for_snapshot(self) -> dict[Symbol, SimulatedPosition]:
        return self._positions

    def _on_add_limit_entry(self, limit_order: LimitOrder) -> None:
        """A resting entry locks what the quantity it would open needs beyond
        the margin its closing leg would release, against the position held
        when it is placed; the fill weighs the position it meets then.
        """
        try:
            rate = self.quote_conversion_rate(limit_order.symbol)
            opening_steps, released_margin = self._opening_leg(
                limit_order, self.order_steps(limit_order.quantity)
            )
            notional = to_quantity(opening_steps) * limit_order.limit_price * rate
            self._require_placement_balance(
                limit_order.symbol, notional, released_margin
            )
            limit_order.locked_margin = max(
                self._margin_of(limit_order.symbol, notional) - released_margin, 0.0
            )
            self._update_balance(locked_delta=limit_order.locked_margin)
        except RuntimeError as e:
            raise ExchangeRecoverableError(
                f"{e}, placing an order on {limit_order.symbol}"
            ) from e

    def _on_release_limit_entry(self, limit_order: LimitOrder) -> None:
        self._update_balance(locked_delta=-limit_order.locked_margin)

    def _after_tick(self, timestamp: datetime, ohlcvs: OHLCVsBySymbol) -> None:
        self._check_liquidations(timestamp, ohlcvs)

    def _require_placement_balance(
        self, symbol: Symbol, notional: float, released_margin: float
    ) -> None:
        """The margin an order's closing leg releases counts towards what its
        opening leg needs, as the venue counts it.

        Raises:
            ExchangeRecoverableError: If the balance, with that margin, is
                under what the fee model's venue holds back for an order of
                this notional.
        """
        required = self._fee_model.placement_requirement(
            notional, self.get_symbol_leverage(symbol), self._fee_rates.taker
        )
        available = self._balances[self._margin_currency].available + released_margin
        if available < required:
            raise ExchangeRecoverableError(
                f"Order of {notional} {self._margin_currency} notional on {symbol} "
                f"requires {required}, {available} is available to it"
            )

    def _margin_of(self, symbol: Symbol, notional: float) -> float:
        return notional / self.get_symbol_leverage(symbol)

    def _opening_leg(
        self, order: LimitOrder | MarketOrder, requested_steps: StepCount
    ) -> tuple[StepCount, float]:
        """A venue verifies the value an order opens; what closes a position it
        holds needs nothing and releases the margin backing it.

        Returns:
            The steps the order opens and the margin its closing leg releases.
        """
        existing = self._positions.get(order.symbol)
        if existing is None or existing.side == _determine_position_side(order):
            return requested_steps, 0.0
        close_steps = min(requested_steps, existing.quantity_steps)
        return requested_steps - close_steps, existing.margin_backing(close_steps)

    def _enter_position(
        self,
        timestamp: datetime,
        order: LimitOrder | MarketOrder,
        price: float,
        fee_rate: float,
        taker_fee_rate: float,
    ) -> None:
        self._order_book.remove_order(order)
        self._validate_tp_sl_prices(order, price)

        requested_steps = self.order_steps(order.quantity)
        entry_side = _determine_position_side(order)
        existing = self._positions.get(order.symbol)
        rate = self.quote_conversion_rate(order.symbol)

        if existing is not None and existing.side != entry_side:
            opening_steps, released_margin = self._opening_leg(order, requested_steps)
            self._require_placement_balance(
                order.symbol, to_quantity(opening_steps) * price * rate, released_margin
            )
            self._handle_opposing_entry(
                timestamp,
                order,
                price,
                requested_steps,
                fee_rate,
                taker_fee_rate,
                existing,
                rate,
            )
        else:
            notional = to_quantity(requested_steps) * price * rate
            self._require_placement_balance(order.symbol, notional, 0.0)
            self._fill_same_side(
                timestamp,
                order,
                price,
                requested_steps,
                fee_rate,
                taker_fee_rate,
                rate,
                notional,
                entry_side,
            )
        self._trades_occurred_this_tick = True

    def _fill_same_side(
        self,
        timestamp: datetime,
        order: LimitOrder | MarketOrder,
        price: float,
        requested_steps: StepCount,
        fee_rate: float,
        taker_fee_rate: float,
        rate: float,
        notional: float,
        entry_side: PositionSide,
    ) -> None:
        required_margin = self._margin_of(order.symbol, notional)
        fee = notional * fee_rate
        net_steps = self._fee_model.entry_steps(requested_steps, fee_rate)
        net_quantity = to_quantity(net_steps)

        try:
            self._update_balance(
                locked_delta=self._fee_model.entry_locked_delta(required_margin, fee),
                total_delta=-fee,
            )
        except RuntimeError as e:
            raise ExchangeRecoverableError(
                f"{e}, filling an order on {order.symbol}"
            ) from e

        created = self._grow_position(
            timestamp,
            order,
            entry_side,
            net_steps,
            price,
            self._margin_of(order.symbol, net_quantity * price * rate),
            taker_fee_rate,
        )

        self._last_fills[order.order_id] = VenueFill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=net_quantity,
            timestamp=timestamp,
        )
        self._record_entry(
            timestamp,
            order,
            price=price,
            requested_quantity=to_quantity(requested_steps),
            net_quantity=net_quantity,
            fee=fee,
            rate=rate,
            fill_type=_get_entry_fill_type(created, entry_side),
            side=entry_side,
        )

    def _grow_position(
        self,
        timestamp: datetime,
        order: LimitOrder | MarketOrder,
        side: PositionSide,
        net_steps: StepCount,
        price: float,
        margin: float,
        taker_fee_rate: float,
    ) -> bool:
        position = self._positions.setdefault(
            order.symbol,
            SimulatedPosition(
                symbol=order.symbol,
                side=side,
                leverage=self.get_symbol_leverage(order.symbol),
                taker_fee_rate=taker_fee_rate,
                entry_time=timestamp,
            ),
        )
        created = position.quantity_steps == 0
        position.add_to_position(net_steps, price)
        position.lock_margin(margin)
        position.group_id = self._open_tp_sl(order) or position.group_id
        return created

    def _exit_position(
        self,
        timestamp: datetime,
        order: LimitOrder | MarketOrder | StopLossOrder | TakeProfitOrder,
        price: float,
        fee_rate: float,
    ) -> None:
        self._order_book.remove_order(order)

        position = self._positions.get(order.symbol)
        if position is None:
            raise ExchangeRecoverableError(
                f"No position to exit for symbol: {order.symbol}"
            )

        if order.quantity is None:
            exit_steps = position.quantity_steps
        else:
            exit_steps = min(position.quantity_steps, to_step_count(order.quantity))
        exit_quantity = to_quantity(exit_steps)

        rate = self.quote_conversion_rate(order.symbol)
        fee = exit_quantity * price * fee_rate * rate
        released_margin = position.release_margin(exit_steps)
        realised_pnl = position.get_realised_PnL(exit_quantity, price) * rate
        self._update_balance(
            locked_delta=-released_margin,
            total_delta=realised_pnl - fee,
        )

        position.reduce_position(exit_steps)
        closed = position.quantity_steps == 0
        if closed:
            self._close_position(position)

        self._last_fills[order.order_id] = VenueFill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=exit_quantity,
            timestamp=timestamp,
        )

        reason = self._record_execution(
            order,
            timestamp=timestamp,
            price=price,
            quantity=exit_quantity,
            realised_profit=realised_pnl,
            fee=fee,
        )
        self._fill_recorder.record_fill(
            timestamp,
            str(order.symbol),
            position.side,
            exit_quantity,
            exit_quantity,
            price,
            fee,
            _get_exit_fill_type(closed, position.side),
            reason=reason,
            tag=tag_of(order.client_order_id),
            extra_fields=order.extra_fields,
            rate=rate,
        )

        if isinstance(order, StopLossOrder):
            logger.info(
                "SL FIRED %s @ %s side=%s qty=%s price=%s",
                order.symbol,
                timestamp,
                position.side.value,
                exit_quantity,
                price,
            )

        self._trades_occurred_this_tick = True

    def _handle_opposing_entry(
        self,
        timestamp: datetime,
        order: LimitOrder | MarketOrder,
        price: float,
        requested_steps: StepCount,
        fee_rate: float,
        taker_fee_rate: float,
        existing: SimulatedPosition,
        rate: float,
    ) -> None:
        """Only called once the caller has confirmed the order's side
        opposes `existing`, so quantity left over after the close always
        flips onto a new position and never adds to the side just closed.
        """
        fee = to_quantity(requested_steps) * price * fee_rate * rate
        close_steps = min(requested_steps, existing.quantity_steps)
        close_fee = self._fee_model.closing_leg_fee(fee, close_steps, requested_steps)
        remainder_fee = fee - close_fee
        excess_steps = requested_steps - close_steps
        close_quantity = to_quantity(close_steps)
        released_margin = existing.release_margin(close_steps)
        realised_pnl = existing.get_realised_PnL(close_quantity, price) * rate

        existing.reduce_position(close_steps)
        position_closed = existing.quantity_steps == 0

        reason = self._record_execution(
            order,
            timestamp=timestamp,
            price=price,
            quantity=close_quantity,
            realised_profit=realised_pnl,
            fee=close_fee,
        )
        self._fill_recorder.record_fill(
            timestamp,
            str(order.symbol),
            existing.side,
            close_quantity,
            close_quantity,
            price,
            close_fee,
            _get_exit_fill_type(position_closed, existing.side),
            reason=reason,
            tag=tag_of(order.client_order_id),
            extra_fields=order.extra_fields,
            rate=rate,
        )

        if position_closed:
            self._close_position(existing)

        remainder_steps = self._open_flipped_remainder(
            timestamp,
            order,
            price,
            excess_steps,
            fee_rate,
            taker_fee_rate,
            rate,
            released_margin,
            realised_pnl,
            fee,
            remainder_fee,
        )

        self._last_fills[order.order_id] = VenueFill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=to_quantity(close_steps + remainder_steps),
            timestamp=timestamp,
        )

    def _open_flipped_remainder(
        self,
        timestamp: datetime,
        order: LimitOrder | MarketOrder,
        price: float,
        excess_steps: StepCount,
        fee_rate: float,
        taker_fee_rate: float,
        rate: float,
        released_margin: float,
        realised_pnl: float,
        order_fee: float,
        remainder_fee: float,
    ) -> StepCount:
        """Open the excess of a flip on the side it opposed, once the close
        leg it flips through has already released its margin.

        Returns:
            The steps opened, zero when the closing leg used the whole order.
        """
        if excess_steps <= 0:
            self._update_balance(
                locked_delta=-released_margin, total_delta=realised_pnl - order_fee
            )
            return 0

        remainder_steps = self._fee_model.entry_steps(excess_steps, fee_rate)
        remainder = to_quantity(remainder_steps)
        new_side = _determine_position_side(order)
        new_margin = self._margin_of(order.symbol, remainder * price * rate)

        self._update_balance(
            locked_delta=-released_margin + new_margin,
            total_delta=realised_pnl - order_fee,
        )

        new_position = SimulatedPosition(
            symbol=order.symbol,
            side=new_side,
            leverage=self.get_symbol_leverage(order.symbol),
            taker_fee_rate=taker_fee_rate,
            entry_time=timestamp,
        )
        new_position.add_to_position(remainder_steps, price)
        new_position.lock_margin(new_margin)
        self._positions[order.symbol] = new_position
        new_position.group_id = self._open_tp_sl(order)

        self._record_entry(
            timestamp,
            order,
            price=price,
            requested_quantity=remainder,
            net_quantity=remainder,
            fee=remainder_fee,
            rate=rate,
            fill_type=_get_entry_fill_type(True, new_side),
            side=new_side,
        )
        return remainder_steps

    def _record_entry(
        self,
        timestamp: datetime,
        order: LimitOrder | MarketOrder,
        *,
        price: float,
        requested_quantity: float,
        net_quantity: float,
        fee: float,
        rate: float,
        fill_type: FillType,
        side: PositionSide,
    ) -> None:
        reason = self._record_execution(
            order,
            timestamp=timestamp,
            price=price,
            quantity=net_quantity,
            realised_profit=0.0,
            fee=fee,
        )
        self._fill_recorder.record_fill(
            timestamp,
            str(order.symbol),
            side,
            requested_quantity,
            net_quantity,
            price,
            fee,
            fill_type,
            reason=reason,
            tag=tag_of(order.client_order_id),
            extra_fields=order.extra_fields,
            rate=rate,
        )

    def _get_position_side(self, symbol: Symbol) -> PositionSide:
        return self._positions[symbol].side

    def _record_execution(
        self,
        order: LimitOrder | MarketOrder | StopLossOrder | TakeProfitOrder,
        *,
        timestamp: datetime,
        price: float,
        quantity: float,
        realised_profit: float,
        fee: float,
    ) -> str | None:
        """Return the fill's reason, recording the execution when the run asks.

        One execution serves the describer and the record alike, so a fill
        builds it at most once, and only when either of them needs it.
        """
        describer = self._describe_fill if order.reason is None else None
        if describer is None and not self._record_executions:
            return reported_reason(order, order.reason)
        execution = _execution_of(
            order,
            timestamp=timestamp,
            price=price,
            quantity=quantity,
            realised_profit=realised_profit,
            fee=fee,
        )
        if self._record_executions:
            self._executions.append(execution)
        named = order.reason if describer is None else describer(execution)
        return reported_reason(order, named)

    def _record_liquidation(
        self,
        timestamp: datetime,
        symbol: Symbol,
        side: PositionSide,
        price: float,
        quantity: float,
        margin: float,
    ) -> None:
        if not self._record_executions:
            return
        self._executions.append(
            Execution(
                execution_id=new_execution_id(),
                order_id=new_order_id(),
                symbol=symbol,
                side=side.closing_side,
                price=price,
                quantity=quantity,
                timestamp=timestamp,
                kind=OrderType.LIQUIDATION,
                realised_profit=-margin,
                fee=0.0,
            )
        )

    def _close_position(self, position: SimulatedPosition) -> None:
        """Drop a fully-closed position and cancel any guard orders still
        resting for it.
        """
        del self._positions[position.symbol]
        self._order_book.resolve_group(position.symbol, position.group_id)

    def _update_balance(
        self,
        *,
        locked_delta: float = 0.0,
        total_delta: float = 0.0,
    ) -> None:
        """
        Args:
            locked_delta: Quantity to add to locked (can be negative).
            total_delta: Quantity to add to total (can be negative).
        """
        self._balances[self._margin_currency] = self._balances[
            self._margin_currency
        ].update_copy(locked_delta=locked_delta, total_delta=total_delta)

    def _check_liquidations(self, timestamp: datetime, ohlcvs: OHLCVsBySymbol) -> None:
        for symbol, position in list(self._positions.items()):
            if symbol not in ohlcvs:
                continue

            ohlcv = ohlcvs[symbol]
            low, high = ohlcv.low, ohlcv.high

            is_long_liquidated = (
                position.side == PositionSide.LONG and low <= position.liquidation_price
            )
            is_short_liquidated = (
                position.side == PositionSide.SHORT
                and high >= position.liquidation_price
            )

            if is_long_liquidated or is_short_liquidated:
                self._liquidate_position(timestamp, symbol, position)

    def _liquidate_position(
        self, timestamp: datetime, symbol: Symbol, position: SimulatedPosition
    ) -> None:
        liquidation_price = position.liquidation_price

        margin = position.locked_margin

        self._update_balance(
            locked_delta=-margin,
            total_delta=-margin,
        )

        fill_type: FillType = (
            "liquidate_long"
            if position.side == PositionSide.LONG
            else "liquidate_short"
        )

        liquidated_quantity = to_quantity(position.quantity_steps)
        self._fill_recorder.record_fill(
            timestamp,
            str(symbol),
            position.side,
            liquidated_quantity,
            liquidated_quantity,
            liquidation_price,
            0.0,
            fill_type,
            reason="liquidation",
            rate=self.quote_conversion_rate(symbol),
        )
        self._record_liquidation(
            timestamp,
            symbol,
            position.side,
            liquidation_price,
            liquidated_quantity,
            margin,
        )

        self._close_position(position)

        logger.warning(
            f"Position liquidated: {position.side.value} {symbol} "
            f"at {liquidation_price:.6f} (entry: {position.average_entry_price:.6f}, "
            f"leverage: {position.leverage:.1f}x)"
        )

        self._trades_occurred_this_tick = True


def _determine_position_side(order: LimitOrder | MarketOrder) -> PositionSide:
    return PositionSide.LONG if order.side == OrderSide.BUY else PositionSide.SHORT


def _get_entry_fill_type(created: bool, side: PositionSide) -> FillType:
    if created:
        if side == PositionSide.LONG:
            return "enter_long"
        if side == PositionSide.SHORT:
            return "enter_short"
    else:
        if side == PositionSide.LONG:
            return "add_to_long"
        if side == PositionSide.SHORT:
            return "add_to_short"
    raise ValueError(f"Invalid state: created={created}, side={side}")


def _get_exit_fill_type(closed: bool, side: PositionSide) -> FillType:
    if closed:
        if side == PositionSide.LONG:
            return "exit_long"
        if side == PositionSide.SHORT:
            return "exit_short"
    else:
        if side == PositionSide.LONG:
            return "reduce_long"
        if side == PositionSide.SHORT:
            return "reduce_short"
    raise ValueError(f"Invalid state: closed={closed}, side={side}")


def _execution_of(
    order: LimitOrder | MarketOrder | StopLossOrder | TakeProfitOrder,
    *,
    timestamp: datetime,
    price: float,
    quantity: float,
    realised_profit: float,
    fee: float,
) -> Execution:
    return Execution(
        execution_id=new_execution_id(),
        order_id=order.order_id,
        symbol=order.symbol,
        side=order.side,
        price=price,
        quantity=quantity,
        timestamp=timestamp,
        kind=_fill_kind(order),
        realised_profit=realised_profit,
        fee=fee,
        client_order_id=order.client_order_id,
    )


def _fill_kind(
    order: LimitOrder | MarketOrder | StopLossOrder | TakeProfitOrder,
) -> OrderType:
    """A trigger-fired order's kind matches what a live venue reports for the
    same fill.
    """
    if isinstance(order, LimitOrder | MarketOrder) and order.triggered:
        return OrderType.TRIGGER
    return order.kind
