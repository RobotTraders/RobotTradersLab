import logging
from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Any, Self, cast

from robottraderslab._core import DEFAULT_FEE_MODE, FeeMode, PlacementReserve, Symbol
from robottraderslab.exceptions import ExchangeRecoverableError, NoOpenPositionError
from robottraderslab.exchanges import (
    Balance,
    Currency,
    Execution,
    FuturesExchangeProtocol,
    MarginMode,
    MarginSettings,
    OrderModifyRequest,
    OrderProtocol,
    OrderRequest,
    OrderSide,
    PlacedOrder,
    PositionSnapshot,
    StopLoss,
    TakeProfit,
    TimeInForce,
    VenueFill,
)
from robottraderslab.futures import modify_in_order, place_in_order

from .fees import FeeRates, fee_model_for
from .fill_recorder import CacheFillRecorder
from .futures_simulation_engine import FuturesSimulationEngine
from .order_models import (
    LimitOrder,
    MarketOrder,
    StopLossOrder,
    TakeProfitOrder,
    TriggerOrder,
)

logger = logging.getLogger(__name__)


class SimulatedFuturesExchange(FuturesExchangeProtocol):
    """Delegates every operation to the simulation engine driving it."""

    def __init__(
        self,
        simulation_engine: FuturesSimulationEngine,
        fill_recorder: CacheFillRecorder,
    ):
        self.simulation_engine = simulation_engine
        self.fill_recorder = fill_recorder
        self.placement_reserve_rate = simulation_engine.placement_reserve_rate

    @classmethod
    def create_from_settings(
        cls,
        *,
        initial_balance: dict[Currency, float],
        maker_fee_rate: float,
        taker_fee_rate: float,
        fee_mode: FeeMode = DEFAULT_FEE_MODE,
        placement_reserve: PlacementReserve = PlacementReserve(),
        **_kwargs: Any,
    ) -> Self:
        """
        Args:
            fee_mode: Whether a fill's fee comes out of the position it opens
                or is debited beside it as cash.
            placement_reserve: What the modelled venue holds back at placement
                beyond the margin and the fee.

        Raises:
            ValueError: If the initial balance names more than one currency.
        """
        if len(initial_balance) != 1:
            raise ValueError("Futures market needs only one margin currency balance")
        margin_currency = list(initial_balance.keys())[0]

        fill_recorder = CacheFillRecorder()
        simulation_engine = FuturesSimulationEngine(
            initial_balance=initial_balance,
            fee_rates=FeeRates(maker=maker_fee_rate, taker=taker_fee_rate),
            fill_recorder=fill_recorder,
            margin_currency=margin_currency,
            fee_model=fee_model_for(fee_mode, placement_reserve),
        )
        return cls(simulation_engine, fill_recorder)

    def placement_requirement_rate(self, leverage: float) -> float:
        return self.simulation_engine.placement_requirement_rate(leverage)

    async def get_balances(self, symbols: Iterable[Symbol]) -> dict[Currency, Balance]:
        return self.simulation_engine.get_balances()

    async def get_equity(self, currency: Currency) -> float:
        return self.simulation_engine.get_equity(currency)

    async def get_margin_settings(
        self, symbols: Iterable[Symbol]
    ) -> dict[Symbol, MarginSettings]:
        """Return isolated margin at the engine's tracked leverage; the
        simulator does not model cross margin.
        """
        return {
            symbol: MarginSettings(
                leverage=self.simulation_engine.get_symbol_leverage(symbol),
                margin_mode=MarginMode.ISOLATED,
            )
            for symbol in symbols
        }

    async def get_open_orders(self, symbols: Iterable[Symbol]) -> list[OrderProtocol]:
        scope = set(symbols)
        orders = self.simulation_engine.open_orders
        return cast(
            list[OrderProtocol], [order for order in orders if order.symbol in scope]
        )

    async def get_open_positions(
        self, symbols: Iterable[Symbol]
    ) -> dict[Symbol, PositionSnapshot]:
        return self.simulation_engine.open_positions

    async def get_quote_conversion_rate(self, symbol: Symbol) -> float:
        """Rate converting the symbol's quote currency into the margin currency."""
        return self.simulation_engine.quote_conversion_rate(symbol)

    async def get_executions_since(
        self, since: datetime, symbols: Iterable[Symbol]
    ) -> list[Execution]:
        """Return every fill the run recorded, in booking order; recorded only
        when an account declared executions.
        """
        return self.simulation_engine.get_executions_since(since)

    async def set_leverage(self, symbol: Symbol, leverage: float) -> None:
        """
        Args:
            leverage: Leverage value, must be >= 1.0.

        Raises:
            ValueError: If leverage < 1.0
        """
        self.simulation_engine.set_symbol_leverage(symbol, leverage)

    async def set_margin_mode(self, symbol: Symbol, margin_mode: MarginMode) -> None:
        """The simulator models isolated margin only, so a cross request is
        warned about and isolated stays in force.
        """
        if margin_mode == MarginMode.CROSS:
            logger.warning(
                f"CROSS margin mode requested for {symbol}. Simulator only supports "
                f"ISOLATED margin for now, continuing with ISOLATED mode."
            )

    async def place_market_order(
        self,
        symbol: Symbol,
        side: OrderSide,
        quantity: float,
        reduce_only: bool = False,
        stop_loss: StopLoss | None = None,
        take_profit: TakeProfit | None = None,
        trigger_price: float | None = None,
        client_order_id: str | None = None,
        reason: str | None = None,
        extra_fields: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> PlacedOrder:
        """
        Args:
            trigger_price: Turns the order into a conditional one that fires
                once the market trades through it.
            extra_fields: Custom fields to store with fill.

        Raises:
            ExchangeRecoverableError: If the quantity is under one step.
        """
        self._require_one_step(quantity, symbol)
        market_order = MarketOrder(
            symbol=symbol,
            side=side,
            quantity=quantity,
            reduce_only=reduce_only,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=reason,
            client_order_id=client_order_id,
            extra_fields=extra_fields,
        )
        if trigger_price is not None:
            trigger_order = TriggerOrder(
                order=market_order, trigger_price=trigger_price
            )
            self.simulation_engine.add_order(trigger_order)
            return PlacedOrder(order_id=trigger_order.order_id)
        else:
            self.simulation_engine.add_order(market_order)
            order_id = market_order.order_id

            async def get_fill() -> VenueFill | None:
                return self.simulation_engine.get_last_fills().get(order_id)

            return PlacedOrder(order_id=order_id, get_fill=get_fill)

    async def place_limit_order(
        self,
        symbol: Symbol,
        side: OrderSide,
        quantity: float,
        price: float,
        reduce_only: bool = False,
        stop_loss: StopLoss | None = None,
        take_profit: TakeProfit | None = None,
        trigger_price: float | None = None,
        client_order_id: str | None = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        reason: str | None = None,
        extra_fields: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> PlacedOrder:
        """
        Args:
            trigger_price: Turns the order into a conditional one that fires
                once the market trades through it.
            time_in_force: Judged against the close the order is placed at,
                or against the first close its symbol reaches when none is
                known then: an `IOC` order fills there when its price reaches
                it and is cancelled otherwise, a `POST_ONLY` one is refused
                when its price reaches it and rests otherwise.
            extra_fields: Custom fields to store with fill.

        Raises:
            ExchangeRecoverableError: If the quantity is under one step, the
                placement reserve refuses a limit entry, or a `POST_ONLY`
                order would take liquidity at the close it is placed at.
        """
        self._require_one_step(quantity, symbol)
        limit_order = LimitOrder(
            symbol=symbol,
            side=side,
            quantity=quantity,
            limit_price=price,
            time_in_force=time_in_force,
            reduce_only=reduce_only,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=reason,
            client_order_id=client_order_id,
            extra_fields=extra_fields,
        )
        if trigger_price is not None:
            trigger_order = TriggerOrder(order=limit_order, trigger_price=trigger_price)
            self.simulation_engine.add_order(trigger_order)
            order_id = trigger_order.order_id
        else:
            self.simulation_engine.add_order(limit_order)
            order_id = limit_order.order_id

        return PlacedOrder(order_id=order_id)

    async def place_orders(
        self, requests: Sequence[OrderRequest]
    ) -> list[PlacedOrder | None]:
        """Queue each order in turn: nothing here waits, so overlap buys nothing."""
        return await place_in_order(self, requests)

    async def modify_orders(
        self, requests: Sequence[OrderModifyRequest]
    ) -> list[PlacedOrder | None]:
        """Reshape each resting order in turn: nothing here waits, so overlap
        buys nothing.
        """
        return await modify_in_order(self, requests)

    async def cancel_order_by_id(self, symbol: Symbol, order_id: str) -> None:
        """Cancel a single open order by id.

        Raises:
            ExchangeRecoverableError: If no order with `order_id` is found.
        """
        if not self.simulation_engine.cancel_order_by_id(symbol, order_id):
            raise ExchangeRecoverableError(f"No open order found with id: {order_id}")

    async def cancel_orders_for_symbol(self, symbol: Symbol) -> None:
        """Cancel all open orders for the symbol."""
        self.simulation_engine.cancel_orders_for_symbol(symbol)

    async def update_position_stop_loss(
        self, symbol: Symbol, trigger_price: float
    ) -> PlacedOrder:
        """Update the stop-loss trigger price of an open position.

        Raises:
            NoOpenPositionError: If no open position exists for `symbol`.
            ExchangeRecoverableError: If no active stop-loss order is attached.
        """
        self._get_open_position(symbol)
        stop_loss_order = self._find_active_tp_sl_order(symbol, StopLossOrder)
        if stop_loss_order is None:
            raise ExchangeRecoverableError(
                f"No active stop-loss order found for position on {symbol}"
            )
        stop_loss_order.trigger_price = trigger_price
        return PlacedOrder(order_id=stop_loss_order.order_id)

    async def update_position_take_profit(
        self, symbol: Symbol, trigger_price: float
    ) -> PlacedOrder:
        """Update the take-profit trigger price of an open position.

        Raises:
            NoOpenPositionError: If no open position exists for `symbol`.
            ExchangeRecoverableError: If no active take-profit order is attached.
        """
        self._get_open_position(symbol)
        take_profit_order = self._find_active_tp_sl_order(symbol, TakeProfitOrder)
        if take_profit_order is None:
            raise ExchangeRecoverableError(
                f"No active take-profit order found for position on {symbol}"
            )
        take_profit_order.trigger_price = trigger_price
        return PlacedOrder(order_id=take_profit_order.order_id)

    def _get_open_position(self, symbol: Symbol) -> PositionSnapshot:
        position = self.simulation_engine.open_positions.get(symbol)
        if position is None:
            raise NoOpenPositionError(f"No open position for symbol: {symbol}")
        return position

    def _find_active_tp_sl_order(
        self,
        symbol: Symbol,
        order_type: type[StopLossOrder] | type[TakeProfitOrder],
    ) -> StopLossOrder | TakeProfitOrder | None:
        for order in self.simulation_engine.open_orders:
            if isinstance(order, order_type) and order.symbol == symbol:
                return order
        return None

    def _require_one_step(self, quantity: float, symbol: Symbol) -> None:
        """A position of no steps has no entry price to average, and the count
        checked here is the one the fill counts, so no placed order can reach
        the engine with nothing to fill.

        Raises:
            ExchangeRecoverableError: If the quantity fills at no step.
        """
        if self.simulation_engine.order_steps(quantity) == 0:
            raise ExchangeRecoverableError(
                f"Order quantity {quantity} of {symbol} is under one quantity step"
            )
