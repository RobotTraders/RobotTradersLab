import math
from collections.abc import Mapping

from .actions.members import (
    carries_this_bot,
    next_client_order_id,
    tag_of,
)
from .order import (
    RESTING_ENTRY_KINDS,
    OrderModifyRequest,
    OrderProtocol,
    OrderRequest,
    OrderSide,
    StopLoss,
    TakeProfit,
    TimeInForce,
    TriggerType,
)
from .price import validate_price


def modify_request(
    resting: OrderProtocol,
    *,
    quantity: float | None = None,
    limit_price: float | None = None,
    trigger_price: float | None = None,
    stop_loss: float | None = None,
    take_profit: float | None = None,
) -> OrderModifyRequest:
    """The read reports no venue fields, so the replacement carries none.

    Args:
        resting: The order as the snapshot's open orders report it.
        quantity: The new size in the base currency.
        limit_price: None leaves the price as the order rests.
        trigger_price: None leaves the trigger as the order rests.
        stop_loss: None leaves the order's level; a level named on an order
            without one adds it.
        take_profit: None leaves the order's level; a level named on an order
            without one adds it.

    Raises:
        ValueError: If no field is named; the order is a stop-loss, a
            take-profit, a trailing stop, a liquidation or a market order;
            the read states no side, quantity or remaining quantity; a limit
            price or a trigger price is named on an order with none; the
            quantity does not exceed what already filled; a value is not
            finite; or a price is not positive.
    """
    named = _named_fields(
        quantity=quantity,
        limit_price=limit_price,
        trigger_price=trigger_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )
    _require_a_field_named(resting, named)
    _require_named_values_finite(resting, named)
    _require_a_reshapeable_kind(resting, named)
    side, size, remaining = _require_the_read_states_side_and_size(resting, named)
    _require_limit_and_trigger_named_on_the_order(resting, named)
    _require_named_quantity_above_filled(resting, size - remaining, named)
    _validate_named_prices(resting, named)
    return OrderModifyRequest(
        order_id=resting.order_id,
        order=_replacement(resting, side, size, named),
    )


def _named_fields(**fields: float | None) -> dict[str, float]:
    return {field: value for field, value in fields.items() if value is not None}


def _require_a_field_named(resting: OrderProtocol, named: Mapping[str, float]) -> None:
    if not named:
        raise ValueError(
            f"{resting.symbol}: order {resting.order_id} is modified with no "
            "field named; name quantity, limit_price, trigger_price, stop_loss "
            "or take_profit"
        )


def _require_named_values_finite(
    resting: OrderProtocol, named: Mapping[str, float]
) -> None:
    for field, value in named.items():
        if not math.isfinite(value):
            raise ValueError(_cannot_change(resting, field, f"{value} is not finite"))


def _require_a_reshapeable_kind(
    resting: OrderProtocol, named: Mapping[str, float]
) -> None:
    if resting.kind not in RESTING_ENTRY_KINDS:
        raise ValueError(
            _cannot_change(
                resting,
                ", ".join(named),
                f"a {resting.kind} order is not reshaped by its fields",
            )
        )


def _require_the_read_states_side_and_size(
    resting: OrderProtocol, named: Mapping[str, float]
) -> tuple[OrderSide, float, float]:
    side, quantity, remaining = (
        resting.side,
        resting.quantity,
        resting.remaining_quantity,
    )
    if side is None or quantity is None or remaining is None:
        raise ValueError(
            _cannot_change(
                resting,
                ", ".join(named),
                "the read states no side or no quantity to keep",
            )
        )
    return side, quantity, remaining


def _require_limit_and_trigger_named_on_the_order(
    resting: OrderProtocol, named: Mapping[str, float]
) -> None:
    resting_levels = {
        "limit_price": resting.limit_price,
        "trigger_price": resting.trigger_price,
    }
    for field, level in resting_levels.items():
        if field in named and level is None:
            raise ValueError(
                _cannot_change(
                    resting,
                    field,
                    f"the order has no {field.replace('_', ' ')}; a change of "
                    "shape is a whole order",
                )
            )


def _require_named_quantity_above_filled(
    resting: OrderProtocol, filled: float, named: Mapping[str, float]
) -> None:
    quantity = named.get("quantity")
    if quantity is not None and not quantity > filled:
        raise ValueError(
            _cannot_change(
                resting,
                "quantity",
                f"{quantity} does not exceed the {filled} already filled",
            )
        )


def _validate_named_prices(resting: OrderProtocol, named: Mapping[str, float]) -> None:
    for field, value in named.items():
        if field == "quantity":
            continue
        try:
            validate_price(field, value)
        except ValueError as error:
            raise ValueError(_cannot_change(resting, field, str(error))) from error


def _replacement(
    resting: OrderProtocol, side: OrderSide, size: float, named: Mapping[str, float]
) -> OrderRequest:
    return OrderRequest(
        symbol=resting.symbol,
        side=side,
        quantity=named.get("quantity", size),
        limit_price=named.get("limit_price", resting.limit_price),
        time_in_force=resting.time_in_force or TimeInForce.GTC,
        reduce_only=bool(resting.reduce_only),
        stop_loss=_protection(
            StopLoss,
            named.get("stop_loss", resting.stop_loss_price),
            resting.stop_loss_trigger_type,
        ),
        take_profit=_protection(
            TakeProfit,
            named.get("take_profit", resting.take_profit_price),
            resting.take_profit_trigger_type,
        ),
        trigger_price=named.get("trigger_price", resting.trigger_price),
        trigger_type=resting.trigger_type or TriggerType.DEFAULT,
        client_order_id=_replacement_client_order_id(resting.client_order_id),
    )


def _protection[P: (StopLoss, TakeProfit)](
    protection: type[P], level: float | None, trigger_type: TriggerType | None
) -> P | None:
    if level is None:
        return None
    return protection(
        trigger_price=level, trigger_type=trigger_type or TriggerType.DEFAULT
    )


def _replacement_client_order_id(client_order_id: str | None) -> str | None:
    if not carries_this_bot(client_order_id):
        return None
    return next_client_order_id(tag_of(client_order_id))


def _cannot_change(resting: OrderProtocol, fields: str, why: str) -> str:
    return f"{resting.symbol}: order {resting.order_id} cannot change {fields}: {why}"
