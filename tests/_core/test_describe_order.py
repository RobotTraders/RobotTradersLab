import pytest

from robottraderslab import Symbol
from robottraderslab.exchanges import OrderRequest, OrderSide, describe_order

SOL = Symbol("SOL", "USD", "USD")


@pytest.mark.parametrize(
    ("side", "reduce_only", "limit_price", "trigger_price", "expected"),
    [
        (OrderSide.BUY, False, 96.38, None, "long limit entry at 96.38"),
        (OrderSide.SELL, False, None, 111.43, "short trigger entry at 111.43"),
        (
            OrderSide.BUY,
            False,
            96.0,
            97.0,
            "long stop-limit entry at 96 (trigger 97)",
        ),
        (OrderSide.SELL, False, None, None, "short market entry"),
        (OrderSide.SELL, True, None, None, "market close of the long"),
        (OrderSide.BUY, True, None, 0.0729, "trigger close of the short at 0.0729"),
    ],
)
def test_names_the_order_by_side_kind_and_price(
    side, reduce_only, limit_price, trigger_price, expected
):
    order = OrderRequest(
        symbol=SOL,
        side=side,
        quantity=1.0,
        limit_price=limit_price,
        trigger_price=trigger_price,
        reduce_only=reduce_only,
    )

    assert describe_order(order) == expected


def test_reads_a_small_price_without_trailing_float_noise():
    order = OrderRequest(
        symbol=SOL, side=OrderSide.BUY, quantity=1.0, limit_price=2.2589699999999998e-05
    )

    assert describe_order(order) == "long limit entry at 2.25897e-05"
