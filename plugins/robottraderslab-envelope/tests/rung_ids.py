from robottraderslab.strategies import OrderSide

PROCESS_TAG = "fakeprocess99-1"


def rung_client_order_id(timeframe: str, side: OrderSide, rung: int) -> str:
    letter = "l" if side is OrderSide.BUY else "s"
    return f"{PROCESS_TAG}-{timeframe}-{letter}{rung}"
