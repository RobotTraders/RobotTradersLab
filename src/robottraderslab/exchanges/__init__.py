import importlib
from typing import Any

_MODULE_BY_NAME = {
    "ActionResult": "robottraderslab._core.actions.members",
    "Balance": "robottraderslab._core.balance",
    "BaseExchangeAction": "robottraderslab._core.actions.members",
    "CUSTOM_TAG_BUDGET": "robottraderslab._core.actions.members",
    "ChainedRateLimiter": "robottraderslab._core.rate_limiting",
    "Currency": "robottraderslab._core.currency",
    "DEFAULT_HTTP_TIMEOUT_SECONDS": "robottraderslab._core.exchanges",
    "Execution": "robottraderslab._core.order",
    "FillEffect": "robottraderslab._core.order",
    "FillSource": "robottraderslab._core.order",
    "FuturesCapabilities": "robottraderslab.futures",
    "FuturesExchangeBase": "robottraderslab.futures",
    "FuturesExchangeProtocol": "robottraderslab.futures",
    "HttpResponse": "robottraderslab._core.exchanges",
    "HttpTransport": "robottraderslab._core.exchanges",
    "JsonLike": "robottraderslab._core.exchanges",
    "MarginMode": "robottraderslab._core.margin",
    "MarginSettings": "robottraderslab._core.margin",
    "MarketType": "robottraderslab._core.market",
    "OHLCVProviderProtocol": "robottraderslab._core.interfaces",
    "OhlcvAdapterProtocol": "robottraderslab._core.exchanges",
    "OhlcvData": "robottraderslab._core.exchanges",
    "OnFillRead": "robottraderslab._core.order",
    "OrderFill": "robottraderslab._core.order",
    "OrderModifyRequest": "robottraderslab._core.order",
    "OrderOutcome": "robottraderslab._core.actions.members",
    "OrderPlacement": "robottraderslab._core.order",
    "OrderProtocol": "robottraderslab._core.order",
    "OrderRequest": "robottraderslab._core.order",
    "OrderSide": "robottraderslab._core.order",
    "OrderType": "robottraderslab._core.order",
    "PlacedOrder": "robottraderslab._core.order",
    "PlacementReserve": "robottraderslab._core.fees",
    "PositionProtocol": "robottraderslab._core.position",
    "PositionSide": "robottraderslab._core.position",
    "PositionSnapshot": "robottraderslab._core.position",
    "RateLimiter": "robottraderslab._core.rate_limiting",
    "RateLimiterProtocol": "robottraderslab._core.rate_limiting",
    "RequestSigner": "robottraderslab._core.exchanges",
    "RequestToSign": "robottraderslab._core.exchanges",
    "SharedRateLimiter": "robottraderslab._core.rate_limiting",
    "SharedRequestWindow": "robottraderslab._core.rate_limiting",
    "SharedTokenBucket": "robottraderslab._core.rate_limiting",
    "SharedVenueWindow": "robottraderslab._core.rate_limiting",
    "StopLoss": "robottraderslab._core.order",
    "TakeProfit": "robottraderslab._core.order",
    "TimeInForce": "robottraderslab._core.order",
    "VenueFill": "robottraderslab._core.order",
    "client_order_id_carrying": "robottraderslab._core.actions.members",
    "fetch_pages": "robottraderslab._core.exchanges",
    "no_candles": "robottraderslab._core.exchanges",
    "page_bounds": "robottraderslab._core.exchanges",
    "tag_of": "robottraderslab._core.actions.members",
    "to_milliseconds": "robottraderslab._core.timeframes",
    "without_the_open_candle": "robottraderslab._core.exchanges",
}

__all__ = sorted(_MODULE_BY_NAME)


def __getattr__(name: str) -> Any:
    """The gate spans packages a command may never reach for, and one of
    them carries the candle libraries, so a name resolves the module that
    defines it at the moment it is read.
    """
    module = _MODULE_BY_NAME.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    return __all__
