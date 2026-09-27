import inspect

import pytest

import robottraderslab
from robottraderslab import exceptions, exchanges, indicators, strategies
from robottraderslab.strategies import futures

GATES = [
    robottraderslab,
    strategies,
    futures,
    exchanges,
    exceptions,
    indicators,
]


@pytest.mark.parametrize("gate", GATES, ids=lambda gate: gate.__name__)
def test_every_advertised_symbol_resolves(gate):
    missing = [name for name in gate.__all__ if not hasattr(gate, name)]

    assert not missing


def test_root_gate():
    assert sorted(robottraderslab.__all__) == [
        "BacktestOutputs",
        "BotConfig",
        "GridSearchResults",
        "Symbol",
        "TimeFrame",
        "TradeFilter",
        "run_backtest",
        "run_grid_search",
        "run_livebot",
    ]


def test_strategies_gate():
    assert sorted(strategies.__all__) == [
        "AccountRequirements",
        "AccountSnapshot",
        "AccountSnapshots",
        "ActionBuilder",
        "Balance",
        "BaseExchangeAction",
        "BookKeeper",
        "Candles",
        "ChartLine",
        "Currency",
        "Execution",
        "FillDescriber",
        "FillEffect",
        "FillSource",
        "MarketType",
        "OHLCVRequirements",
        "OHLCVs",
        "OnFillRead",
        "OrderFill",
        "OrderProtocol",
        "OrderSide",
        "OrderType",
        "PlacedOrder",
        "PositionSide",
        "PositionTracker",
        "Profile",
        "ProfileProtocol",
        "ProfileStrategy",
        "ProfileTag",
        "StopLoss",
        "StrategyProtocol",
        "StrategyRequirements",
        "SymbolTimeframe",
        "TakeProfit",
        "TrackedPosition",
        "TrackerRequirements",
        "TrackingId",
        "TradingMode",
        "TradingSystem",
        "profile_tag",
        "tag_of",
    ]


def test_futures_strategies_gate():
    assert sorted(futures.__all__) == [
        "AvailableBalanceRatio",
        "BatchableOrder",
        "BatchableOrderAction",
        "CancelOrderByIdAction",
        "CancelOrdersForSymbolAction",
        "ClosePositionAction",
        "EquityRatio",
        "FuturesAccount",
        "FuturesCapabilities",
        "FuturesCloseBuilder",
        "FuturesLimitOrderAction",
        "FuturesMarketOrderAction",
        "FuturesOrderBatchAction",
        "FuturesOrderBuilder",
        "FuturesOrderModifyAction",
        "FuturesPositionTarget",
        "Margin",
        "MarginMode",
        "MarginSettings",
        "Notional",
        "OrderModification",
        "PositionSnapshot",
        "Quantity",
        "RiskRatio",
        "SetLeverageAction",
        "SetMarginModeAction",
        "SizingRule",
        "TimeInForce",
        "TotalBalanceRatio",
        "UpdatePositionStopLossAction",
        "UpdatePositionTakeProfitAction",
    ]


def test_exchanges_gate():
    assert sorted(exchanges.__all__) == [
        "ActionResult",
        "Balance",
        "BaseExchangeAction",
        "CUSTOM_TAG_BUDGET",
        "ChainedRateLimiter",
        "Currency",
        "DEFAULT_HTTP_TIMEOUT_SECONDS",
        "Execution",
        "FillEffect",
        "FillSource",
        "FuturesCapabilities",
        "FuturesExchangeBase",
        "FuturesExchangeProtocol",
        "HttpResponse",
        "HttpTransport",
        "JsonLike",
        "MarginMode",
        "MarginSettings",
        "MarketType",
        "OHLCVProviderProtocol",
        "OhlcvAdapterProtocol",
        "OhlcvData",
        "OnFillRead",
        "OrderFill",
        "OrderModifyRequest",
        "OrderOutcome",
        "OrderPlacement",
        "OrderProtocol",
        "OrderRequest",
        "OrderSide",
        "OrderType",
        "PlacedOrder",
        "PlacementReserve",
        "PositionProtocol",
        "PositionSide",
        "PositionSnapshot",
        "RateLimiter",
        "RateLimiterProtocol",
        "RequestSigner",
        "RequestToSign",
        "SharedRateLimiter",
        "SharedRequestWindow",
        "SharedTokenBucket",
        "SharedVenueWindow",
        "StopLoss",
        "TakeProfit",
        "TimeInForce",
        "VenueFill",
        "client_order_id_carrying",
        "fetch_pages",
        "no_candles",
        "page_bounds",
        "tag_of",
        "to_milliseconds",
        "without_the_open_candle",
    ]


def test_exceptions_gate():
    assert sorted(exceptions.__all__) == [
        "DataError",
        "DownloadError",
        "ExchangeConnectionError",
        "ExchangeCriticalError",
        "ExchangeRecoverableError",
        "ExchangeTransientError",
        "InvalidTimeframeError",
        "MissingOhlcvDataError",
        "NoOpenPositionError",
        "RateLimitExceededError",
        "StrategyCriticalError",
    ]


def test_indicators_gate():
    assert sorted(indicators.__all__) == [
        "MAType",
        "Trix",
        "donchian_midline",
        "ema",
        "moving_average",
        "sma",
        "trix",
        "wma",
    ]


def _public_members(cls: type) -> list[str]:
    return sorted(name for name in dir(cls) if not name.startswith("_"))


def test_order_builder_surface():
    assert _public_members(futures.FuturesOrderBuilder) == [
        "build",
        "extra_fields",
        "limit",
        "reason",
        "size",
        "stop_loss",
        "tag",
        "take_profit",
        "trigger",
        "when_filled",
    ]


def test_position_target_surface():
    assert _public_members(futures.FuturesPositionTarget) == ["quantity", "size"]


def test_position_tracker_surface():
    assert _public_members(strategies.PositionTracker) == ["get"]


def test_strategy_requirements_take_no_argument():
    assert list(inspect.signature(strategies.StrategyRequirements).parameters) == []


def test_tracker_declaration_signature():
    parameters = inspect.signature(strategies.TrackerRequirements.add).parameters

    assert [
        (name, parameter.kind, parameter.default)
        for name, parameter in parameters.items()
    ] == [
        ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        ("account", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
        ("symbols", inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.empty),
        ("tags", inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.empty),
    ]
