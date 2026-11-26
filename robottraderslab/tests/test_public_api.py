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
        "TriggerType",
        "UpdatePositionStopLossAction",
        "UpdatePositionTakeProfitAction",
        "UpdatePositionTrailingStopAction",
    ]


def test_exchanges_gate():
    assert sorted(exchanges.__all__) == [
        "AccountState",
        "AccountStateRequest",
        "ActionResult",
        "Balance",
        "BaseExchangeAction",
        "CUSTOM_TAG_BUDGET",
        "ChainedRateLimiter",
        "Closure",
        "Currency",
        "Execution",
        "FillEffect",
        "FillSource",
        "FuturesCapabilities",
        "FuturesExchangeBase",
        "FuturesExchangeProtocol",
        "HttpResponse",
        "HttpTransport",
        "Instrument",
        "JsonLike",
        "MarginMode",
        "MarginSettings",
        "MarketType",
        "NO_VENUE_PARAMS",
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
        "PositionSide",
        "PositionSnapshot",
        "Quote",
        "RateLimiter",
        "RateLimiterProtocol",
        "RequestSigner",
        "RequestToSign",
        "SharedAnnouncedWindow",
        "SharedRequestWindow",
        "SharedTokenBucket",
        "SharedVenueWindow",
        "StopLoss",
        "TakeProfit",
        "TimeInForce",
        "TradingStatus",
        "TriggerType",
        "VenueFill",
        "VenueParams",
        "client_order_id_carrying",
        "describe_order",
        "fetch_pages",
        "lacks_a_closed_candle",
        "modify_request",
        "next_client_order_id",
        "no_candles",
        "page_bounds",
        "process_bot_bytes",
        "replace_trailing_stops",
        "replace_trailing_stops_with",
        "report_order_refusal",
        "reported_trigger_type",
        "retry_on_transient",
        "tag_of",
        "to_milliseconds",
        "trailing_percent",
        "venue_trigger_type",
        "with_resting_reduce_only",
        "with_venue_params",
        "without_the_open_candle",
    ]


def test_exceptions_gate():
    assert sorted(exceptions.__all__) == [
        "BotCriticalError",
        "DataError",
        "DownloadError",
        "ExchangeConnectionError",
        "ExchangeCriticalError",
        "ExchangeRecoverableError",
        "ExchangeTransientError",
        "InsufficientMarginError",
        "InvalidTimeframeError",
        "MissingOhlcvDataError",
        "NoOpenPositionError",
        "ProfileConfigError",
        "ProfileError",
        "RateLimitExceededError",
        "SymbolNotListedError",
    ]


def test_strategy_critical_error_is_the_bot_critical_error():
    assert exceptions.StrategyCriticalError is exceptions.BotCriticalError


def test_indicators_gate():
    assert sorted(indicators.__all__) == [
        "MAType",
        "Trix",
        "crossover",
        "crossunder",
        "donchian_midline",
        "ema",
        "moving_average",
        "sma",
        "trix",
        "value_periods_ago",
        "wma",
    ]


def _public_members(cls: type) -> list[str]:
    return sorted(name for name in dir(cls) if not name.startswith("_"))


def test_order_builder_surface():
    assert _public_members(futures.FuturesOrderBuilder) == [
        "build",
        "extra_fields",
        "limit",
        "params",
        "reason",
        "size",
        "stop_loss",
        "tag",
        "take_profit",
        "trigger",
        "when_filled",
    ]


def test_futures_exchange_protocol_surface():
    assert _public_members(exchanges.FuturesExchangeProtocol) == [
        "cancel_order_by_id",
        "cancel_orders_for_symbol",
        "capabilities",
        "get_account_state",
        "get_balances",
        "get_equity",
        "get_executions_since",
        "get_instrument",
        "get_margin_settings",
        "get_open_orders",
        "get_open_positions",
        "get_price_increment",
        "get_quote",
        "get_quote_conversion_rate",
        "modify_orders",
        "place_limit_order",
        "place_market_order",
        "place_orders",
        "place_trailing_stop_order",
        "placement_requirement_rate",
        "placement_reserve_rate",
        "round_price",
        "round_quantity",
        "set_leverage",
        "set_margin_mode",
        "update_position_stop_loss",
        "update_position_take_profit",
        "update_position_trailing_stop",
        "update_position_trailing_stop_ratio",
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
        ("netted", inspect.Parameter.KEYWORD_ONLY, False),
    ]
