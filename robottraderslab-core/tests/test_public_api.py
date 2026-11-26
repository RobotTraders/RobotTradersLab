import ast
from pathlib import Path

import pytest

import robottraderslab_core
from robottraderslab_core import (
    _internal,
    engine,
    exceptions,
    exchanges,
    futures,
    ohlcv,
    strategies,
)

GATES = [
    robottraderslab_core,
    _internal,
    exchanges,
    exceptions,
    futures,
    strategies,
    ohlcv,
    engine,
]

STRATEGY_SIDE_FUTURES_NAMES = [
    "FuturesAccount",
    "FuturesCloseBuilder",
    "FuturesOrderBuilder",
    "FuturesPositionTarget",
]


@pytest.mark.parametrize("gate", GATES, ids=lambda gate: gate.__name__)
def test_every_advertised_symbol_resolves(gate):
    missing = [name for name in gate.__all__ if not hasattr(gate, name)]

    assert not missing


def test_root_gate():
    assert sorted(robottraderslab_core.__all__) == ["Symbol", "TimeFrame"]


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


def test_futures_gate():
    assert sorted(futures.__all__) == [
        "BatchableOrder",
        "BatchableOrderAction",
        "CancelOrderByIdAction",
        "CancelOrdersForSymbolAction",
        "ClosePositionAction",
        "FuturesCapabilities",
        "FuturesExchangeBase",
        "FuturesExchangeProtocol",
        "FuturesLimitOrderAction",
        "FuturesMarketOrderAction",
        "FuturesOrderBatchAction",
        "FuturesOrderModifyAction",
        "OrderModification",
        "SetLeverageAction",
        "SetMarginModeAction",
        "UpdatePositionStopLossAction",
        "UpdatePositionTakeProfitAction",
        "UpdatePositionTrailingStopAction",
        "close_held_position",
        "modify_in_order",
        "place_in_order",
        "read_account_state",
    ]


def test_strategies_gate():
    assert sorted(strategies.__all__) == [
        "AccountRequirements",
        "AccountSnapshot",
        "AccountSnapshots",
        "ActionBuilder",
        "AvailableBalanceRatio",
        "BookKeeper",
        "EquityRatio",
        "FillDescriber",
        "Margin",
        "Notional",
        "OHLCVRequirements",
        "OHLCVs",
        "PositionTracker",
        "ProfileTag",
        "Quantity",
        "RiskRatio",
        "SizingRule",
        "StrategyProtocol",
        "StrategyRequirements",
        "TotalBalanceRatio",
        "TrackedPosition",
        "TrackerRequirements",
        "TrackingId",
        "profile_tag",
    ]


def test_ohlcv_gate():
    assert sorted(ohlcv.__all__) == [
        "CandleStep",
        "CandleWindows",
        "OHLCVFetcher",
        "OHLCVRequirement",
        "OHLCVRow",
        "OHLCVsBySymbol",
        "OHLCVsByTimeframeAndSymbol",
        "OhlcvValidationError",
        "SymbolStr",
        "TIMEFRAMES",
        "TimeframeSnapshot",
        "candle_closes_at",
        "candle_stamp_before",
        "candle_stamps",
        "candle_step",
        "candle_windows",
        "candles_back",
        "last_per_month",
        "newest_settled_candle_start",
        "to_seconds",
    ]


def test_engine_gate():
    assert sorted(engine.__all__) == [
        "AccountProtocol",
        "AccountRequirement",
        "ActionID",
        "BASE_DELAY_SECONDS",
        "BRACKET_SUFFIXES",
        "DEFAULT_FEE_MODE",
        "ENGINE_TAGS",
        "EXECUTOR_LOGGER_NAME",
        "FLATTEN_TAG",
        "FeeMode",
        "Halt",
        "MAX_ATTEMPTS",
        "Notifications",
        "OnOrderFilled",
        "OnOrderPlaced",
        "OrderReplacement",
        "PROTECTION_KINDS",
        "PlacementNotifier",
        "PositionProtectionAction",
        "ProtectionKind",
        "ProtectionVenue",
        "RESTING_ENTRY_KINDS",
        "StepCount",
        "VENUE_FIRED_SOURCES",
        "attribute_fill_effects",
        "bot_leading_bytes",
        "bot_segment",
        "carries_this_bot",
        "drain_tasks",
        "engine_tag_refusal",
        "execute_setting",
        "execute_trading_actions",
        "fetch_account_snapshots",
        "finished",
        "is_unlabelled_protection",
        "name_process",
        "on_symbol",
        "position_steps",
        "profile_identity",
        "quantity_from_total_balance",
        "read_instruments",
        "release_lock",
        "require_not_a_bracket_suffix",
        "require_within_budget",
        "run_async",
        "settle_booked_fill",
        "signed_position",
        "signed_steps",
        "sizing_rules",
        "to_quantity",
        "to_step_count",
        "try_lock",
        "validate_closing_ratio",
        "validate_price",
        "validate_ratio",
    ]


@pytest.mark.parametrize("name", STRATEGY_SIDE_FUTURES_NAMES)
def test_futures_gate_lacks_the_strategy_side_names(name):
    assert name not in futures.__all__
    assert not hasattr(futures, name)


@pytest.mark.parametrize(
    "gate", [strategies, ohlcv, engine, exchanges], ids=lambda gate: gate.__name__
)
def test_a_lazy_gate_has_no_name_it_does_not_list(gate):
    with pytest.raises(AttributeError, match="no attribute 'NotAName'"):
        gate.NotAName


@pytest.mark.parametrize(
    "gate", [strategies, ohlcv, engine, exchanges], ids=lambda gate: gate.__name__
)
def test_the_stub_of_a_lazy_gate_lists_the_names_of_the_gate(gate):
    stub = ast.parse(Path(gate.__file__).with_suffix(".pyi").read_text())
    stubbed = [
        ast.literal_eval(node.value)
        for node in stub.body
        if isinstance(node, ast.Assign) and node.targets[0].id == "__all__"
    ]

    assert sorted(stubbed[0]) == sorted(gate.__all__)
