from robottraderslab_core._internal import resolve_on_read

_MODULE_BY_NAME = {
    "CandleStep": "robottraderslab_core._internal.candle_grid",
    "CandleWindows": "robottraderslab_core._internal.ohlcv",
    "OHLCVFetcher": "robottraderslab_core._internal.interfaces",
    "OHLCVRequirement": "robottraderslab_core._internal.ohlcv",
    "OHLCVRow": "robottraderslab_core._internal.ohlcv",
    "OHLCVsBySymbol": "robottraderslab_core._internal.ohlcv",
    "OHLCVsByTimeframeAndSymbol": "robottraderslab_core._internal.ohlcv.ohlcvs",
    "OhlcvValidationError": "robottraderslab_core._internal.exceptions",
    "SymbolStr": "robottraderslab_core._internal.symbol",
    "TIMEFRAMES": "robottraderslab_core._internal.timeframes",
    "TimeframeSnapshot": "robottraderslab_core._internal.ohlcv",
    "candle_closes_at": "robottraderslab_core._internal.timeframes",
    "candle_stamp_before": "robottraderslab_core._internal.timeframes",
    "candle_stamps": "robottraderslab_core._internal.candle_grid",
    "candle_step": "robottraderslab_core._internal.candle_grid",
    "candle_windows": "robottraderslab_core._internal.ohlcv",
    "candles_back": "robottraderslab_core._internal.candle_grid",
    "last_per_month": "robottraderslab_core._internal.candle_grid",
    "newest_settled_candle_start": "robottraderslab_core._internal.timeframes",
    "to_seconds": "robottraderslab_core._internal.timeframes",
}

__all__ = sorted(_MODULE_BY_NAME)

__getattr__, __dir__ = resolve_on_read(__name__, _MODULE_BY_NAME)
