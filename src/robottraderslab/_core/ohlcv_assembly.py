import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import cast

import pandas as pd

from .exceptions import ExchangeCriticalError, StrategyCriticalError
from .interfaces import OHLCVFetcher, OHLCVProviderProtocol
from .ohlcv.ohlcv_requirements import OHLCVRequirement, OHLCVRequirements
from .ohlcv.ohlcvs import OHLCVs, OHLCVsByTimeframeAndSymbol
from .retry import BASE_DELAY_SECONDS, MAX_ATTEMPTS, retry_on_transient
from .symbol import Symbol
from .symbol_timeframe import SymbolTimeframe
from .timeframes import TimeFrame

logger = logging.getLogger(__name__)

_Pair = tuple[Symbol, TimeFrame]


async def assemble_all_ohlcvs_from_requirements(
    provider: OHLCVProviderProtocol, requirements: OHLCVRequirements
) -> OHLCVs:
    """Assemble every declared pair, or raise the first failure.

    Every pair is fetched concurrently, one attempt each. Any failure aborts
    the assembly, so a backtest never runs on a portfolio other than the one
    configured.

    Args:
        requirements: Requirements collected from strategy.setup().

    Raises:
        StrategyCriticalError: If strategy declared no OHLCV requirements.
    """
    symbol_timeframes = _declared_symbol_timeframes(provider, requirements)
    pairs = _unique_pairs(symbol_timeframes)
    outcomes = await _fetch_pairs(provider, pairs, max_attempts=1, base_delay=0.0)

    frames: dict[_Pair, pd.DataFrame] = {}
    for pair, outcome in zip(pairs, outcomes, strict=True):
        if isinstance(outcome, BaseException):
            raise outcome
        frames[pair] = outcome
    return _assembled(frames, symbol_timeframes)


async def assemble_available_ohlcvs_from_requirements(
    provider: OHLCVProviderProtocol,
    requirements: OHLCVRequirements,
    *,
    max_attempts: int = MAX_ATTEMPTS,
    base_delay: float = BASE_DELAY_SECONDS,
) -> OHLCVs | None:
    """Assemble OHLCVs from the declared pairs whose data can be fetched.

    Every pair is fetched concurrently and retried on transient errors. A
    pair that still cannot be fetched is dropped from this assembly and
    reported at ERROR, so one symbol's data failure never blocks the others.

    Args:
        requirements: Requirements collected from strategy.setup().
        max_attempts: Total fetch attempts per pair.
        base_delay: Ceiling of the first retry delay in seconds.

    Returns:
        Assembled OHLCVs, or None when no declared pair could be fetched.

    Raises:
        StrategyCriticalError: If the strategy declared no OHLCV requirements,
            or a fetch reports a symbol the venue cannot trade.
        ExchangeCriticalError: If a fetch reports the venue unusable.
    """
    symbol_timeframes = _declared_symbol_timeframes(provider, requirements)
    pairs = _unique_pairs(symbol_timeframes)
    outcomes = await _fetch_pairs(
        provider, pairs, max_attempts=max_attempts, base_delay=base_delay
    )

    frames: dict[_Pair, pd.DataFrame] = {}
    for (symbol, timeframe), outcome in zip(pairs, outcomes, strict=True):
        if isinstance(outcome, (ExchangeCriticalError, StrategyCriticalError)):
            raise outcome
        if isinstance(outcome, BaseException):
            logger.error("%s@%s: skipping this cycle: %s", symbol, timeframe, outcome)
            continue
        frames[(symbol, timeframe)] = outcome
    if not frames:
        return None

    available = [st for st in symbol_timeframes if (st.symbol, st.timeframe) in frames]
    return _assembled(frames, available)


def _declared_symbol_timeframes(
    provider: OHLCVProviderProtocol, requirements: OHLCVRequirements
) -> list[SymbolTimeframe]:
    """Validate the declarations and pass their lookbacks to the provider.

    Raises:
        StrategyCriticalError: If strategy declared no OHLCV requirements.
    """
    all_requirements = requirements._get_all()
    if not all_requirements:
        raise StrategyCriticalError(
            "Strategy declared no OHLCV requirements. "
            "Check that strategy.setup() registers at least one symbol/timeframe."
        )
    _set_required_lookbacks(provider, all_requirements)
    return [SymbolTimeframe(r.symbol, r.timeframe) for r in all_requirements]


def _unique_pairs(symbol_timeframes: Sequence[SymbolTimeframe]) -> list[_Pair]:
    return list(dict.fromkeys((st.symbol, st.timeframe) for st in symbol_timeframes))


async def _fetch_pairs(
    provider: OHLCVProviderProtocol,
    pairs: Sequence[_Pair],
    *,
    max_attempts: int,
    base_delay: float,
) -> list[pd.DataFrame | BaseException]:
    """Fetch every pair concurrently and return one outcome per pair.

    Transient failures are retried up to max_attempts, so one attempt means
    no retry. Every task's outcome is collected, so no exception is left
    unretrieved.
    """
    fetch_pair = _pair_fetcher(provider)

    async def fetch_one(symbol: Symbol, timeframe: TimeFrame) -> pd.DataFrame:
        return await retry_on_transient(
            fetch_pair,
            symbol,
            timeframe,
            max_attempts=max_attempts,
            base_delay=base_delay,
        )

    tasks = [
        asyncio.create_task(fetch_one(symbol, timeframe)) for symbol, timeframe in pairs
    ]
    return cast(
        list[pd.DataFrame | BaseException],
        await asyncio.gather(*tasks, return_exceptions=True),
    )


def _assembled(
    frames: dict[_Pair, pd.DataFrame], symbol_timeframes: list[SymbolTimeframe]
) -> OHLCVs:
    def fetcher(symbol: Symbol, timeframe: TimeFrame) -> pd.DataFrame:
        return frames[(symbol, timeframe)]

    return OHLCVs(_fetch_and_format(fetcher, symbol_timeframes))


def _set_required_lookbacks(
    provider: OHLCVProviderProtocol, all_requirements: list[OHLCVRequirement]
) -> None:
    lookbacks: dict[TimeFrame, int] = {}
    for r in all_requirements:
        if r.lookback > 0:
            lookbacks[r.timeframe] = max(lookbacks.get(r.timeframe, 0), r.lookback)
    if lookbacks and hasattr(provider, "set_required_lookbacks"):
        provider.set_required_lookbacks(lookbacks)


def _pair_fetcher(
    provider: OHLCVProviderProtocol,
) -> Callable[[Symbol, TimeFrame], Awaitable[pd.DataFrame]]:
    """Read one pair through the provider's bulk API when it has one."""
    fetch_ohlcvs = getattr(provider, "fetch_ohlcvs", None)
    if callable(fetch_ohlcvs):

        async def fetch_pair(symbol: Symbol, timeframe: TimeFrame) -> pd.DataFrame:
            frames = await fetch_ohlcvs([(symbol, timeframe)])
            return cast(pd.DataFrame, frames[(symbol, timeframe)])

    else:

        async def fetch_pair(symbol: Symbol, timeframe: TimeFrame) -> pd.DataFrame:
            return provider.fetch_ohlcv(symbol, timeframe)

    return fetch_pair


def _fetch_and_format(
    ohlcv_fetcher: OHLCVFetcher,
    symbol_timeframes: list[SymbolTimeframe],
) -> OHLCVsByTimeframeAndSymbol:
    result: OHLCVsByTimeframeAndSymbol = {}
    for st in symbol_timeframes:
        df = ohlcv_fetcher(st.symbol, st.timeframe).sort_index()
        result.setdefault(st.timeframe, {})[st.symbol] = df
    return {tf: _align_symbols(symbol_dict) for tf, symbol_dict in result.items()}


def _align_symbols(
    symbol_dict: dict[Symbol, pd.DataFrame],
) -> dict[Symbol, pd.DataFrame]:
    """Pad shorter symbols with NaN to match the longest index in the timeframe."""
    if len(symbol_dict) <= 1:
        return symbol_dict
    symbols = list(symbol_dict.keys())
    combined = pd.concat(symbol_dict.values(), axis=1, keys=symbols)
    return {symbol: cast(pd.DataFrame, combined[symbol]) for symbol in symbols}
