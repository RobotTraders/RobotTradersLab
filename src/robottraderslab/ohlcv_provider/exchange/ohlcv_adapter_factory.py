import asyncio
import inspect
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, NamedTuple

from robottraderslab._core import (
    ClassLoadingError,
    load_class,
    settings_the_factory_does_not_take,
)
from robottraderslab.exchanges import OhlcvAdapterProtocol

from .missing_candles import MarketOpenMask

logger = logging.getLogger(__name__)

_ENTRY_POINT_GROUP = "robot_traders_lab.ohlcv_adapters"
_CCXT_PREFIX = "ccxt_"
_CCXT_VENUE_SETTING = "exchange_name"
_QUALIFIER_SEPARATOR = "-"


class _LoopLock(NamedTuple):
    loop: asyncio.AbstractEventLoop
    lock: asyncio.Lock


class OhlcvAdapterFactory:
    """Factory owning the OHLCV adapter its downloads share."""

    def __init__(self) -> None:
        self._adapter: OhlcvAdapterProtocol | None = None
        self._holders = 0
        self._opening: _LoopLock | None = None

    @asynccontextmanager
    async def using(
        self, adapter: str, **kwargs: Any
    ) -> AsyncIterator[OhlcvAdapterProtocol]:
        """Hand a download the open adapter for as long as it holds it.

        Every download running at the same time is handed one instance, so a
        download never enters or closes what it is given, and one factory
        serves one adapter name.

        Args:
            adapter: Adapter identifier:
                - "ccxt_<name>" for CCXT adapters
                - "<name>" for plugin adapters resolved via entry points
            **kwargs: Adapter-specific configuration passed to .create()

        Raises:
            ExchangeCriticalError: If the venue is unusable, at open or at
                close.
        """
        instance = await self._acquire(adapter, **kwargs)
        try:
            yield instance
        finally:
            await self._release(adapter)

    async def _acquire(self, adapter: str, **kwargs: Any) -> OhlcvAdapterProtocol:
        """Opening the adapter takes a round trip, so the downloads starting
        together wait on the first to finish it.
        """
        async with self._lock_of_this_run():
            opened = self._adapter
            if opened is None:
                opened = await _create_adapter_instance(adapter, **kwargs).__aenter__()
                self._adapter = opened
            self._holders += 1
            return opened

    def _lock_of_this_run(self) -> asyncio.Lock:
        """A lock two downloads contend for belongs to the loop that contended
        it, and each run brings a loop of its own, never swapped while a
        download waits on it.
        """
        loop = asyncio.get_running_loop()
        opening = self._opening
        if opening is None or opening.loop is not loop:
            opening = _LoopLock(loop, asyncio.Lock())
            self._opening = opening
        return opening.lock

    async def _release(self, adapter: str) -> None:
        """No suspension separates the count reaching zero from the close, so
        the adapter closed is never one a download still holds and the close
        needs no lock of its own. A close that fails without leaving the venue
        unusable leaves nothing a download depends on, so it is only reported.

        Raises:
            ExchangeCriticalError: If the venue is unusable.
        """
        self._holders -= 1
        closing = self._adapter
        if self._holders == 0 and closing is not None:
            self._adapter = None
            try:
                await closing.__aexit__(None, None, None)
            except Exception:
                logger.warning(
                    "OHLCV adapter `%s` failed to close once its downloads ended",
                    adapter,
                    exc_info=True,
                )


def dataset_name(adapter: str, adapter_config: dict[str, Any]) -> str:
    """Name the series an adapter serves under this configuration.

    Candles are read back by whoever asks for the same symbol and timeframe,
    so two series a source keeps apart must carry different names. The adapter
    names the choices behind its own series, and sorting those names leaves
    each series with a single name whatever order its configuration was
    written in.

    Args:
        adapter: Adapter identifier, as configuration names it.
        adapter_config: The configuration the adapter is built from.

    Returns:
        The adapter's own name when it serves what the source serves by
        default, and that name extended by each qualifier otherwise.
    """
    qualifiers = _adapter_class(adapter).dataset_qualifiers(**adapter_config)
    return _QUALIFIER_SEPARATOR.join([adapter, *sorted(qualifiers)])


def market_open_mask_of(adapter: str) -> MarketOpenMask:
    """The calendar is the venue's, so it is read from the adapter's class
    before any adapter is open.

    Args:
        adapter: Adapter identifier, as configuration names it.

    Raises:
        RuntimeError: If no adapter is registered under the identifier.
    """
    market_open_mask: MarketOpenMask = _adapter_class(adapter).market_open_mask
    return market_open_mask


def resolve_adapter_class(adapter: str) -> Any | None:
    """Resolve the class an adapter identifier names.

    Args:
        adapter: "ccxt_<name>" for CCXT adapters, "<name>" for a plugin
            adapter resolved through the `robot_traders_lab.ohlcv_adapters`
            entry point group.

    Returns:
        The class, or None when no such adapter is registered.
    """
    name = _entry_point_name(adapter)
    try:
        return load_class(name, _ENTRY_POINT_GROUP)
    except ClassLoadingError:
        logger.debug("No OHLCV adapter registered under '%s'", name)
        return None


def settings_the_factory_supplies(adapter: str) -> frozenset[str]:
    """Name the settings this factory fills in from the adapter name itself.

    A `ccxt_<venue>` name carries its venue, so the configuration that names
    the adapter never writes the venue a second time.
    """
    if adapter.lower().startswith(_CCXT_PREFIX):
        return frozenset({_CCXT_VENUE_SETTING})
    return frozenset()


def _create_adapter_instance(adapter: str, **kwargs: Any) -> OhlcvAdapterProtocol:
    adapter_class = _adapter_class(adapter)
    if adapter.lower().startswith(_CCXT_PREFIX):
        kwargs[_CCXT_VENUE_SETTING] = adapter[len(_CCXT_PREFIX) :]

    _warn_about_settings_the_adapter_does_not_take(adapter, adapter_class, kwargs)
    instance: OhlcvAdapterProtocol = adapter_class.create(**kwargs)
    logger.debug(f"Loaded OHLCV adapter: {adapter}")
    return instance


def _adapter_class(adapter: str) -> Any:
    try:
        return load_class(_entry_point_name(adapter), _ENTRY_POINT_GROUP)
    except ClassLoadingError as e:
        raise RuntimeError(
            f"OHLCV adapter '{adapter}' not found. "
            f"Register it in the '{_ENTRY_POINT_GROUP}' entry point group "
            f"or use 'ccxt_<name>' for CCXT adapters."
        ) from e


def _entry_point_name(adapter: str) -> str:
    return "ccxt" if adapter.lower().startswith(_CCXT_PREFIX) else adapter


def _warn_about_settings_the_adapter_does_not_take(
    adapter: str, adapter_class: Any, settings: dict[str, Any]
) -> None:
    """Say which configured settings this adapter has no use for.

    The adapter is still built, on its own defaults, because a setting it
    cannot use is not a reason to stop trading.
    """
    unknown = settings_the_factory_does_not_take(
        inspect.signature(adapter_class.create).parameters, settings
    )
    if unknown:
        logger.warning(
            "OHLCV adapter '%s' takes no %s; it will use its own defaults",
            adapter,
            ", ".join(f"`{setting}`" for setting in unknown),
        )
