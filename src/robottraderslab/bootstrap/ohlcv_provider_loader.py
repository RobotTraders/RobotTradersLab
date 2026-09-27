import inspect
import logging
from pathlib import Path
from typing import Any

from robottraderslab._core import OHLCVProviderProtocol
from robottraderslab.exceptions import StrategyCriticalError
from robottraderslab.ohlcv_provider import (
    CSVOHLCVProvider,
    ExchangeOHLCVProvider,
    MockOHLCVProvider,
    resolve_adapter_class,
    settings_the_factory_supplies,
)

from . import ohlcv_sources

logger = logging.getLogger(__name__)

_SOURCE_KEY = "ohlcv_provider"
_EXCHANGE_KEY = "exchange"
_VARIADIC_KINDS = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)


def load_ohlcv_provider(
    ohlcv_provider_config: dict[str, Any], **auto_params: Any
) -> OHLCVProviderProtocol:
    """Factory that loads the OHLCV provider a config names.

    Args:
        ohlcv_provider_config: Must contain `ohlcv_provider` key with one of:
            - "csv": loads CSVOHLCVProvider
            - "mock": loads MockOHLCVProvider
            - any other value (e.g. "bitget", "ccxt_binance"): loads
              ExchangeOHLCVProvider
    """
    config = dict(ohlcv_provider_config)
    config.update(**auto_params)
    provider_name: str = config.pop(_SOURCE_KEY)

    match provider_name.lower():
        case ohlcv_sources.CSV:
            return CSVOHLCVProvider(**config)
        case ohlcv_sources.MOCK:
            return MockOHLCVProvider(**config)
        case _:
            return ExchangeOHLCVProvider(exchange=provider_name, **config)


def require_ohlcv_provider(
    ohlcv_provider_config: dict[str, Any],
    section_label: str,
    config_file: Path | None,
) -> None:
    """A broken table is refused before the source is built, naming the
    settings file and the section it came from.

    Args:
        ohlcv_provider_config: A `[backtest.ohlcv_provider]` or
            `[live.ohlcv_provider]` table, read as declared.
        section_label: What the table is called in the config, e.g.
            `backtest.ohlcv_provider`.
        config_file: The settings file the table was declared in, or None
            when the config was not read from one.

    Raises:
        StrategyCriticalError: If the table names no source, or the named
            source is missing a key it needs or is given one it does not take.
    """
    missing, unknown = _ohlcv_provider_key_faults(ohlcv_provider_config)
    if not missing and not unknown:
        return
    raise StrategyCriticalError(
        _broken_ohlcv_provider_message(section_label, config_file, missing, unknown)
    )


def _ohlcv_provider_key_faults(
    ohlcv_provider_config: dict[str, Any],
) -> tuple[frozenset[str], frozenset[str]]:
    """Name the keys a source table gets wrong for the source it names.

    Returns:
        The keys the named source needs and does not find, and the keys it
        is given and does not take.
    """
    if _SOURCE_KEY not in ohlcv_provider_config:
        return frozenset({_SOURCE_KEY}), frozenset()

    contract = _provider_contract(ohlcv_provider_config[_SOURCE_KEY])
    if contract is None:
        return frozenset(), frozenset()
    required, accepted = contract

    given = frozenset(ohlcv_provider_config) - {_SOURCE_KEY}
    return required - given, given - accepted


def _provider_contract(
    provider_name: str,
) -> tuple[frozenset[str], frozenset[str]] | None:
    """The keys a named source needs and takes, or None when nothing about
    it can be checked.
    """
    match provider_name.lower():
        case ohlcv_sources.CSV:
            return _init_contract(CSVOHLCVProvider)
        case ohlcv_sources.MOCK:
            return _init_contract(MockOHLCVProvider)
        case _:
            return _exchange_source_contract(provider_name)


def _init_contract(provider_class: type[Any]) -> tuple[frozenset[str], frozenset[str]]:
    parameters = inspect.signature(provider_class).parameters
    accepted = frozenset(
        name
        for name, parameter in parameters.items()
        if parameter.kind not in _VARIADIC_KINDS
    )
    required = frozenset(
        name for name in accepted if parameters[name].default is inspect.Parameter.empty
    )
    return required, accepted


def _exchange_source_contract(
    provider_name: str,
) -> tuple[frozenset[str], frozenset[str]] | None:
    """The keys an exchange source and its adapter take, or None when the
    adapter cannot be resolved or declares nothing but a catch-all: a factory
    with no named parameter states no contract to check.

    A key the adapter's factory leaves no default for is one the table has
    to carry. A key the factory fills in from the adapter name itself is the
    factory's alone: the table is neither asked for it nor allowed to write
    it. The source's own keys all have defaults, so none of them is required.
    """
    adapter_class = resolve_adapter_class(provider_name)
    if adapter_class is None:
        return None

    adapter_parameters = {
        name: parameter
        for name, parameter in inspect.signature(
            adapter_class.create
        ).parameters.items()
        if parameter.kind not in _VARIADIC_KINDS
    }
    if not adapter_parameters:
        return None

    supplied = settings_the_factory_supplies(provider_name)
    required = (
        frozenset(
            name
            for name, parameter in adapter_parameters.items()
            if parameter.default is inspect.Parameter.empty
        )
        - supplied
    )
    exchange_keys = frozenset(
        name
        for name, parameter in inspect.signature(
            ExchangeOHLCVProvider
        ).parameters.items()
        if name != _EXCHANGE_KEY and parameter.kind not in _VARIADIC_KINDS
    )
    return required, exchange_keys | (frozenset(adapter_parameters) - supplied)


def _broken_ohlcv_provider_message(
    section_label: str,
    config_file: Path | None,
    missing: frozenset[str],
    unknown: frozenset[str],
) -> str:
    named_source = f"`{config_file}`" if config_file is not None else "the given config"
    complaints = []
    if missing:
        complaints.append(f"needs {_quoted(missing)}")
    if unknown:
        complaints.append(f"does not take {_quoted(unknown)}")
    return f"{named_source}: `{section_label}` " + " and ".join(complaints)


def _quoted(keys: frozenset[str]) -> str:
    return ", ".join(f"`{key}`" for key in sorted(keys))
