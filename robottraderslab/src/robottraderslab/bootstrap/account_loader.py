import asyncio
import logging
from typing import Any

from robottraderslab._internal import (
    NO_ORDER_OVERRIDES,
    OrderOverrides,
    registered_entry_point_name,
)
from robottraderslab.exceptions import BotCriticalError, ExchangeCriticalError
from robottraderslab.exchanges import FuturesExchangeProtocol
from robottraderslab.futures import FuturesAccount
from robottraderslab_core.exchanges import TriggerType, VenueParams

from .exchange_loader import load_exchange
from .settings import (
    EXCHANGE_ENTRY_POINT_GROUP,
    SECRET_REFERENCE_KEY,
    TRADING_ACCOUNT_SECTION,
    AccountConfig,
    SecretsByName,
    resolve_secret_references,
)

logger = logging.getLogger(__name__)

_ACCOUNT_NAME_KEY = "account_name"
_EXCHANGE_KEY = "exchange"


async def load_accounts(
    accounts_config: list[AccountConfig], secrets_by_name: SecretsByName
) -> dict[str, FuturesAccount]:
    """Connect to each account's exchange and return the accounts by name.

    Args:
        accounts_config: Each config names an exchange.
        secrets_by_name: API credentials keyed by account name; a
            `secret_name` a config names is defined here.

    Returns:
        The accounts keyed by their names, in the order of the configs.

    Raises:
        BotCriticalError: If two accounts share a name, or an account
            references a secret the secrets file does not define.
        ExchangeCriticalError: If an account names no exchange.
    """
    tasks = [
        _load_account(account_config, secrets_by_name)
        for account_config in accounts_config
    ]
    named_accounts = await asyncio.gather(*tasks)

    accounts: dict[str, FuturesAccount] = {}
    for name, _, account in named_accounts:
        if accounts.setdefault(name, account) is not account:
            raise BotCriticalError(f"Duplicate account name: `{name}`")
    return accounts


async def load_single_account(
    account_config: AccountConfig,
    secrets_by_name: SecretsByName,
    order_overrides: OrderOverrides = NO_ORDER_OVERRIDES,
) -> FuturesAccount:
    """Connect to the account's exchange and return the account.

    Args:
        account_config: Names an exchange.
        secrets_by_name: API credentials keyed by account name; a
            `secret_name` the config names is defined here.
        order_overrides: The `[orders]` settings the account's orders take.

    Raises:
        BotCriticalError: If the account references a secret the secrets file
            does not define, or `[orders]` names a trigger type the venue
            offers no trigger on or a venue field its connector reserves.
        ExchangeCriticalError: If the config names no exchange.
    """
    _, _, account = await _load_account(
        account_config, secrets_by_name, order_overrides
    )
    return account


async def load_single_account_with_exchange(
    account_config: AccountConfig, secrets_by_name: SecretsByName
) -> tuple[FuturesExchangeProtocol, FuturesAccount]:
    """Return the exchange and the account built on its one connection.

    A command that reads the venue and builds a strategy on the account opens
    the venue once.

    Args:
        account_config: Names an exchange.
        secrets_by_name: API credentials keyed by account name; a
            `secret_name` the config names is defined here.

    Raises:
        BotCriticalError: If the config names a `secret_name` the secrets
            file does not define.
        ExchangeCriticalError: If the config names no exchange.
    """
    _, exchange, account = await _load_account(account_config, secrets_by_name)
    return exchange, account


async def load_single_exchange(
    account_config: AccountConfig, secrets_by_name: SecretsByName
) -> FuturesExchangeProtocol:
    """Load the exchange an account's config points at.

    Args:
        account_config: Names an exchange.
        secrets_by_name: API credentials keyed by account name; a
            `secret_name` the config names is defined here.

    Raises:
        BotCriticalError: If the config names a `secret_name` the secrets
            file does not define.
        ExchangeCriticalError: If the config names no exchange.
    """
    _, merged_config = _named_config(account_config, secrets_by_name)
    return await load_exchange(merged_config)


async def _load_account(
    account_config: AccountConfig,
    secrets_by_name: SecretsByName,
    order_overrides: OrderOverrides = NO_ORDER_OVERRIDES,
) -> tuple[str, FuturesExchangeProtocol, FuturesAccount]:
    name, merged_config = _named_config(account_config, secrets_by_name)
    exchange_name = merged_config[_EXCHANGE_KEY]
    exchange = await load_exchange(merged_config)
    _refuse_unoffered_trigger_types(exchange_name, exchange, order_overrides)
    _refuse_reserved_venue_fields(exchange_name, exchange, order_overrides)
    account = FuturesAccount(
        exchange, name=name, order_overrides=order_overrides, venue=exchange_name
    )
    logger.debug(f"FuturesAccount `{name}` loaded for exchange `{exchange_name}`")
    return name, exchange, account


def _named_config(
    account_config: AccountConfig, secrets_by_name: SecretsByName
) -> tuple[str, dict[str, Any]]:
    """Name an account and merge its config with its secrets entry.

    `secret_name` lets an account claim a secrets entry filed under a name
    other than its own.

    Raises:
        ExchangeCriticalError: If the config names no exchange.
        BotCriticalError: If the config names a `secret_name` that is not defined
            in the secrets file.
    """
    config = dict(account_config)
    name = config.pop(_ACCOUNT_NAME_KEY, "") or config.get(_EXCHANGE_KEY, "")
    if SECRET_REFERENCE_KEY in config:
        label = name or TRADING_ACCOUNT_SECTION
        merged = resolve_secret_references({label: config}, secrets_by_name)[label]
    else:
        merged = {**config, **secrets_by_name.get(name, {})}
    if _EXCHANGE_KEY not in merged:
        raise ExchangeCriticalError(f"`{_EXCHANGE_KEY}` is a required config key")
    return name, merged


def _refuse_unoffered_trigger_types(
    exchange_name: str,
    exchange: FuturesExchangeProtocol,
    order_overrides: OrderOverrides,
) -> None:
    """Stop the run before any order if a configured trigger type is not offered.

    A trigger type the venue does not offer would stop the run at its first
    order.

    Raises:
        BotCriticalError: If `[orders]` names a trigger type the venue
            offers no trigger on.
    """
    configured = {
        "orders.entry.trigger_type": order_overrides.entry.trigger_type,
        "orders.exit.trigger_type": order_overrides.exit.trigger_type,
        "orders.stop_loss.trigger_type": order_overrides.stop_loss_trigger_type,
        "orders.take_profit.trigger_type": order_overrides.take_profit_trigger_type,
    }
    offered = exchange.capabilities.trigger_types
    refused = [
        f"`{key} = {value}`"
        for key, value in configured.items()
        if value is not None and value not in offered
    ]
    if refused:
        names = ", ".join(sorted(offered - {TriggerType.DEFAULT})) or "none"
        raise BotCriticalError(
            f"{exchange_name} offers no trigger on {', '.join(refused)}; "
            f"it offers {names}"
        )


def _refuse_reserved_venue_fields(
    exchange_name: str,
    exchange: FuturesExchangeProtocol,
    order_overrides: OrderOverrides,
) -> None:
    """Stop the run before any order if a configured venue field is reserved.

    The fields are looked up under the name the connector registers, the
    key `[orders]` addresses it by, however `exchange` spells it; a connector
    no entry point registers is addressed by no key.

    Raises:
        BotCriticalError: If `[orders.entry]` or `[orders.exit]` sets a venue
            field the connector reserves.
    """
    entry_params = order_overrides.entry.params
    exit_params = order_overrides.exit.params
    if not entry_params and not exit_params:
        return
    registered = registered_entry_point_name(exchange_name, EXCHANGE_ENTRY_POINT_GROUP)
    if registered is None:
        return
    reserved = exchange.capabilities.reserved_order_fields
    refused = [
        *_reserved_fields_of_section("entry", entry_params, registered, reserved),
        *_reserved_fields_of_section("exit", exit_params, registered, reserved),
    ]
    if refused:
        raise BotCriticalError(
            f"{registered} sets {', '.join(refused)} from the order itself; "
            "drop them from `bot.toml`"
        )


def _reserved_fields_of_section(
    role: str, params: VenueParams, registered: str, reserved: frozenset[str]
) -> list[str]:
    """List the reserved fields one section of `[orders]` sets for the connector."""
    fields = reserved & params.get(registered, {}).keys()
    return [f"`orders.{role}.params.{registered}.{field}`" for field in sorted(fields)]
