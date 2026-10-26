import asyncio
import logging
from typing import Any

from robottraderslab._core import NO_ORDER_OVERRIDES, OrderOverrides, TriggerType
from robottraderslab.exceptions import BotCriticalError, ExchangeCriticalError
from robottraderslab.exchanges import FuturesExchangeProtocol
from robottraderslab.futures import FuturesAccount

from .exchange_loader import load_exchange
from .settings import (
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
    _, _, account = await _load_account(
        account_config, secrets_by_name, order_overrides
    )
    return account


async def load_single_account_with_exchange(
    account_config: AccountConfig, secrets_by_name: SecretsByName
) -> tuple[FuturesExchangeProtocol, FuturesAccount]:
    """One connection serves a reader of the venue and the account a strategy
    is built on, so a command doing both opens the venue once.

    Args:
        secrets_by_name: API credentials keyed by account name.
    """
    _, exchange, account = await _load_account(account_config, secrets_by_name)
    return exchange, account


async def load_single_exchange(
    account_config: AccountConfig, secrets_by_name: SecretsByName
) -> FuturesExchangeProtocol:
    """Load the exchange an account's config points at.

    Args:
        secrets_by_name: API credentials keyed by account name.
    """
    _, merged_config = _named_config(account_config, secrets_by_name)
    return await load_exchange(merged_config)


async def _load_account(
    account_config: AccountConfig,
    secrets_by_name: SecretsByName,
    order_overrides: OrderOverrides = NO_ORDER_OVERRIDES,
) -> tuple[str, FuturesExchangeProtocol, FuturesAccount]:
    name, merged_config = _named_config(account_config, secrets_by_name)
    exchange_name = merged_config.get(_EXCHANGE_KEY)
    exchange = await load_exchange(merged_config)
    _refuse_unoffered_trigger_types(exchange_name, exchange, order_overrides)
    account = FuturesAccount(exchange, name=name, order_overrides=order_overrides)
    logger.debug(f"FuturesAccount `{name}` loaded for exchange `{exchange_name}`")
    return name, exchange, account


def _refuse_unoffered_trigger_types(
    exchange_name: str | None,
    exchange: FuturesExchangeProtocol,
    order_overrides: OrderOverrides,
) -> None:
    """A configured trigger type the venue does not offer would stop the
    run at its first order, so it stops the run before any.

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


def _named_config(
    account_config: AccountConfig, secrets_by_name: SecretsByName
) -> tuple[str, dict[str, Any]]:
    """`secret_name` lets an account claim a secrets entry filed under a name
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
