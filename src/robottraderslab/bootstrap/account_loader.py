import asyncio
import logging
from typing import Any

from robottraderslab.exceptions import ExchangeCriticalError, StrategyCriticalError
from robottraderslab.exchanges import FuturesExchangeProtocol
from robottraderslab.futures import FuturesAccount

from .exchange_loader import load_exchange
from .settings import (
    SECRET_REFERENCE_KEY,
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
            raise StrategyCriticalError(f"Duplicate account name: `{name}`")
    return accounts


async def load_single_account(
    account_config: AccountConfig, secrets_by_name: SecretsByName
) -> FuturesAccount:
    _, _, account = await _load_account(account_config, secrets_by_name)
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
    account_config: AccountConfig, secrets_by_name: SecretsByName
) -> tuple[str, FuturesExchangeProtocol, FuturesAccount]:
    name, merged_config = _named_config(account_config, secrets_by_name)
    exchange_name = merged_config.get(_EXCHANGE_KEY)
    exchange = await load_exchange(merged_config)
    account = FuturesAccount(exchange, name=name)
    logger.debug(f"FuturesAccount `{name}` loaded for exchange `{exchange_name}`")
    return name, exchange, account


def _named_config(
    account_config: AccountConfig, secrets_by_name: SecretsByName
) -> tuple[str, dict[str, Any]]:
    """`secret_name` lets an account claim a secrets entry filed under a name
    other than its own.

    Raises:
        ExchangeCriticalError: If the config names no exchange.
        StrategyCriticalError: If the config names a `secret_name` that is not defined
            in the secrets file.
    """
    config = dict(account_config)
    name = config.pop(_ACCOUNT_NAME_KEY, "") or config.get(_EXCHANGE_KEY, "")
    if SECRET_REFERENCE_KEY in config:
        merged = resolve_secret_references({name: config}, secrets_by_name)[name]
    else:
        merged = {**config, **secrets_by_name.get(name, {})}
    if _EXCHANGE_KEY not in merged:
        raise ExchangeCriticalError(f"`{_EXCHANGE_KEY}` is a required config key")
    return name, merged
