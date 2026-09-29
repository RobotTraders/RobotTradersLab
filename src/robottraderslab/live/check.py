from pathlib import Path
from typing import Any

from robottraderslab._core import Balance, Currency, Symbol, run_async
from robottraderslab.bootstrap import (
    SECRET_REFERENCE_KEY,
    SecretsByName,
    check_secret,
    load_secrets,
    load_single_exchange,
)

from .check_error import CheckError

_ACCOUNT_NAME_KEY = "account_name"
_EXCHANGE_KEY = "exchange"
_WEBHOOK_KEY = "webhook_url"
_AMOUNT_DECIMALS = 8


def main(entry: str, secrets: Path, symbol: str, workspace: Path) -> None:
    """
    Args:
        entry: The name of the `[[secrets]]` entry.
        secrets: The secrets file, relative to the workspace.
        symbol: The market whose account a venue key is read on.

    Raises:
        CheckError: If the file or the entry is missing, the entry names no
            `exchange` and carries no `webhook_url`, or it carries a
            `webhook_url` and no installed plugin checks one.
        ExchangeCriticalError: If the connector is not installed, the venue
            refuses the key, or an account setting is not one the connector
            trades under.
        ExchangeRecoverableError: If the venue rejects or fails the read.
        Exception: If the plugin checking a webhook refuses it.
    """
    secrets_file = workspace / secrets
    secrets_by_name = _entries(secrets_file, entry)
    fields = secrets_by_name[entry]
    if _WEBHOOK_KEY in fields:
        run_async(_check_webhook(entry, fields))
    elif _EXCHANGE_KEY in fields:
        run_async(_check_venue(entry, secrets_by_name, Symbol.create(symbol)))
    else:
        raise CheckError(
            f"`{entry}` in `{secrets_file}` names no `{_EXCHANGE_KEY}` and carries "
            f"no `{_WEBHOOK_KEY}`, so there is nothing to check it against"
        )


def _entries(secrets_file: Path, entry: str) -> SecretsByName:
    if not secrets_file.is_file():
        raise CheckError(f"there is no secrets file at `{secrets_file}`")
    secrets_by_name = load_secrets(secrets_file)
    if entry not in secrets_by_name:
        raise CheckError(
            f"`{secrets_file}` holds no `[[secrets]]` entry named `{entry}`"
        )
    return secrets_by_name


async def _check_webhook(entry: str, fields: dict[str, Any]) -> None:
    lines = await check_secret(_WEBHOOK_KEY, fields, entry=entry)
    if lines is None:
        raise CheckError(
            f"`{entry}` carries a `{_WEBHOOK_KEY}`, and the Discord notifications plugin, which "
            "checks one, is not installed"
        )
    for line in lines:
        print(line)


async def _check_venue(
    entry: str, secrets_by_name: SecretsByName, symbol: Symbol
) -> None:
    """The connector's own account check runs first, since a connector
    refuses to build on an account set to a mode it does not trade under.
    """
    fields = secrets_by_name[entry]
    exchange_name = fields[_EXCHANGE_KEY]
    print(f"`{entry}` is a {exchange_name} key, read on the account of {symbol}")
    settings = {key: value for key, value in fields.items() if key != _EXCHANGE_KEY}
    lines = await check_secret(exchange_name, settings, entry=entry, symbol=symbol)
    for line in lines or []:
        print(line)
    exchange = await load_single_exchange(
        {_ACCOUNT_NAME_KEY: entry, SECRET_REFERENCE_KEY: entry}, secrets_by_name
    )
    balances = await exchange.get_balances([symbol])
    for currency, balance in sorted(balances.items()):
        print(_balance_line(currency, balance))


def _balance_line(currency: Currency, balance: Balance) -> str:
    return (
        f"{currency}: total {_amount(balance.total)}, locked {_amount(balance.locked)}"
    )


def _amount(value: float) -> str:
    return f"{value:,.{_AMOUNT_DECIMALS}f}".rstrip("0").rstrip(".")
