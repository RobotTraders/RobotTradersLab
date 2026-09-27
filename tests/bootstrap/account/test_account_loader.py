from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, Mock, patch

import pytest

from robottraderslab.bootstrap.account_loader import load_accounts, load_single_exchange
from robottraderslab.exceptions import ExchangeCriticalError, StrategyCriticalError
from robottraderslab.exchanges import FuturesExchangeProtocol

_SECRETS = {
    "bitget": {"api_key": "", "secret_key": "", "passphrase": ""},
    "my_account": {"api_key": "", "secret_key": "", "passphrase": ""},
}


@pytest.fixture
async def mock_exchange() -> AsyncMock:
    return AsyncMock(spec=FuturesExchangeProtocol)


@pytest.fixture(autouse=True)
async def mock_load_exchange(mock_exchange: AsyncMock) -> AsyncIterator[Mock]:
    """Stand in for the venue, so no test in this module needs credentials."""
    with patch(
        "robottraderslab.bootstrap.account_loader.load_exchange",
        return_value=mock_exchange,
    ) as load_exchange:
        yield load_exchange


async def test_accounts_are_optional():
    assert await load_accounts([], _SECRETS) == {}


async def test_exchange_is_required():
    with pytest.raises(
        ExchangeCriticalError, match="`exchange` is a required config key"
    ):
        await load_accounts([{}], _SECRETS)


async def test_exchange_is_the_default_name_of_account():
    accounts = await load_accounts([{"exchange": "bitget"}], _SECRETS)

    assert set(accounts.keys()) == {"bitget"}


async def test_account_name_is_used_when_provided():
    accounts = await load_accounts(
        [{"exchange": "bitget", "account_name": "my_account"}], _SECRETS
    )

    assert set(accounts.keys()) == {"my_account"}


config_without_account_name = {"exchange": "bitget"}
config_with_account_name = {
    "exchange": "simulator",
    "account_name": "bitget",
    "initial_balance": {"USDT": 1000.0},
    "maker_fee_rate": 0.001,
    "taker_fee_rate": 0.001,
}


@pytest.mark.parametrize(
    "accounts_config",
    [
        [config_without_account_name, config_without_account_name],
        [config_without_account_name, config_with_account_name],
        [config_with_account_name, config_with_account_name],
    ],
)
async def test_cant_have_accounts_with_same_account_name(accounts_config):
    with pytest.raises(StrategyCriticalError, match="Duplicate account name: `bitget`"):
        await load_accounts(accounts_config, _SECRETS)


async def test_account_carries_its_configured_name():
    accounts = await load_accounts(
        [{"exchange": "bitget", "account_name": "hedge"}], _SECRETS
    )

    assert accounts["hedge"].name == "hedge"


async def test_account_falls_back_to_the_exchange_name():
    accounts = await load_accounts([{"exchange": "bitget"}], _SECRETS)

    assert accounts["bitget"].name == "bitget"


async def test_single_exchange_is_the_loaded_instance(mock_exchange):
    exchange = await load_single_exchange({"exchange": "bitget"}, _SECRETS)

    assert exchange is mock_exchange


async def test_single_exchange_requires_an_exchange():
    with pytest.raises(
        ExchangeCriticalError, match="`exchange` is a required config key"
    ):
        await load_single_exchange({}, _SECRETS)


async def test_single_exchange_receives_the_named_accounts_secrets(mock_load_exchange):
    await load_single_exchange(
        {"exchange": "bitget", "account_name": "my_account", "testnet": True},
        {"my_account": {"api_key": "from-secrets"}},
    )

    mock_load_exchange.assert_called_once_with(
        {"exchange": "bitget", "testnet": True, "api_key": "from-secrets"}
    )


async def test_a_secret_entry_may_supply_the_exchange(mock_load_exchange):
    await load_single_exchange(
        {"account_name": "my_account"},
        {"my_account": {"exchange": "bitget", "api_key": "from-secrets"}},
    )

    mock_load_exchange.assert_called_once_with(
        {"exchange": "bitget", "api_key": "from-secrets"}
    )


async def test_an_explicit_secret_name_is_used_over_the_account_name(
    mock_load_exchange,
):
    await load_single_exchange(
        {"exchange": "bitget", "account_name": "hedge", "secret_name": "my_account"},
        {"my_account": {"api_key": "from-secrets"}},
    )

    mock_load_exchange.assert_called_once_with(
        {"exchange": "bitget", "api_key": "from-secrets"}
    )


async def test_an_unknown_secret_name_is_refused(mock_load_exchange):
    with pytest.raises(
        StrategyCriticalError, match="`bitget` references secret `missing`.*not defined"
    ):
        await load_single_exchange({"exchange": "bitget", "secret_name": "missing"}, {})
