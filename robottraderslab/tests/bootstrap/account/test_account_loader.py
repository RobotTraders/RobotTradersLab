from collections.abc import AsyncIterator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest
import time_machine

from robottraderslab import Symbol, TimeFrame
from robottraderslab._internal import OrderOverrides, OrderSettings
from robottraderslab.bootstrap.account_loader import (
    load_accounts,
    load_single_account,
    load_single_exchange,
)
from robottraderslab.exceptions import BotCriticalError, ExchangeCriticalError
from robottraderslab.exchanges import (
    Closure,
    FuturesCapabilities,
    FuturesExchangeProtocol,
    Instrument,
    TradingStatus,
)
from robottraderslab.futures import FuturesAccount
from robottraderslab.live.livebot import LiveBot
from robottraderslab.strategies import (
    AccountSnapshots,
    BookKeeper,
    OHLCVs,
    StrategyProtocol,
    StrategyRequirements,
)
from robottraderslab_core.exchanges import OHLCVProviderProtocol, TriggerType

_SECRETS = {
    "bitget": {"api_key": "", "secret_key": "", "passphrase": ""},
    "my_account": {"api_key": "", "secret_key": "", "passphrase": ""},
}


@pytest.fixture
async def mock_exchange() -> AsyncMock:
    return AsyncMock(spec=FuturesExchangeProtocol)


@pytest.fixture
async def mock_load_exchange(mock_exchange: AsyncMock) -> AsyncIterator[Mock]:
    """A test that requests it never reaches a venue, so it needs no credentials."""
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


@pytest.mark.usefixtures("mock_load_exchange")
async def test_exchange_is_the_default_name_of_account():
    accounts = await load_accounts([{"exchange": "bitget"}], _SECRETS)

    assert set(accounts.keys()) == {"bitget"}


async def test_a_halted_symbol_is_logged_under_the_exchange_the_configuration_names(
    mock_load_exchange, mock_exchange, caplog
):
    mock_exchange.capabilities = FuturesCapabilities()
    mock_exchange.get_instrument.return_value = Instrument(
        quantity_step=1.0,
        minimum_order_value=None,
        maximum_leverage=None,
        maximum_order_quantity=None,
        maximum_market_order_quantity=None,
        closure=Closure(TradingStatus.CLOSED_TO_EVERY_ORDER, "maintain"),
    )
    btc = Symbol.create("BTC/USDT:USDT")
    accounts = await load_accounts(
        [{"exchange": "bitget", "account_name": "my_account"}], _SECRETS
    )

    class Strategy(StrategyProtocol):
        market_type = "futures"

        async def setup(self, requirements: StrategyRequirements) -> None:
            requirements.ohlcv.add(btc, "1d")
            requirements.account.add(accounts["my_account"], symbols=[btc])

        def generate_trading_signals(self, ohlcvs: OHLCVs) -> None:
            pass

        def book_trading_actions(
            self,
            ohlcvs: OHLCVs,
            account_snapshots: AccountSnapshots,
            timestamp: datetime,
            bookkeeper: BookKeeper,
            triggered_timeframes: list[TimeFrame],
        ) -> None:
            pass

    candles = pd.DataFrame(
        {"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]},
        index=pd.to_datetime(["2024-01-01"]),
    )
    provider = Mock(spec=OHLCVProviderProtocol)
    provider.fetch_ohlcv.return_value = candles

    with time_machine.travel(datetime(2026, 1, 1, tzinfo=UTC), tick=False):
        await LiveBot(Strategy(), provider).run()

    assert (
        "BTC/USDT:USDT on bitget (account my_account) is closed_to_every_order: "
        "the venue reports status maintain"
    ) in caplog.messages


@pytest.mark.usefixtures("mock_load_exchange")
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


@pytest.mark.usefixtures("mock_load_exchange")
@pytest.mark.parametrize(
    "accounts_config",
    [
        [config_without_account_name, config_without_account_name],
        [config_without_account_name, config_with_account_name],
        [config_with_account_name, config_with_account_name],
    ],
)
async def test_cant_have_accounts_with_same_account_name(accounts_config):
    with pytest.raises(BotCriticalError, match="Duplicate account name: `bitget`"):
        await load_accounts(accounts_config, _SECRETS)


@pytest.mark.usefixtures("mock_load_exchange")
async def test_account_carries_its_configured_name():
    accounts = await load_accounts(
        [{"exchange": "bitget", "account_name": "hedge"}], _SECRETS
    )

    assert accounts["hedge"].name == "hedge"


@pytest.mark.usefixtures("mock_load_exchange")
async def test_account_falls_back_to_the_exchange_name():
    accounts = await load_accounts([{"exchange": "bitget"}], _SECRETS)

    assert accounts["bitget"].name == "bitget"


@pytest.mark.usefixtures("mock_load_exchange")
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


async def test_an_unknown_secret_name_is_refused():
    with pytest.raises(
        BotCriticalError, match="`bitget` references secret `missing`.*not defined"
    ):
        await load_single_exchange({"exchange": "bitget", "secret_name": "missing"}, {})


async def test_an_account_without_a_name_is_named_by_its_section():
    with pytest.raises(
        BotCriticalError,
        match="`live.trading_account` references secret `missing`.*not defined",
    ):
        await load_single_exchange({"secret_name": "missing"}, {})


async def test_a_configured_trigger_type_the_venue_does_not_offer(mock_load_exchange):
    capabilities = FuturesCapabilities(
        trigger_types=frozenset({TriggerType.DEFAULT, TriggerType.MARK_PRICE})
    )
    overrides = OrderOverrides(stop_loss_trigger_type=TriggerType.LAST_PRICE)

    with pytest.raises(BotCriticalError, match="orders.stop_loss.trigger_type"):
        await _load_account_configuring(
            mock_load_exchange, "hyperliquid", capabilities, overrides
        )


async def test_a_configured_trigger_type_the_venue_offers(mock_load_exchange):
    capabilities = FuturesCapabilities(
        trigger_types=frozenset({TriggerType.DEFAULT, TriggerType.MARK_PRICE})
    )
    overrides = OrderOverrides(stop_loss_trigger_type=TriggerType.MARK_PRICE)

    account = await _load_account_configuring(
        mock_load_exchange, "hyperliquid", capabilities, overrides
    )
    moved = account.move_stop_loss(Symbol.create("BTC/USDC:USDC"), 90.0)

    assert moved.trigger_type is TriggerType.MARK_PRICE


async def test_a_reserved_field_under_the_name_the_connector_registers(
    mock_load_exchange,
):
    capabilities = FuturesCapabilities(reserved_order_fields=frozenset({"price"}))
    overrides = OrderOverrides(
        entry=OrderSettings(params={"simulator": {"price": "1"}})
    )

    with pytest.raises(BotCriticalError, match="orders.entry.params.simulator.price"):
        await _load_account_configuring(
            mock_load_exchange, "Simulator", capabilities, overrides
        )


async def test_a_reserved_field_in_each_section(mock_load_exchange):
    capabilities = FuturesCapabilities(
        reserved_order_fields=frozenset({"price", "size"})
    )
    overrides = OrderOverrides(
        entry=OrderSettings(params={"simulator": {"price": "1"}}),
        exit=OrderSettings(params={"simulator": {"size": "1"}}),
    )

    with pytest.raises(BotCriticalError) as refusal:
        await _load_account_configuring(
            mock_load_exchange, "simulator", capabilities, overrides
        )

    assert "orders.entry.params.simulator.price" in str(refusal.value)
    assert "orders.exit.params.simulator.size" in str(refusal.value)


async def test_a_reserved_field_addressed_to_another_connector(mock_load_exchange):
    capabilities = FuturesCapabilities(reserved_order_fields=frozenset({"price"}))
    overrides = OrderOverrides(entry=OrderSettings(params={"kraken": {"price": "1"}}))

    account = await _load_account_configuring(
        mock_load_exchange, "simulator", capabilities, overrides
    )

    assert account.name == "simulator"


async def test_a_venue_field_on_a_connector_reserving_none(mock_load_exchange):
    overrides = OrderOverrides(
        entry=OrderSettings(params={"simulator": {"price": "1"}})
    )

    account = await _load_account_configuring(
        mock_load_exchange, "simulator", FuturesCapabilities(), overrides
    )

    assert account.name == "simulator"


async def test_a_reserved_field_on_a_connector_no_entry_point_registers(
    mock_load_exchange,
):
    unregistered = "unregistered_module.create_exchange"
    capabilities = FuturesCapabilities(reserved_order_fields=frozenset({"price"}))
    overrides = OrderOverrides(
        entry=OrderSettings(params={unregistered: {"price": "1"}})
    )

    account = await _load_account_configuring(
        mock_load_exchange, unregistered, capabilities, overrides
    )

    assert account.name == unregistered


async def _load_account_configuring(
    mock_load_exchange: Mock,
    exchange: str,
    capabilities: FuturesCapabilities,
    order_overrides: OrderOverrides,
) -> FuturesAccount:
    mock_load_exchange.return_value.capabilities = capabilities
    return await load_single_account({"exchange": exchange}, {}, order_overrides)
