import logging
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest

from robottraderslab import Symbol
from robottraderslab._core import NO_ORDER_OVERRIDES, OHLCVProviderProtocol, TriggerType
from robottraderslab.bootstrap import BotConfig
from robottraderslab.exceptions import (
    ExchangeCriticalError,
    ExchangeRecoverableError,
    ExchangeTransientError,
    SymbolNotListedError,
)
from robottraderslab.exchanges import FuturesExchangeProtocol
from robottraderslab.futures import FuturesAccount
from robottraderslab.live.runners import _load_account, run_livebot
from robottraderslab.strategies import (
    Profile,
    ProfileStrategy,
    TradingMode,
    TradingSystem,
)


def _build_bot_config(**live_overrides: Any) -> BotConfig:
    return BotConfig.model_validate(
        {
            "strategy": {"strategy_class": "fake-strategy-99"},
            "live": {
                "trading_account": {},
                "ohlcv_provider": {"ohlcv_provider": "mock"},
                **live_overrides,
            },
        }
    )


@patch("robottraderslab.live.runners.LiveBot")
@patch("robottraderslab.live.runners.load_strategy")
@patch("robottraderslab.live.runners.load_ohlcv_provider")
@patch("robottraderslab.live.runners._load_account")
@patch("robottraderslab.live.runners.load_notifiers")
def test_notifiers_passed_to_livebot(
    mock_load_notifiers,
    mock_load_account,
    mock_load_ohlcv_provider,
    mock_load_strategy,
    mock_livebot_class,
):
    notifier_config = {"discord": {"fills": {"secret_name": "discord_notifs"}}}
    mock_notifier = AsyncMock()
    mock_load_notifiers.return_value = (
        [mock_notifier],
        [mock_notifier.notify_placement],
    )
    mock_livebot_class.return_value.run = AsyncMock()
    bot_config = _build_bot_config(notifier=notifier_config)

    run_livebot(bot_config)

    mock_load_notifiers.assert_called_once_with(notifier_config, {})
    mock_livebot_class.assert_called_once_with(
        mock_load_strategy.return_value,
        mock_load_ohlcv_provider.return_value,
        on_fill=[mock_notifier],
        on_placement=[mock_notifier.notify_placement],
        max_attempts=3,
        base_delay=1.0,
    )


@patch("robottraderslab.live.runners.LiveBot")
@patch("robottraderslab.live.runners.load_strategy")
@patch("robottraderslab.live.runners.load_ohlcv_provider")
@patch("robottraderslab.live.runners._load_account")
@patch("robottraderslab.live.runners.load_notifiers")
def test_retry_config_forwarded_to_livebot(
    mock_load_notifiers,
    mock_load_account,
    mock_load_ohlcv_provider,
    mock_load_strategy,
    mock_livebot_class,
):
    mock_load_notifiers.return_value = ([], [])
    mock_livebot_class.return_value.run = AsyncMock()
    bot_config = _build_bot_config(retry={"max_attempts": 5, "base_delay_seconds": 2.0})

    run_livebot(bot_config)

    mock_livebot_class.assert_called_once_with(
        mock_load_strategy.return_value,
        mock_load_ohlcv_provider.return_value,
        on_fill=[],
        on_placement=[],
        max_attempts=5,
        base_delay=2.0,
    )


@patch("robottraderslab.live.runners.LiveBot")
@patch("robottraderslab.live.runners.load_strategy")
@patch("robottraderslab.live.runners.load_ohlcv_provider")
@patch("robottraderslab.live.runners._load_account")
@patch("robottraderslab.live.runners.load_notifiers")
def test_the_configured_order_settings_reach_the_account(
    mock_load_notifiers,
    mock_load_account,
    mock_load_ohlcv_provider,
    mock_load_strategy,
    mock_livebot_class,
):
    mock_load_notifiers.return_value = ([], [])
    mock_livebot_class.return_value.run = AsyncMock()
    bot_config = BotConfig.model_validate(
        {
            "strategy": {"strategy_class": "fake-strategy-99"},
            "live": {
                "trading_account": {},
                "ohlcv_provider": {"ohlcv_provider": "mock"},
            },
            "orders": {"stop_loss": {"trigger_type": "mark_price"}},
        }
    )

    run_livebot(bot_config)

    overrides = mock_load_account.call_args.args[2]
    assert overrides.stop_loss_trigger_type is TriggerType.MARK_PRICE


@patch("robottraderslab.live.runners.LiveBot")
@patch("robottraderslab.live.runners.load_strategy")
@patch("robottraderslab.live.runners.load_ohlcv_provider")
@patch("robottraderslab.live.runners._load_account")
@patch("robottraderslab.live.runners.load_notifiers")
def test_retry_config_forwarded_to_load_account(
    mock_load_notifiers,
    mock_load_account,
    mock_load_ohlcv_provider,
    mock_load_strategy,
    mock_livebot_class,
):
    mock_load_notifiers.return_value = ([], [])
    mock_livebot_class.return_value.run = AsyncMock()
    bot_config = _build_bot_config(
        trading_account={"exchange": "bitget"},
        retry={"max_attempts": 5, "base_delay_seconds": 2.0},
    )

    run_livebot(bot_config)

    mock_load_account.assert_called_once_with(
        {"exchange": "bitget"},
        {},
        NO_ORDER_OVERRIDES,
        max_attempts=5,
        base_delay=2.0,
    )


@patch("robottraderslab._core.retry.asyncio.sleep", new_callable=AsyncMock)
@patch("robottraderslab.live.runners.load_single_account", new_callable=AsyncMock)
async def test_load_account_transient_error_with_recovery(mock_load, mock_sleep):
    expected_account = Mock()
    mock_load.side_effect = [ExchangeTransientError("502"), expected_account]

    account = await _load_account(
        {}, {}, NO_ORDER_OVERRIDES, max_attempts=3, base_delay=1.0
    )

    assert account is expected_account
    assert mock_load.call_count == 2


@patch("robottraderslab._core.retry.asyncio.sleep", new_callable=AsyncMock)
@patch("robottraderslab.live.runners.load_single_account", new_callable=AsyncMock)
async def test_load_account_transient_error_exhausted(mock_load, mock_sleep):
    mock_load.side_effect = ExchangeTransientError("502")

    with pytest.raises(ExchangeTransientError, match="502"):
        await _load_account({}, {}, NO_ORDER_OVERRIDES, max_attempts=3, base_delay=1.0)


@patch("robottraderslab._core.retry.asyncio.sleep", new_callable=AsyncMock)
@patch("robottraderslab.live.runners.load_single_account", new_callable=AsyncMock)
async def test_load_account_retries_a_venue_rejection(mock_load, mock_sleep):
    expected_account = Mock()
    mock_load.side_effect = [
        ExchangeRecoverableError("400 Bad Request | Bitget msg: 40015"),
        expected_account,
    ]

    account = await _load_account(
        {}, {}, NO_ORDER_OVERRIDES, max_attempts=3, base_delay=1.0
    )

    assert account is expected_account
    assert mock_load.call_count == 2


@patch("robottraderslab._core.retry.asyncio.sleep", new_callable=AsyncMock)
@patch("robottraderslab.live.runners.load_single_account", new_callable=AsyncMock)
async def test_load_account_stops_when_the_venue_keeps_rejecting(
    mock_load, mock_sleep, caplog
):
    mock_load.side_effect = ExchangeRecoverableError("Bitget msg: 40015")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ExchangeRecoverableError, match="40015"):
            await _load_account(
                {}, {}, NO_ORDER_OVERRIDES, max_attempts=3, base_delay=1.0
            )

    assert mock_load.call_count == 3
    assert "Cannot initialize account" in caplog.records[-1].getMessage()


@patch("robottraderslab._core.retry.asyncio.sleep", new_callable=AsyncMock)
@patch("robottraderslab.live.runners.load_single_account", new_callable=AsyncMock)
async def test_load_account_stops_on_a_critical_error(mock_load, mock_sleep):
    mock_load.side_effect = ExchangeCriticalError("auth failed")

    with pytest.raises(ExchangeCriticalError, match="auth failed"):
        await _load_account({}, {}, NO_ORDER_OVERRIDES, max_attempts=3, base_delay=1.0)

    assert mock_load.call_count == 1


@patch("robottraderslab._core.retry.random.uniform", side_effect=lambda low, high: high)
@patch("robottraderslab._core.retry.asyncio.sleep", new_callable=AsyncMock)
@patch("robottraderslab.live.runners.load_single_account", new_callable=AsyncMock)
async def test_load_account_respects_custom_retry_params(
    mock_load, mock_sleep, mock_uniform
):
    mock_load.side_effect = ExchangeTransientError("502")

    with pytest.raises(ExchangeTransientError):
        await _load_account({}, {}, NO_ORDER_OVERRIDES, max_attempts=2, base_delay=5.0)

    assert mock_load.call_count == 2
    mock_sleep.assert_called_once_with(5.0)


BTC = Symbol.create("BTC/USDT:USDT")
TAO = Symbol.create("TAO/USDT:USDT")


@patch("robottraderslab.live.runners.load_notifiers")
@patch("robottraderslab.live.runners.load_ohlcv_provider")
@patch("robottraderslab.live.runners.load_strategy")
@patch("robottraderslab.live.runners.load_single_account", new_callable=AsyncMock)
def test_a_configuration_naming_an_unlisted_symbol_starts_the_bot(
    mock_load_account,
    mock_load_strategy,
    mock_load_ohlcv_provider,
    mock_load_notifiers,
    caplog,
):
    mock_load_notifiers.return_value = ([], [])
    exchange = Mock(spec=FuturesExchangeProtocol)
    exchange.get_open_positions.return_value = {}
    account = FuturesAccount(exchange)
    mock_load_account.return_value = account

    class Strategy(ProfileStrategy[Profile]):
        market_type = "futures"

        def __init__(self):
            super().__init__(
                account=account,
                trading_system=TradingSystem(trading_mode=TradingMode.LIVE),
                config_dir=None,
                profiles=[
                    {"symbol": str(symbol), "timeframe": "1m"} for symbol in (TAO, BTC)
                ],
            )

        async def setup(self, requirements):
            for profile in self.profiles:
                requirements.ohlcv.add(profile.symbol, profile.timeframe)
            requirements.account.add(
                account, symbols=[profile.symbol for profile in self.profiles]
            )

        def book_profile_actions(
            self, profile, ohlcvs, account_snapshots, timestamp, bookkeeper
        ):
            ohlcvs.column(profile.symbol, profile.timeframe, "close")
            bookkeeper.add(account.long_entry(profile.symbol, 1.0))

    def fetch_ohlcv(symbol, timeframe):
        if symbol == TAO:
            raise SymbolNotListedError("TAO/USDT:USDT has no usable contract.")
        return pd.DataFrame(
            {
                "open": [100.0],
                "high": [110.0],
                "low": [90.0],
                "close": [105.0],
                "volume": [1000.0],
            },
            index=pd.to_datetime(["2024-01-01"]),
        )

    ohlcv_provider = Mock(spec=OHLCVProviderProtocol)
    ohlcv_provider.fetch_ohlcv.side_effect = fetch_ohlcv
    mock_load_ohlcv_provider.return_value = ohlcv_provider
    mock_load_strategy.return_value = Strategy()

    with caplog.at_level(logging.WARNING):
        run_livebot(_build_bot_config())

    mock_load_account.assert_awaited_once()
    exchange.place_market_order.assert_awaited_once()
    assert exchange.place_market_order.await_args.kwargs["symbol"] == BTC
    assert [
        record for record in caplog.records if record.levelno >= logging.ERROR
    ] == []
