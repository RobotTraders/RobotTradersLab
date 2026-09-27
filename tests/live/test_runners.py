import logging
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from robottraderslab.bootstrap import BotConfig
from robottraderslab.exceptions import (
    ExchangeCriticalError,
    ExchangeRecoverableError,
    ExchangeTransientError,
)
from robottraderslab.live.runners import _load_account, run_livebot


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
        max_attempts=5,
        base_delay=2.0,
    )


@patch("robottraderslab._core.retry.asyncio.sleep", new_callable=AsyncMock)
@patch("robottraderslab.live.runners.load_single_account", new_callable=AsyncMock)
async def test_load_account_transient_error_with_recovery(mock_load, mock_sleep):
    expected_account = Mock()
    mock_load.side_effect = [ExchangeTransientError("502"), expected_account]

    account = await _load_account({}, {}, max_attempts=3, base_delay=1.0)

    assert account is expected_account
    assert mock_load.call_count == 2


@patch("robottraderslab._core.retry.asyncio.sleep", new_callable=AsyncMock)
@patch("robottraderslab.live.runners.load_single_account", new_callable=AsyncMock)
async def test_load_account_transient_error_exhausted(mock_load, mock_sleep):
    mock_load.side_effect = ExchangeTransientError("502")

    with pytest.raises(ExchangeTransientError, match="502"):
        await _load_account({}, {}, max_attempts=3, base_delay=1.0)


@patch("robottraderslab._core.retry.asyncio.sleep", new_callable=AsyncMock)
@patch("robottraderslab.live.runners.load_single_account", new_callable=AsyncMock)
async def test_load_account_retries_a_venue_rejection(mock_load, mock_sleep):
    expected_account = Mock()
    mock_load.side_effect = [
        ExchangeRecoverableError("400 Bad Request | Bitget msg: 40015"),
        expected_account,
    ]

    account = await _load_account({}, {}, max_attempts=3, base_delay=1.0)

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
            await _load_account({}, {}, max_attempts=3, base_delay=1.0)

    assert mock_load.call_count == 3
    assert "Cannot initialize account" in caplog.records[-1].getMessage()


@patch("robottraderslab._core.retry.asyncio.sleep", new_callable=AsyncMock)
@patch("robottraderslab.live.runners.load_single_account", new_callable=AsyncMock)
async def test_load_account_stops_on_a_critical_error(mock_load, mock_sleep):
    mock_load.side_effect = ExchangeCriticalError("auth failed")

    with pytest.raises(ExchangeCriticalError, match="auth failed"):
        await _load_account({}, {}, max_attempts=3, base_delay=1.0)

    assert mock_load.call_count == 1


@patch("robottraderslab._core.retry.random.uniform", side_effect=lambda low, high: high)
@patch("robottraderslab._core.retry.asyncio.sleep", new_callable=AsyncMock)
@patch("robottraderslab.live.runners.load_single_account", new_callable=AsyncMock)
async def test_load_account_respects_custom_retry_params(
    mock_load, mock_sleep, mock_uniform
):
    mock_load.side_effect = ExchangeTransientError("502")

    with pytest.raises(ExchangeTransientError):
        await _load_account({}, {}, max_attempts=2, base_delay=5.0)

    assert mock_load.call_count == 2
    mock_sleep.assert_called_once_with(5.0)
