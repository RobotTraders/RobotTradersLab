import logging
from unittest.mock import AsyncMock, patch

import pytest

from robottraderslab._core import retry_on_transient
from robottraderslab.exceptions import (
    ExchangeCriticalError,
    ExchangeRecoverableError,
    ExchangeTransientError,
)


async def test_first_attempt_success():
    fn = AsyncMock(return_value="ok")

    returned_value = await retry_on_transient(fn, max_attempts=3)

    assert returned_value == "ok"
    assert fn.call_count == 1


@patch("robottraderslab._core.retry.random.uniform", side_effect=[0.4, 1.7])
@patch("robottraderslab._core.retry.asyncio.sleep", new_callable=AsyncMock)
async def test_transient_failures_then_success(mock_sleep, mock_uniform):
    fn = AsyncMock(
        side_effect=[ExchangeTransientError("502"), ExchangeTransientError("502"), "ok"]
    )

    returned_value = await retry_on_transient(fn, max_attempts=3, base_delay=1.0)

    assert returned_value == "ok"
    assert fn.call_count == 3
    mock_uniform.assert_any_call(0, 1.0)
    mock_uniform.assert_any_call(0, 2.0)
    mock_sleep.assert_any_call(0.4)
    mock_sleep.assert_any_call(1.7)


@patch("robottraderslab._core.retry.asyncio.sleep", new_callable=AsyncMock)
async def test_all_attempts_exhausted(mock_sleep):
    fn = AsyncMock(
        side_effect=[
            ExchangeTransientError("1"),
            ExchangeTransientError("2"),
            ExchangeTransientError("last"),
        ]
    )

    with pytest.raises(ExchangeTransientError, match="last"):
        await retry_on_transient(fn, max_attempts=3, base_delay=1.0)

    assert fn.call_count == 3
    assert mock_sleep.call_count == 2


async def test_non_transient_recoverable_error():
    fn = AsyncMock(side_effect=ExchangeRecoverableError("insufficient balance"))

    with pytest.raises(ExchangeRecoverableError, match="insufficient balance"):
        await retry_on_transient(fn, max_attempts=3)

    assert fn.call_count == 1


@patch("robottraderslab._core.retry.asyncio.sleep", new_callable=AsyncMock)
async def test_a_widened_retryable_class_retries_a_recoverable_error(mock_sleep):
    fn = AsyncMock(side_effect=[ExchangeRecoverableError("40015"), "ok"])

    returned_value = await retry_on_transient(
        fn, max_attempts=3, base_delay=1.0, retryable=ExchangeRecoverableError
    )

    assert returned_value == "ok"
    assert fn.call_count == 2


async def test_critical_error():
    fn = AsyncMock(side_effect=ExchangeCriticalError("auth failed"))

    with pytest.raises(ExchangeCriticalError, match="auth failed"):
        await retry_on_transient(fn, max_attempts=3)

    assert fn.call_count == 1


async def test_generic_exception():
    fn = AsyncMock(side_effect=RuntimeError("unexpected"))

    with pytest.raises(RuntimeError, match="unexpected"):
        await retry_on_transient(fn, max_attempts=3)

    assert fn.call_count == 1


async def test_with_positional_arguments():
    fn = AsyncMock(return_value="ok")

    returned_value = await retry_on_transient(fn, "a", "b")

    fn.assert_called_once_with("a", "b")
    assert returned_value == "ok"


@patch("robottraderslab._core.retry.asyncio.sleep", new_callable=AsyncMock)
async def test_a_first_retry_leaves_a_breadcrumb_without_warning(mock_sleep, caplog):
    fn = AsyncMock(side_effect=[ExchangeTransientError("502"), "ok"])
    fn.__qualname__ = "FuturesAccount.get_positions"

    with caplog.at_level(logging.DEBUG):
        await retry_on_transient(fn, max_attempts=3, base_delay=1.0)

    breadcrumb = caplog.records[-1]
    assert breadcrumb.levelno == logging.DEBUG
    assert "FuturesAccount.get_positions failed" in breadcrumb.getMessage()


@patch("robottraderslab._core.retry.asyncio.sleep", new_callable=AsyncMock)
async def test_a_repeated_retry_warns(mock_sleep, caplog):
    fn = AsyncMock(
        side_effect=[ExchangeTransientError("502"), ExchangeTransientError("502"), "ok"]
    )
    fn.__qualname__ = "FuturesAccount.get_positions"

    with caplog.at_level(logging.WARNING):
        await retry_on_transient(fn, max_attempts=3, base_delay=1.0)

    warning = caplog.records[-1]
    assert warning.levelno == logging.WARNING
    assert "attempt 2/3" in warning.getMessage()
