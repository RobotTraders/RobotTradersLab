from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest

from robottraderslab import Symbol
from robottraderslab._core import (
    OHLCVProviderProtocol,
    OHLCVRequirements,
    assemble_all_ohlcvs_from_requirements,
    assemble_available_ohlcvs_from_requirements,
)
from robottraderslab.exceptions import (
    ExchangeCriticalError,
    ExchangeTransientError,
    MissingOhlcvDataError,
    StrategyCriticalError,
)
from robottraderslab.strategies import OHLCVs, SymbolTimeframe


@pytest.fixture
def btc_symbol() -> Symbol:
    return Symbol.create("BTC/USDT:USDT")


@pytest.fixture
def eth_symbol() -> Symbol:
    return Symbol.create("ETH/USDT:USDT")


@pytest.fixture
def provider() -> Mock:
    bulk_provider = Mock()
    bulk_provider.fetch_ohlcvs = AsyncMock(
        side_effect=lambda pairs: {pair: object() for pair in pairs}
    )
    return bulk_provider


class TestAssembleOhlcvsFromRequirements:
    async def test_empty_requirements_raises(self, provider):
        empty_requirements = OHLCVRequirements()

        with pytest.raises(
            StrategyCriticalError, match="declared no OHLCV requirements"
        ):
            await assemble_all_ohlcvs_from_requirements(provider, empty_requirements)

    @patch(
        "robottraderslab._core.ohlcv_assembly._fetch_and_format",
        return_value={},
    )
    async def test_multiple_requirements(
        self, mock_fetch, provider, btc_symbol, eth_symbol
    ):
        requirements = OHLCVRequirements()
        requirements.add(btc_symbol, "4h")
        requirements.add(eth_symbol, "1d")

        await assemble_all_ohlcvs_from_requirements(provider, requirements)

        assert mock_fetch.call_args.args[1] == [
            SymbolTimeframe(btc_symbol, "4h"),
            SymbolTimeframe(eth_symbol, "1d"),
        ]

    @patch(
        "robottraderslab._core.ohlcv_assembly._fetch_and_format",
        return_value={},
    )
    async def test_provider_with_lookback_support(
        self, mock_fetch, provider, btc_symbol
    ):
        requirements = OHLCVRequirements()
        requirements.add(btc_symbol, "4h", lookback=999)

        await assemble_all_ohlcvs_from_requirements(provider, requirements)

        provider.set_required_lookbacks.assert_called_once_with({"4h": 999})

    @patch(
        "robottraderslab._core.ohlcv_assembly._fetch_and_format",
        return_value={},
    )
    async def test_provider_without_lookback_support(self, mock_fetch, btc_symbol):
        protocol_provider = Mock(spec=OHLCVProviderProtocol)
        requirements = OHLCVRequirements()
        requirements.add(btc_symbol, "4h", lookback=999)

        await assemble_all_ohlcvs_from_requirements(protocol_provider, requirements)

        assert not hasattr(protocol_provider, "set_required_lookbacks")

    @patch(
        "robottraderslab._core.ohlcv_assembly._fetch_and_format",
        return_value={},
    )
    async def test_zero_lookback_values(
        self, mock_fetch, provider, btc_symbol, eth_symbol
    ):
        requirements = OHLCVRequirements()
        requirements.add(btc_symbol, "4h", lookback=0)
        requirements.add(eth_symbol, "1d", lookback=888)

        await assemble_all_ohlcvs_from_requirements(provider, requirements)

        provider.set_required_lookbacks.assert_called_once_with({"1d": 888})

    @patch(
        "robottraderslab._core.ohlcv_assembly._fetch_and_format",
        return_value={},
    )
    async def test_max_lookback_per_timeframe(
        self, mock_fetch, provider, btc_symbol, eth_symbol
    ):
        """When multiple symbols share a timeframe, keep the largest lookback."""
        requirements = OHLCVRequirements()
        requirements.add(btc_symbol, "1d", lookback=20)
        requirements.add(eth_symbol, "1d", lookback=15)

        await assemble_all_ohlcvs_from_requirements(provider, requirements)

        provider.set_required_lookbacks.assert_called_once_with({"1d": 20})

    @patch(
        "robottraderslab._core.ohlcv_assembly._fetch_and_format",
        return_value={},
    )
    async def test_provider_with_bulk_fetch(
        self, mock_fetch, provider, btc_symbol, eth_symbol
    ):
        btc_frame = object()
        eth_frame = object()
        frames = {(btc_symbol, "4h"): btc_frame, (eth_symbol, "1d"): eth_frame}
        provider.fetch_ohlcvs.side_effect = lambda pairs: {
            pair: frames[pair] for pair in pairs
        }
        requirements = OHLCVRequirements()
        requirements.add(btc_symbol, "4h")
        requirements.add(eth_symbol, "1d")

        await assemble_all_ohlcvs_from_requirements(provider, requirements)

        assert provider.fetch_ohlcvs.await_count == 2
        provider.fetch_ohlcvs.assert_any_await([(btc_symbol, "4h")])
        provider.fetch_ohlcvs.assert_any_await([(eth_symbol, "1d")])
        fetcher = mock_fetch.call_args.args[0]
        assert fetcher(btc_symbol, "4h") is btc_frame
        assert fetcher(eth_symbol, "1d") is eth_frame

    @patch(
        "robottraderslab._core.ohlcv_assembly._fetch_and_format",
        return_value={},
    )
    async def test_provider_without_bulk_fetch(self, mock_fetch, btc_symbol):
        protocol_provider = Mock(spec=OHLCVProviderProtocol)
        frame = object()
        protocol_provider.fetch_ohlcv.return_value = frame
        requirements = OHLCVRequirements()
        requirements.add(btc_symbol, "4h")

        await assemble_all_ohlcvs_from_requirements(protocol_provider, requirements)

        protocol_provider.fetch_ohlcv.assert_called_once_with(btc_symbol, "4h")
        fetcher = mock_fetch.call_args.args[0]
        assert fetcher(btc_symbol, "4h") is frame
        assert mock_fetch.call_args.args[1] == [SymbolTimeframe(btc_symbol, "4h")]

    async def test_a_failing_pair_aborts_the_assembly(self, btc_symbol, eth_symbol):
        provider = Mock()

        async def fetch_ohlcvs(pairs):
            (symbol, timeframe) = pairs[0]
            if symbol == eth_symbol:
                raise Exception("not listed")
            return {(symbol, timeframe): object()}

        provider.fetch_ohlcvs = fetch_ohlcvs
        requirements = OHLCVRequirements()
        requirements.add(btc_symbol, "4h")
        requirements.add(eth_symbol, "4h")

        with pytest.raises(Exception, match="not listed"):
            await assemble_all_ohlcvs_from_requirements(provider, requirements)


def _frame(periods: int = 3) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=periods, freq="4h")
    return pd.DataFrame(
        {
            "open": [1.0] * periods,
            "high": [2.0] * periods,
            "low": [0.5] * periods,
            "close": [1.0] * periods,
            "volume": [1000.0] * periods,
        },
        index=dates,
    )


class TestAssembleAvailableOhlcvsFromRequirements:
    async def test_a_failing_pair_is_dropped_and_the_others_assemble(
        self, btc_symbol, eth_symbol, caplog
    ):
        provider = Mock()

        async def fetch_ohlcvs(pairs):
            (symbol, timeframe) = pairs[0]
            if symbol == eth_symbol:
                raise Exception("not listed")
            return {(symbol, timeframe): _frame()}

        provider.fetch_ohlcvs = fetch_ohlcvs
        requirements = OHLCVRequirements()
        requirements.add(btc_symbol, "4h")
        requirements.add(eth_symbol, "4h")

        ohlcvs = await assemble_available_ohlcvs_from_requirements(
            provider, requirements
        )

        assert ohlcvs is not None
        assert len(ohlcvs.column(btc_symbol, "4h", "close")) == 3
        with pytest.raises(MissingOhlcvDataError):
            ohlcvs.column(eth_symbol, "4h", "close")
        assert "skipping this cycle" in caplog.text

    async def test_returns_none_when_no_pair_could_be_fetched(self, btc_symbol, caplog):
        provider = Mock()
        provider.fetch_ohlcvs = AsyncMock(side_effect=Exception("not listed"))
        requirements = OHLCVRequirements()
        requirements.add(btc_symbol, "4h")

        ohlcvs = await assemble_available_ohlcvs_from_requirements(
            provider, requirements
        )

        assert ohlcvs is None
        assert "skipping this cycle" in caplog.text

    @patch("robottraderslab._core.retry.asyncio.sleep", new_callable=AsyncMock)
    async def test_a_transient_failure_is_retried_per_pair(
        self, mock_sleep, btc_symbol
    ):
        provider = Mock()
        attempts = {"count": 0}

        async def fetch_ohlcvs(pairs):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise ExchangeTransientError("502 Bad Gateway")
            (symbol, timeframe) = pairs[0]
            return {(symbol, timeframe): _frame()}

        provider.fetch_ohlcvs = fetch_ohlcvs
        requirements = OHLCVRequirements()
        requirements.add(btc_symbol, "4h")

        ohlcvs = await assemble_available_ohlcvs_from_requirements(
            provider, requirements
        )

        assert ohlcvs is not None
        assert len(ohlcvs.column(btc_symbol, "4h", "close")) == 3
        assert attempts["count"] == 2

    @pytest.mark.parametrize(
        "failure",
        [
            ExchangeCriticalError("auth"),
            StrategyCriticalError("auth"),
        ],
        ids=["exchange_critical", "strategy_critical"],
    )
    async def test_a_critical_error_propagates(self, btc_symbol, failure):
        provider = Mock()
        provider.fetch_ohlcvs = AsyncMock(side_effect=failure)
        requirements = OHLCVRequirements()
        requirements.add(btc_symbol, "4h")

        with pytest.raises(type(failure), match="auth"):
            await assemble_available_ohlcvs_from_requirements(provider, requirements)

    async def test_a_provider_without_bulk_fetch_still_isolates_failures(
        self, btc_symbol, eth_symbol, caplog
    ):
        protocol_provider = Mock(spec=OHLCVProviderProtocol)

        def fetch_ohlcv(symbol, timeframe):
            if symbol == eth_symbol:
                raise Exception("not listed")
            return _frame()

        protocol_provider.fetch_ohlcv.side_effect = fetch_ohlcv
        requirements = OHLCVRequirements()
        requirements.add(btc_symbol, "4h")
        requirements.add(eth_symbol, "4h")

        ohlcvs = await assemble_available_ohlcvs_from_requirements(
            protocol_provider, requirements
        )

        assert ohlcvs is not None
        assert len(ohlcvs.column(btc_symbol, "4h", "close")) == 3
        with pytest.raises(MissingOhlcvDataError):
            ohlcvs.column(eth_symbol, "4h", "close")

    async def test_empty_requirements_raises(self, btc_symbol):
        provider = Mock()

        with pytest.raises(
            StrategyCriticalError, match="declared no OHLCV requirements"
        ):
            await assemble_available_ohlcvs_from_requirements(
                provider, OHLCVRequirements()
            )


class _FakeProvider:
    def fetch_ohlcv(self, symbol: Symbol, timeframe: str) -> pd.DataFrame:
        freq = "h" if timeframe == "1h" else "D"
        periods = 3 if timeframe == "1h" else 2
        close_base = 104.0 if timeframe == "1h" else 50000.0
        index = pd.date_range("2024-01-01", periods=periods, freq=freq)
        return pd.DataFrame(
            {
                "open": [close_base - 4 + i for i in range(periods)],
                "high": [close_base + 1 + i for i in range(periods)],
                "low": [close_base - 5 + i for i in range(periods)],
                "close": [close_base + i for i in range(periods)],
                "volume": [1000.0 + i * 100 for i in range(periods)],
            },
            index=index,
        )


@pytest.fixture
def fake_provider() -> _FakeProvider:
    return _FakeProvider()


class TestAssembledFrames:
    async def test_builds_ohlcvs(self, fake_provider):
        symbol = Symbol.create("BTC/USDT:USDT")
        requirements = OHLCVRequirements()
        requirements.add(symbol, "1h")

        assembled_ohlcvs = await assemble_all_ohlcvs_from_requirements(
            fake_provider, requirements
        )

        assert isinstance(assembled_ohlcvs, OHLCVs)
        close_col = assembled_ohlcvs.column(symbol, "1h", "close")
        assert list(close_col) == [104.0, 105.0, 106.0]

    async def test_same_symbol_on_multiple_timeframes(self, fake_provider):
        btc = Symbol.create("BTC/USDT:USDT")
        requirements = OHLCVRequirements()
        requirements.add(btc, "1h")
        requirements.add(btc, "1d")

        assembled_ohlcvs = await assemble_all_ohlcvs_from_requirements(
            fake_provider, requirements
        )

        hourly = assembled_ohlcvs.column(btc, "1h", "close")
        daily = assembled_ohlcvs.column(btc, "1d", "close")
        assert len(hourly) == 3
        assert len(daily) == 2
        assert list(hourly) == [104.0, 105.0, 106.0]
        assert list(daily) == [50000.0, 50001.0]
