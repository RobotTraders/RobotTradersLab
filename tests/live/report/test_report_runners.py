import logging
from collections.abc import Awaitable, Callable, Iterator
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest

from robottraderslab._core import Balance, Execution, OrderSide, Symbol
from robottraderslab.bootstrap import BotConfig
from robottraderslab.exceptions import StrategyCriticalError
from robottraderslab.exchanges import FuturesExchangeProtocol
from robottraderslab.futures import FuturesAccount
from robottraderslab.live.report.runners import run_report
from robottraderslab.strategies import StrategyRequirements, TradingMode

INSIDE_THE_WINDOW = datetime.now(UTC) - timedelta(days=5)

_PROFILES = [
    {"symbol": "BTC/USDT:USDT", "timeframe": "1h", "trix_length": 31},
    {"symbol": "ETH/USDT:USDT", "timeframe": "4h", "trix_length": 15},
]


@pytest.fixture
def bot_config() -> BotConfig:
    return BotConfig.model_validate(
        {
            "strategy": {"strategy_class": "impulse", "profiles": _PROFILES},
            "live": {
                "trading_account": {"exchange": "bitget"},
                "ohlcv_provider": {"ohlcv_provider": "mock"},
            },
            "config_dir": "workspace/impulse",
        }
    )


@pytest.fixture
def exchange() -> AsyncMock:
    exchange = AsyncMock(spec=FuturesExchangeProtocol)
    exchange.get_executions_since.return_value = []
    exchange.get_open_positions.return_value = {}
    exchange.get_balances.return_value = {"USDT": Balance(locked=0.0, total=1000.0)}
    return exchange


@pytest.fixture
def account() -> Mock:
    return Mock(spec=FuturesAccount)


@pytest.fixture
def analyser() -> Mock:
    return Mock()


def _setup_declaring(
    lookbacks: dict[str, int],
) -> Callable[[StrategyRequirements], Awaitable[None]]:
    """Return a fake `setup` shared by every test here, varying only the lookback it declares."""

    async def setup(requirements: StrategyRequirements) -> None:
        for profile in _PROFILES:
            timeframe = profile["timeframe"]
            requirements.ohlcv.add(
                Symbol.create(profile["symbol"]), timeframe, lookbacks.get(timeframe, 0)
            )

    return setup


@pytest.fixture
def strategy() -> Mock:
    strategy = Mock()
    strategy.setup = AsyncMock(side_effect=_setup_declaring({}))
    return strategy


@pytest.fixture
def collaborators(
    exchange: AsyncMock, account: Mock, strategy: Mock, analyser: Mock
) -> Iterator[dict[str, Mock]]:
    with (
        patch(
            "robottraderslab.live.report.runners.load_single_account_with_exchange",
            return_value=(exchange, account),
        ) as load_single_account_with_exchange,
        patch(
            "robottraderslab.live.report.runners.load_strategy",
            return_value=strategy,
        ) as load_strategy,
        patch(
            "robottraderslab.live.report.runners.load_ohlcv_provider"
        ) as load_ohlcv_provider,
        patch("robottraderslab.live.report.runners.Analyser") as analyser_class,
    ):
        load_ohlcv_provider.return_value.fetch_ohlcvs = AsyncMock(
            side_effect=lambda pairs: {pair: pd.DataFrame() for pair in pairs}
        )
        analyser_class.from_executions.return_value = analyser
        yield {
            "load_single_account_with_exchange": load_single_account_with_exchange,
            "load_strategy": load_strategy,
            "load_ohlcv_provider": load_ohlcv_provider,
            "analyser_class": analyser_class,
            "strategy": strategy,
        }


def _execution(**overrides: object) -> Execution:
    arguments = {
        "execution_id": "execution-1",
        "order_id": "order-1",
        "symbol": Symbol.create("BTC/USDT:USDT"),
        "side": OrderSide.BUY,
        "price": 100.0,
        "quantity": 1.0,
        "timestamp": INSIDE_THE_WINDOW,
        "kind": "market",
        "realised_profit": 0.0,
        "fee": 0.5,
    }
    arguments.update(overrides)
    return Execution(**arguments)  # type: ignore[arg-type]


def _handed_to_the_analyser(collaborators: dict[str, Mock]) -> dict[str, Any]:
    return collaborators["analyser_class"].from_executions.call_args.kwargs


class TestAccountStateIsRequested:
    def test_executions_are_requested_for_every_profile_symbol(
        self, bot_config, exchange, collaborators
    ):
        run_report(bot_config, days=30)

        called_symbols = exchange.get_executions_since.call_args.args[1]
        assert set(called_symbols) == {
            Symbol.create("BTC/USDT:USDT"),
            Symbol.create("ETH/USDT:USDT"),
        }

    def test_open_positions_are_requested_for_every_profile_symbol(
        self, bot_config, exchange, collaborators
    ):
        run_report(bot_config, days=30)

        called_symbols = exchange.get_open_positions.call_args.args[0]
        assert set(called_symbols) == {
            Symbol.create("BTC/USDT:USDT"),
            Symbol.create("ETH/USDT:USDT"),
        }

    def test_the_window_starts_the_requested_days_back(
        self, bot_config, exchange, collaborators
    ):
        run_report(bot_config, days=7)

        since = exchange.get_executions_since.call_args.args[0]
        assert (
            abs((since - (datetime.now(UTC) - timedelta(days=7))).total_seconds()) < 60
        )


class TestTheBalanceTheWindowOpenedOn:
    def test_the_balance_is_requested_for_every_profile_symbol(
        self, bot_config, exchange, collaborators
    ):
        run_report(bot_config, days=30)

        called_symbols = exchange.get_balances.call_args.args[0]
        assert set(called_symbols) == {
            Symbol.create("BTC/USDT:USDT"),
            Symbol.create("ETH/USDT:USDT"),
        }


class TestTheReportWindow:
    def test_the_provider_receives_the_window_unmodified(
        self, bot_config, exchange, collaborators
    ):
        run_report(bot_config, days=30)

        since = exchange.get_executions_since.call_args.args[0]
        fetch_start, fetch_end = collaborators[
            "load_ohlcv_provider"
        ].return_value.set_dates.call_args[0]
        assert fetch_start == since.replace(tzinfo=None).isoformat(
            sep=" ", timespec="seconds"
        )
        assert datetime.now(UTC) - datetime.fromisoformat(fetch_end).replace(
            tzinfo=UTC
        ) < timedelta(minutes=1)

    def test_every_declared_market_is_fetched(self, bot_config, collaborators):
        run_report(bot_config, days=30)

        provider = collaborators["load_ohlcv_provider"].return_value
        fetched_pairs = {
            pair
            for call in provider.fetch_ohlcvs.call_args_list
            for pair in call.args[0]
        }
        assert fetched_pairs == {
            (Symbol.create("BTC/USDT:USDT"), "1h"),
            (Symbol.create("ETH/USDT:USDT"), "4h"),
        }


class TestTheStrategyIsBuilt:
    def test_the_strategy_is_built_for_the_loaded_account_in_live_mode(
        self, bot_config, account, collaborators
    ):
        run_report(bot_config, days=30)

        call = collaborators["load_strategy"].call_args
        assert call.args[0]["strategy_class"] == "impulse"
        assert call.args[0]["profiles"] == _PROFILES
        assert call.kwargs["account"] is account
        assert call.kwargs["trading_system"].trading_mode == TradingMode.LIVE
        assert call.kwargs["config_dir"] == bot_config.config_dir

    def test_one_venue_connection_serves_the_account_state_and_the_strategy(
        self, bot_config, collaborators
    ):
        run_report(bot_config, days=30)

        collaborators["load_single_account_with_exchange"].assert_called_once_with(
            bot_config.live.trading_account, {}
        )


class TestStrategyDeclaredLookback:
    def test_the_declared_lookback_reaches_the_provider(
        self, bot_config, collaborators
    ):
        collaborators["strategy"].setup.side_effect = _setup_declaring({"4h": 330})

        run_report(bot_config, days=30)

        provider = collaborators["load_ohlcv_provider"].return_value
        provider.set_required_lookbacks.assert_called_once_with({"4h": 330})

    def test_no_declared_lookback_sets_no_lookback_on_the_provider(
        self, bot_config, collaborators
    ):
        run_report(bot_config, days=30)

        provider = collaborators["load_ohlcv_provider"].return_value
        provider.set_required_lookbacks.assert_not_called()


class TestAnalysisIsBuilt:
    def test_the_analyser_is_built_from_the_configured_provider(
        self, bot_config, collaborators
    ):
        run_report(bot_config, days=30)

        handed = _handed_to_the_analyser(collaborators)
        assert (
            handed["ohlcv_provider"]
            is collaborators["load_ohlcv_provider"].return_value
        )

    def test_the_analyser_is_built_with_the_configured_profiles_as_profiles(
        self, bot_config, collaborators
    ):
        run_report(bot_config, days=30)

        assert _handed_to_the_analyser(collaborators)["profiles"] == _PROFILES

    def test_the_declared_report_section_is_the_one_measured_by(
        self, exchange, collaborators
    ):
        declared = BotConfig.model_validate(
            {
                "strategy": {"strategy_class": "impulse", "profiles": _PROFILES},
                "live": {
                    "trading_account": {"exchange": "bitget"},
                    "ohlcv_provider": {"ohlcv_provider": "mock"},
                },
                "report": {"reference_symbol": "BTC/USDT:USDT"},
            }
        )

        run_report(declared, days=30)

        handed = _handed_to_the_analyser(collaborators)["report_config"]
        assert handed.reference_symbol == "BTC/USDT:USDT"

    def test_the_window_read_is_the_window_measured(
        self, bot_config, exchange, collaborators
    ):
        run_report(bot_config, days=30)

        handed = _handed_to_the_analyser(collaborators)
        assert handed["since"] == exchange.get_executions_since.call_args.args[0]
        assert (datetime.now(UTC) - handed["until"]).total_seconds() < 60


class TestChartIsDrawn:
    def test_the_chart_is_drawn_with_the_configured_indicators(
        self, bot_config, analyser, collaborators
    ):
        run_report(bot_config, days=30)

        assert analyser.plot_candlesticks.call_args.kwargs["indicators_name"] == (
            "impulse"
        )

    def test_the_window_start_reaches_the_chart(
        self, bot_config, exchange, analyser, collaborators
    ):
        run_report(bot_config, days=30)

        since = exchange.get_executions_since.call_args.args[0]
        assert analyser.plot_candlesticks.call_args.kwargs["start_date"] == (
            since.replace(tzinfo=None).isoformat(sep=" ", timespec="seconds")
        )


def test_without_profiles_there_is_nothing_to_chart(bot_config, collaborators):
    bot_config.strategy.profiles = []

    with pytest.raises(StrategyCriticalError, match="strategy.profiles"):
        run_report(bot_config, days=30)

    collaborators["load_single_account_with_exchange"].assert_not_called()


def test_the_chart_service_draws_from_the_configured_provider(
    bot_config, collaborators
):
    run_report(bot_config, days=30)

    collaborators["load_ohlcv_provider"].assert_called_once_with(
        {"ohlcv_provider": "mock"}
    )


class TestReadingTheWindowOnce:
    def test_executions_are_read_once(self, bot_config, exchange, collaborators):
        exchange.get_executions_since.return_value = []

        run_report(bot_config, days=30)

        assert exchange.get_executions_since.call_count == 1

    def test_an_unpairable_close_is_read_once_like_any_other(
        self, bot_config, exchange, collaborators
    ):
        close_with_nothing_open = _execution(side=OrderSide.SELL, realised_profit=5.0)
        exchange.get_executions_since.return_value = [close_with_nothing_open]

        run_report(bot_config, days=30)

        assert exchange.get_executions_since.call_count == 1

    def test_the_window_is_read_from_the_requested_start(
        self, bot_config, exchange, collaborators
    ):
        run_report(bot_config, days=30)

        since = exchange.get_executions_since.call_args.args[0]
        assert (datetime.now(UTC) - since).days == 30


class TestTheCoveredWindowIsStated:
    def test_the_window_covered_is_logged(
        self, bot_config, exchange, collaborators, caplog
    ):
        with caplog.at_level(logging.INFO):
            run_report(bot_config, days=30)

        assert "Covered" in caplog.text

    def test_the_oldest_execution_reached_is_logged(
        self, bot_config, exchange, collaborators, caplog
    ):
        exchange.get_executions_since.return_value = [
            _execution(timestamp=INSIDE_THE_WINDOW)
        ]

        with caplog.at_level(logging.INFO):
            run_report(bot_config, days=30)

        assert f"reaching back to {INSIDE_THE_WINDOW:%Y-%m-%d}" in caplog.text

    def test_a_window_the_venue_answered_nothing_for_says_so(
        self, bot_config, exchange, collaborators, caplog
    ):
        exchange.get_executions_since.return_value = []

        with caplog.at_level(logging.INFO):
            run_report(bot_config, days=30)

        assert "with no execution in reach" in caplog.text
