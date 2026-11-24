import logging
import sys
import types

import pytest

from robottraderslab.bootstrap.exchange_loader import load_exchange
from robottraderslab.exceptions import ExchangeCriticalError


class _StubDemoConnector:
    def __init__(self, *, demo_trading: bool) -> None:
        self.demo_trading = demo_trading


@pytest.fixture
def _stub_demo_connector(monkeypatch) -> None:
    connector_module = types.ModuleType("stub_demo_connector")
    connector_module.DemoConnector = _StubDemoConnector
    monkeypatch.setitem(sys.modules, "stub_demo_connector", connector_module)


class TestLoadExchange:
    async def test_simulator(self):
        settings = {
            "exchange": "simulator",
            "initial_balance": {"USDT": 1000.0},
            "maker_fee_rate": 0.001,
            "taker_fee_rate": 0.001,
        }

        exchange = await load_exchange(settings)

        assert exchange.__class__.__name__ == "SimulatedFuturesExchange"

    async def test_simulator_naming_no_convention_reserves_what_a_venue_holds_back(
        self,
    ):
        settings = {
            "exchange": "simulator",
            "initial_balance": {"USDT": 1000.0},
            "maker_fee_rate": 0.001,
            "taker_fee_rate": 0.0007,
        }

        exchange = await load_exchange(settings)

        assert exchange.placement_reserve_rate == pytest.approx(
            0.01 + 0.0007 * 1.005 + 0.0025
        )

    async def test_simulator_naming_the_cost_convention_reserves_nothing(self, caplog):
        settings = {
            "exchange": "simulator",
            "initial_balance": {"USDT": 1000.0},
            "maker_fee_rate": 0.001,
            "taker_fee_rate": 0.0007,
            "fee_mode": "cost",
        }

        with caplog.at_level(logging.WARNING):
            exchange = await load_exchange(settings)

        assert exchange.placement_reserve_rate == 0.0
        assert "takes no" not in caplog.text

    async def test_simulator_case_insensitive(self):
        settings = {
            "exchange": "SIMULATOR",
            "initial_balance": {"USDT": 1000.0},
            "maker_fee_rate": 0.001,
            "taker_fee_rate": 0.001,
        }

        exchange = await load_exchange(settings)

        assert exchange.__class__.__name__ == "SimulatedFuturesExchange"

    async def test_exchange_setting_removed_before_passing(self):
        settings = {
            "exchange": "simulator",
            "initial_balance": {"USDT": 1000.0},
            "maker_fee_rate": 0.001,
            "taker_fee_rate": 0.001,
        }

        await load_exchange(settings)

        assert "exchange" not in settings


class TestLoadExchangeErrors:
    async def test_missing_exchange_setting(self):
        settings = {"api_key": "test_key"}

        with pytest.raises(
            ExchangeCriticalError, match="`exchange` is a required config key"
        ):
            await load_exchange(settings)

    async def test_unknown_short_name(self):
        settings = {"exchange": "unknown_exchange"}

        with pytest.raises(
            ExchangeCriticalError,
            match="'unknown_exchange' not found in entry point group",
        ):
            await load_exchange(settings)

    async def test_nonexistent_module(self):
        settings = {"exchange": "nonexistent.module.SomeExchange"}

        with pytest.raises(
            ExchangeCriticalError,
            match="Module `nonexistent.module` can't be imported",
        ):
            await load_exchange(settings)

    async def test_non_dotted_name_not_in_entry_points(self):
        settings = {"exchange": "NotRegistered"}

        with pytest.raises(
            ExchangeCriticalError,
            match="'NotRegistered' not found in entry point group",
        ):
            await load_exchange(settings)


class TestSettingsContract:
    async def test_a_setting_the_connector_does_not_take_is_reported(self, caplog):
        settings = {
            "exchange": "simulator",
            "initial_balance": {"USDT": 1000.0},
            "maker_fee_rate": 0.001,
            "taker_fee_rate": 0.001,
            "rate_limit_safety_factor": 1.2,
        }

        with caplog.at_level(logging.WARNING):
            exchange = await load_exchange(settings)

        assert exchange.__class__.__name__ == "SimulatedFuturesExchange"
        assert "rate_limit_safety_factor" in caplog.text

    async def test_settings_the_connector_takes_are_not_reported(self, caplog):
        settings = {
            "exchange": "simulator",
            "initial_balance": {"USDT": 1000.0},
            "maker_fee_rate": 0.001,
            "taker_fee_rate": 0.001,
        }

        with caplog.at_level(logging.WARNING):
            await load_exchange(settings)

        assert "takes no" not in caplog.text

    async def test_a_demo_the_connector_does_not_offer_stops_the_load(self):
        settings = {
            "exchange": "simulator",
            "initial_balance": {"USDT": 1000.0},
            "maker_fee_rate": 0.001,
            "taker_fee_rate": 0.001,
            "demo_trading": True,
        }

        with pytest.raises(
            ExchangeCriticalError, match="'simulator' offers no demo environment"
        ):
            await load_exchange(settings)

    async def test_declining_a_demo_the_connector_does_not_offer_loads(self, caplog):
        settings = {
            "exchange": "simulator",
            "initial_balance": {"USDT": 1000.0},
            "maker_fee_rate": 0.001,
            "taker_fee_rate": 0.001,
            "demo_trading": False,
        }

        with caplog.at_level(logging.WARNING):
            exchange = await load_exchange(settings)

        assert exchange.__class__.__name__ == "SimulatedFuturesExchange"
        assert "demo_trading" in caplog.text

    async def test_a_factory_declaring_only_a_catch_all_has_nothing_reported(
        self, caplog
    ):
        settings = {"exchange": "argparse:Namespace", "flavour": "plain"}

        with caplog.at_level(logging.WARNING):
            await load_exchange(settings)

        assert "takes no" not in caplog.text

    async def test_a_factory_declaring_only_a_catch_all_offers_no_demo_either(self):
        settings = {"exchange": "argparse:Namespace", "demo_trading": True}

        with pytest.raises(
            ExchangeCriticalError,
            match="'argparse:Namespace' offers no demo environment",
        ):
            await load_exchange(settings)

    @pytest.mark.usefixtures("_stub_demo_connector")
    async def test_a_demo_the_connector_offers_reaches_it(self):
        settings = {
            "exchange": "stub_demo_connector:DemoConnector",
            "demo_trading": True,
        }

        connector = await load_exchange(settings)

        assert connector.demo_trading is True
