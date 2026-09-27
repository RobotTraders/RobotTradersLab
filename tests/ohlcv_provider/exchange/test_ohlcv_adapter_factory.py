import asyncio
import logging
from collections.abc import Callable
from typing import Any

import pytest

from robottraderslab.ohlcv_provider.exchange.ohlcv_adapter_factory import (
    OhlcvAdapterFactory,
    dataset_name,
    resolve_adapter_class,
)

_GATED_ADAPTER_NAME = "gated_exchange"


class _StubOhlcvAdapter:
    def __init__(self) -> None:
        self.created_with: dict[str, Any] = {}
        self.exits = 0

    @classmethod
    def dataset_qualifiers(
        cls, *, flavour: str = "plain", **_kwargs: Any
    ) -> tuple[str, ...]:
        return () if flavour == "plain" else (flavour,)

    @classmethod
    def create(cls, *, flavour: str = "plain", **kwargs: Any) -> "_StubOhlcvAdapter":
        adapter = cls()
        adapter.created_with = {"flavour": flavour, **kwargs}
        return adapter

    async def __aenter__(self) -> "_StubOhlcvAdapter":
        return self

    async def __aexit__(self, *args: object) -> None:
        self.exits += 1


class _Gates:
    """The moments of an adapter's life the test decides."""

    def __init__(self) -> None:
        self.opening = asyncio.Event()
        self.may_open = asyncio.Event()
        self.closing = asyncio.Event()
        self.may_close = asyncio.Event()
        self.refuse_open = False
        self.may_open.set()
        self.may_close.set()


class _GatedAdapter:
    def __init__(self, gates: _Gates) -> None:
        self._gates = gates

    async def __aenter__(self) -> "_GatedAdapter":
        self._gates.opening.set()
        await self._gates.may_open.wait()
        if self._gates.refuse_open:
            raise OSError("the venue refused the connection")
        return self

    async def __aexit__(self, *_args: object) -> None:
        self._gates.closing.set()
        await self._gates.may_close.wait()


@pytest.fixture
def _stub_adapter_entry_points(
    make_registered_adapter: Callable[..., None],
) -> None:
    make_registered_adapter(_StubOhlcvAdapter, "ccxt", "stub_exchange")


@pytest.fixture
def gates() -> _Gates:
    return _Gates()


@pytest.fixture
def gated_adapters(
    make_registered_adapter: Callable[..., None], gates: _Gates
) -> list[_GatedAdapter]:
    built: list[_GatedAdapter] = []

    class _Registered(_GatedAdapter):
        @classmethod
        def create(cls, **_kwargs: Any) -> "_Registered":
            adapter = cls(gates)
            built.append(adapter)
            return adapter

    make_registered_adapter(_Registered, _GATED_ADAPTER_NAME)
    return built


async def _hold(
    factory: OhlcvAdapterFactory, holding: asyncio.Event, release: asyncio.Event
) -> None:
    async with factory.using(_GATED_ADAPTER_NAME):
        holding.set()
        await release.wait()


@pytest.mark.usefixtures("_stub_adapter_entry_points")
class TestOhlcvAdapterFactory:
    async def test_loads_adapter_registered_via_entry_point(self):
        factory = OhlcvAdapterFactory()

        async with factory.using("stub_exchange") as adapter:
            assert isinstance(adapter, _StubOhlcvAdapter)

    async def test_routes_ccxt_prefix_to_the_ccxt_entry_point(self):
        factory = OhlcvAdapterFactory()

        async with factory.using("ccxt_binance") as adapter:
            assert adapter.created_with["exchange_name"] == "binance"

    async def test_passes_config_through_to_the_adapter(self):
        factory = OhlcvAdapterFactory()

        async with factory.using("ccxt_binance", custom_param="test_value") as adapter:
            assert adapter.created_with["custom_param"] == "test_value"

    async def test_downloads_running_together_share_one_adapter(self):
        factory = OhlcvAdapterFactory()

        async with factory.using("ccxt_binance") as first:
            async with factory.using("ccxt_binance") as second:
                assert first is second

    async def test_a_download_after_the_last_one_left_opens_its_own(self):
        factory = OhlcvAdapterFactory()

        async with factory.using("ccxt_binance") as first:
            pass
        async with factory.using("ccxt_binance") as second:
            assert second is not first

    async def test_the_adapter_is_closed_once_no_download_holds_it(self):
        factory = OhlcvAdapterFactory()

        async with factory.using("ccxt_binance") as adapter:
            assert adapter.exits == 0

        assert adapter.exits == 1

    async def test_with_nonexistent_adapter(self):
        factory = OhlcvAdapterFactory()

        with pytest.raises(RuntimeError, match="OHLCV adapter .* not found"):
            async with factory.using("nonexistent_module_12345"):
                pass


@pytest.mark.usefixtures("_stub_adapter_entry_points")
class TestDatasetName:
    def test_a_source_on_its_defaults_keeps_its_own_name(self):
        assert dataset_name("stub_exchange", {}) == "stub_exchange"

    def test_a_choice_that_changes_the_candles_extends_the_name(self):
        assert (
            dataset_name("stub_exchange", {"flavour": "mark"}) == "stub_exchange-mark"
        )

    def test_a_setting_the_adapter_ignores_leaves_the_name_alone(self):
        assert dataset_name("stub_exchange", {"timeout_seconds": 30}) == "stub_exchange"


@pytest.mark.usefixtures("_stub_adapter_entry_points")
class TestResolveAdapterClass:
    def test_a_registered_name_resolves_to_its_class(self):
        assert resolve_adapter_class("stub_exchange") is _StubOhlcvAdapter

    def test_a_ccxt_prefixed_name_resolves_through_the_ccxt_entry_point(self):
        assert resolve_adapter_class("ccxt_binance") is _StubOhlcvAdapter

    def test_an_unregistered_name_resolves_to_none(self):
        assert resolve_adapter_class("nonexistent_module_12345") is None


@pytest.mark.usefixtures("_stub_adapter_entry_points")
class TestUnusableSettings:
    async def test_a_setting_the_adapter_does_not_take_is_reported(self, caplog):
        factory = OhlcvAdapterFactory()

        with caplog.at_level(logging.WARNING):
            async with factory.using("stub_exchange", tick_type="mark"):
                pass

        assert "`tick_type`" in caplog.text

    async def test_the_adapter_is_still_built_on_its_defaults(self):
        factory = OhlcvAdapterFactory()

        async with factory.using("stub_exchange", tick_type="mark") as adapter:
            assert adapter.created_with["flavour"] == "plain"

    async def test_a_setting_the_adapter_takes_is_not_reported(self, caplog):
        factory = OhlcvAdapterFactory()

        with caplog.at_level(logging.WARNING):
            async with factory.using("stub_exchange", flavour="mark"):
                pass

        assert caplog.text == ""


class _CatchAllOhlcvAdapter:
    @classmethod
    def create(cls, **_kwargs: Any) -> "_CatchAllOhlcvAdapter":
        return cls()

    async def __aenter__(self) -> "_CatchAllOhlcvAdapter":
        return self

    async def __aexit__(self, *args: object) -> None:
        pass


@pytest.fixture
def _catch_all_adapter_entry_point(
    make_registered_adapter: Callable[..., None],
) -> None:
    make_registered_adapter(_CatchAllOhlcvAdapter, "catch_all_exchange")


@pytest.mark.usefixtures("_catch_all_adapter_entry_point")
class TestCatchAllAdapterSettings:
    async def test_an_adapter_declaring_only_a_catch_all_has_nothing_reported(
        self, caplog
    ):
        factory = OhlcvAdapterFactory()

        with caplog.at_level(logging.WARNING):
            async with factory.using("catch_all_exchange", foo=1):
                pass

        assert "takes no" not in caplog.text


class TestRealAdaptersDeclaringTheirSettings:
    async def test_oanda_documented_settings_draw_no_warning(self, caplog):
        pytest.importorskip("robottraderslab_oanda")
        factory = OhlcvAdapterFactory()

        with caplog.at_level(logging.WARNING):
            async with factory.using(
                "oanda",
                access_token="token",
                account_id="101-004-1234567-001",
                practice=True,
                price="B",
            ):
                pass

        assert "takes no" not in caplog.text

    async def test_dukascopy_documented_settings_draw_no_warning(self, caplog):
        pytest.importorskip("robottraderslab_dukascopy")
        factory = OhlcvAdapterFactory()

        with caplog.at_level(logging.WARNING):
            async with factory.using("dukascopy", offer_side="ask"):
                pass

        assert "takes no" not in caplog.text


class TestAdapterOpenedWhileAnotherCloses:
    async def test_a_download_arriving_during_the_close_opens_one_adapter(
        self, gates: _Gates, gated_adapters: list[_GatedAdapter]
    ):
        factory = OhlcvAdapterFactory()
        first_holds, first_leaves = asyncio.Event(), asyncio.Event()
        second_holds, third_holds = asyncio.Event(), asyncio.Event()
        others_leave = asyncio.Event()

        first = asyncio.create_task(_hold(factory, first_holds, first_leaves))
        await first_holds.wait()
        gates.may_close.clear()
        first_leaves.set()
        await gates.closing.wait()

        second = asyncio.create_task(_hold(factory, second_holds, others_leave))
        gates.may_open.clear()
        gates.opening.clear()
        gates.may_close.set()
        await gates.opening.wait()
        third = asyncio.create_task(_hold(factory, third_holds, others_leave))
        await asyncio.sleep(0)

        assert len(gated_adapters) == 2

        gates.may_open.set()
        await second_holds.wait()
        await third_holds.wait()

        others_leave.set()
        await asyncio.gather(first, second, third)


class TestAnOpenThatFails:
    async def test_it_leaves_the_next_download_a_factory_holding_nothing(
        self, gates: _Gates, gated_adapters: list[_GatedAdapter]
    ):
        factory = OhlcvAdapterFactory()
        gates.refuse_open = True
        with pytest.raises(OSError, match="refused the connection"):
            async with factory.using(_GATED_ADAPTER_NAME):
                pass
        gates.refuse_open = False

        async with factory.using(_GATED_ADAPTER_NAME) as second:
            assert second is gated_adapters[1]

        assert gates.closing.is_set()
