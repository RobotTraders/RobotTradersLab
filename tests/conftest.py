import json
import re
import sys
import types
from collections.abc import Callable, Iterator
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd
import pytest
from typer.testing import CliRunner

from robottraderslab._core.dynamic_class_loading import _resolve_entry_point

matplotlib.use("Agg")

_PAYLOAD_PATTERN = re.compile(r"window\.__CHART__ = (.*);")
_ADAPTER_GROUP = "robot_traders_lab.ohlcv_adapters"
_PLUGIN_MODULE = "registered_ohlcv_adapter_plugin"
_STUB_EXCHANGE = "stub_exchange"

type _AdapterRegistration = Callable[..., None]


class StubOhlcvAdapter:
    """An OHLCV adapter shaped as a plugin ships one."""

    @classmethod
    def create(cls, *, variant: str = "") -> "StubOhlcvAdapter":
        return cls()

    @classmethod
    def dataset_qualifiers(cls, *, variant: str = "") -> tuple[str, ...]:
        return (variant,) if variant else ()

    @classmethod
    def market_open_mask(cls, index: pd.DatetimeIndex) -> None:
        return None

    async def __aenter__(self) -> "StubOhlcvAdapter":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None


@pytest.fixture
def make_registered_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[_AdapterRegistration]:
    plugin_module = types.ModuleType(_PLUGIN_MODULE)
    monkeypatch.setitem(sys.modules, _PLUGIN_MODULE, plugin_module)
    served: dict[str, list[str]] = {}

    def entry_points(group: str) -> list[EntryPoint]:
        return [
            EntryPoint(
                name=name,
                value=f"{_PLUGIN_MODULE}:{_attribute(group, name)}",
                group=group,
            )
            for name in served.get(group, [])
        ]

    monkeypatch.setattr(
        "robottraderslab._core.dynamic_class_loading.entry_points", entry_points
    )

    def register(contribution: Any, *names: str, group: str = _ADAPTER_GROUP) -> None:
        for name in names:
            served.setdefault(group, []).append(name)
            setattr(plugin_module, _attribute(group, name), contribution)
        _resolve_entry_point.cache_clear()

    yield register
    _resolve_entry_point.cache_clear()


def _attribute(group: str, name: str) -> str:
    return f"{group.rpartition('.')[2]}_{name}"


@pytest.fixture
def cli() -> CliRunner:
    return CliRunner()


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The project folder a command runs from, holding an empty workspace."""
    folder = tmp_path / "RobotTradersLab"
    (folder / "workspace").mkdir(parents=True)
    (folder / "pyproject.toml").write_text("[project]\nname = 'lab'\n")
    monkeypatch.chdir(folder)
    return folder


@pytest.fixture
def stub_exchange(make_registered_adapter: _AdapterRegistration) -> str:
    """The name a configuration gives the stub adapter's venue."""
    make_registered_adapter(StubOhlcvAdapter, _STUB_EXCHANGE)
    return _STUB_EXCHANGE


@pytest.fixture
def browser_openings(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    openings: list[str] = []
    monkeypatch.setattr(
        "robottraderslab.chart.renderer.webbrowser.open", openings.append
    )
    return openings


@pytest.fixture
def embedded_payload() -> Callable[[str], dict[str, Any]]:
    """Reads back the data a rendered chart document draws."""

    def read(document: str) -> dict[str, Any]:
        embedded = _PAYLOAD_PATTERN.search(document)
        assert embedded is not None
        return json.loads(embedded.group(1))

    return read


@pytest.fixture
def read_payload(
    embedded_payload: Callable[[str], dict[str, Any]],
) -> Callable[[Path], dict[str, Any]]:
    def read(document: Path) -> dict[str, Any]:
        return embedded_payload(document.read_text(encoding="utf-8"))

    return read
