import json
import logging
import os
import re
import stat
import subprocess
import sys
import types
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import matplotlib
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from robottraderslab import Symbol, TimeFrame
from robottraderslab._internal import hold_instance_lock
from robottraderslab._internal.dynamic_class_loading import _resolve_entry_point
from robottraderslab.exchanges import (
    Balance,
    FuturesExchangeProtocol,
    Instrument,
    OrderProtocol,
    OrderSide,
    OrderType,
    TimeInForce,
    TriggerType,
)
from robottraderslab.futures import FuturesAccount, read_account_state
from robottraderslab.strategies import AccountSnapshot, AccountSnapshots, OHLCVs
from robottraderslab_core.engine import name_process
from robottraderslab_core.exchanges import BaseExchangeAction

matplotlib.use("Agg")

_PAYLOAD_PATTERN = re.compile(r"window\.__CHART__ = (.*);")
_ADAPTER_GROUP = "robot_traders_lab.ohlcv_adapters"
_PLUGIN_MODULE = "registered_ohlcv_adapter_plugin"
_STUB_EXCHANGE = "stub_exchange"
_UNNAMED_PROCESS_LAUNCH = datetime.now(UTC)

type _AdapterRegistration = Callable[..., None]
type UvAnswer = Callable[..., Path]

_FAKE_UV = """
import os
import sys
from pathlib import Path

Path(os.environ["FAKE_UV_CALL"]).write_text(
    " ".join(sys.argv[1:]) + "\\n" + os.getcwd(), encoding="utf-8"
)
sys.stderr.write(os.environ["FAKE_UV_STDERR"])
sys.exit(int(os.environ["FAKE_UV_EXIT"]))
"""


@pytest.fixture(autouse=True)
def _restore_process_identity() -> Iterator[None]:
    yield
    name_process("", _UNNAMED_PROCESS_LAUNCH)


@pytest.fixture
def logged_once(caplog: pytest.LogCaptureFixture) -> Callable[[str], logging.LogRecord]:
    """Return the one captured record whose message holds a text.

    The lookup fails unless exactly one record holds it.
    """

    def find(text: str) -> logging.LogRecord:
        (record,) = (r for r in caplog.records if text in r.getMessage())
        return record

    return find


@pytest.fixture
def make_ohlcv_dict() -> Callable[..., dict[Symbol, pd.DataFrame]]:
    def make(
        timestamps: list[str],
        symbols: list[Symbol],
        close_values: list[list[float]],
    ) -> dict[Symbol, pd.DataFrame]:
        """Build a per-symbol dict of DataFrames matching the assembler format.

        Args:
            timestamps: ISO timestamp strings for the index.
            symbols: Symbols to include as keys.
            close_values: One inner list per symbol, each with len(timestamps) values.
        """
        index = pd.DatetimeIndex([pd.Timestamp(t) for t in timestamps])
        return {
            symbol: pd.DataFrame(
                {
                    "open": symbol_close,
                    "high": symbol_close,
                    "low": symbol_close,
                    "close": symbol_close,
                    "volume": [1.0] * len(timestamps),
                },
                index=index,
            )
            for symbol, symbol_close in zip(symbols, close_values)
        }

    return make


@pytest.fixture
def make_dated_ohlcvs(
    make_ohlcv_dict: Callable[..., dict[Symbol, pd.DataFrame]],
) -> Callable[..., OHLCVs]:
    def make(
        symbol: Symbol, timeframe: TimeFrame, freq: str, closes: list[float]
    ) -> OHLCVs:
        """Build the candles of one symbol on one timeframe, from 2024-01-01 UTC.

        Args:
            freq: The pandas frequency of the timeframe, such as "D" for "1d".
            closes: One close per candle.
        """
        dates = pd.date_range("2024-01-01", periods=len(closes), freq=freq, tz="UTC")
        return OHLCVs(
            {
                timeframe: make_ohlcv_dict(
                    [str(date) for date in dates], [symbol], [closes]
                )
            }
        )

    return make


@pytest.fixture
def candles_where_raised() -> Callable[[OHLCVs, Symbol, TimeFrame, str], list[int]]:
    def read(
        ohlcvs: OHLCVs, symbol: Symbol, timeframe: TimeFrame, column: str
    ) -> list[int]:
        """Return the indexes of the candles a column is raised on."""
        return np.flatnonzero(ohlcvs.column(symbol, timeframe, column)).tolist()

    return read


@pytest.fixture
def make_account_snapshots() -> Callable[..., AccountSnapshots]:
    def make(
        account_name: str, usdt_total: float, **snapshot_fields: Any
    ) -> AccountSnapshots:
        """Build the snapshots of one account holding `usdt_total` USDT, none locked.

        Args:
            snapshot_fields: The other fields of the account's snapshot.
        """
        account_snapshot = AccountSnapshot(
            account_name=account_name,
            balances={"USDT": Balance(locked=0.0, total=usdt_total)},
            **snapshot_fields,
        )
        return AccountSnapshots({account_name: account_snapshot})

    return make


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
def fresh_entry_point_lookups() -> Iterator[None]:
    _resolve_entry_point.cache_clear()
    yield
    _resolve_entry_point.cache_clear()


@pytest.fixture
def install_entry_points(
    monkeypatch: pytest.MonkeyPatch, fresh_entry_point_lookups: None
) -> Callable[[list[EntryPoint]], Mock]:
    """The distributions installed are the entry points given, and the reader
    the lookup goes through is returned.
    """

    def install(entries: list[EntryPoint]) -> Mock:
        reader = Mock(
            side_effect=lambda group: [
                entry for entry in entries if entry.group == group
            ]
        )
        monkeypatch.setattr(
            "robottraderslab._internal.dynamic_class_loading.entry_points", reader
        )
        return reader

    return install


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
        "robottraderslab._internal.dynamic_class_loading.entry_points", entry_points
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
def make_raiser() -> Callable[[BaseException], Callable[..., None]]:
    def make(error: BaseException) -> Callable[..., None]:
        def raiser(*_args: Any, **_kwargs: Any) -> None:
            raise error

        return raiser

    return make


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
def fake_uv(monkeypatch: pytest.MonkeyPatch) -> str:
    """The path the engine finds uv at, which no test runs."""
    path = "/fake/bin/uv"
    monkeypatch.setenv("UV", path)
    return path


@pytest.fixture
def locked_project(project: Path) -> Path:
    (project / "uv.lock").write_text("version = 1\n")
    return project


@pytest.fixture
def make_uv_answer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> UvAnswer:
    bin_dir = tmp_path / "fake-uv"
    bin_dir.mkdir()
    script = bin_dir / "fake_uv.py"
    script.write_text(_FAKE_UV, encoding="utf-8")
    if os.name == "nt":
        command = bin_dir / "uv.bat"
        command.write_text(f'@"{sys.executable}" "{script}" %*\n')
    else:
        command = bin_dir / "uv"
        command.write_text(f"#!{sys.executable}\n{_FAKE_UV}")
        command.chmod(command.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("UV", str(command))
    call = tmp_path / "uv-call.txt"
    monkeypatch.setenv("FAKE_UV_CALL", str(call))

    def answer(stderr: str, exit_code: int = 0) -> Path:
        monkeypatch.setenv("FAKE_UV_STDERR", stderr)
        monkeypatch.setenv("FAKE_UV_EXIT", str(exit_code))
        return call

    return answer


@pytest.fixture
def installs(
    fake_uv: str, monkeypatch: pytest.MonkeyPatch
) -> list[tuple[list[str], Path]]:
    """The uv commands the strategy install ran, each with the folder it ran in,
    all succeeding.
    """
    runs: list[tuple[list[str], Path]] = []

    def run(
        command: list[str], cwd: Path, **options: Any
    ) -> subprocess.CompletedProcess[str]:
        runs.append((command, cwd))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("subprocess.run", run)
    return runs


@pytest.fixture
def failed_install(fake_uv: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A strategy install whose uv exits 2."""
    monkeypatch.setattr(
        "subprocess.run",
        lambda command, cwd, **options: subprocess.CompletedProcess(command, 2),
    )


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


@pytest.fixture
def make_venue() -> Callable[..., Mock]:
    def make(
        quantity_step: float = 1e-8, minimum_order_value: float | None = None
    ) -> Mock:
        venue = Mock(spec=FuturesExchangeProtocol)
        venue.placement_reserve_rate = 0.0
        venue.get_account_state.side_effect = partial(read_account_state, venue)
        venue.get_instrument.return_value = Instrument(
            quantity_step=quantity_step,
            minimum_order_value=minimum_order_value,
            maximum_leverage=None,
            maximum_order_quantity=None,
            maximum_market_order_quantity=None,
        )
        return venue

    return make


@pytest.fixture
def make_futures_account(
    make_venue: Callable[..., Mock],
) -> Callable[..., FuturesAccount]:
    def make(**venue_options: Any) -> FuturesAccount:
        """Build an account over a fresh venue double.

        Args:
            venue_options: The arguments `make_venue` takes.
        """
        return FuturesAccount(make_venue(**venue_options))

    return make


@dataclass(kw_only=True)
class RestingOrder:
    order_id: str = "order-7"
    symbol: Symbol = Symbol.create("BTC/USDT:USDT")
    kind: OrderType = OrderType.LIMIT
    side: OrderSide | None = OrderSide.SELL
    quantity: float | None = 2.0
    remaining_quantity: float = 2.0
    limit_price: float | None = 105.0
    trigger_price: float | None = None
    reduce_only: bool | None = True
    time_in_force: TimeInForce | None = TimeInForce.POST_ONLY
    trigger_type: TriggerType | None = None
    stop_loss_price: float | None = None
    stop_loss_trigger_type: TriggerType | None = None
    take_profit_price: float | None = None
    take_profit_trigger_type: TriggerType | None = None
    client_order_id: str | None = field(
        default_factory=lambda: (
            BaseExchangeAction(
                symbol=Symbol.create("BTC/USDT:USDT"), tag="1h-fast"
            ).client_order_id
        )
    )
    trailing_distance: float | None = None


@pytest.fixture
def create_resting_order() -> Callable[..., OrderProtocol]:
    return RestingOrder


@pytest.fixture
def create_lock_file() -> Callable[[Path], None]:
    def take_and_release_lock(config: Path) -> None:
        """Take the bot's instance lock and release it, leaving the lock file."""
        with hold_instance_lock(config):
            pass

    return take_and_release_lock


@pytest.fixture
def create_earlier_lock() -> Callable[[Path, Path], Path]:
    def write_earlier_lock(workspace: Path, config: Path) -> Path:
        """Write the lock of the earlier placement, under the workspace's `.rtlab`."""
        lock_file = (
            workspace
            / ".rtlab"
            / config.parent.relative_to(workspace)
            / config.with_suffix(".lock").name
        )
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        lock_file.write_text(str(config.resolve()), encoding="utf-8")
        return lock_file

    return write_earlier_lock


@pytest.fixture
def names_in_tool_directory() -> Callable[[Path], list[str]]:
    def list_tool_directory(folder: Path) -> list[str]:
        """List the names in the `.rtlab` folder of the folder, sorted."""
        return sorted(path.name for path in (folder / ".rtlab").iterdir())

    return list_tool_directory
