import importlib
import sys
import tomllib
from collections.abc import Callable
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest

from robottraderslab.scaffold import create_strategy_package

_ENTRY_POINT_GROUP = "robot_traders_lab.examples"

type RegisterExample = Callable[..., Path]


@pytest.fixture
def register_example(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> RegisterExample:
    """Create an importable package with example files and register it."""
    registered: list[EntryPoint] = []
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(
        "robottraderslab.scaffold.examples.entry_points",
        lambda group: registered if group == _ENTRY_POINT_GROUP else [],
    )

    def _register(name: str, package: str, *, with_examples: bool = True) -> Path:
        package_dir = tmp_path / package
        examples_dir = package_dir / "examples"
        if with_examples:
            examples_dir.mkdir(parents=True)
        else:
            package_dir.mkdir(parents=True)
        (package_dir / "__init__.py").touch()
        sys.modules.pop(package, None)
        importlib.invalidate_caches()
        registered.append(
            EntryPoint(name=name, value=package, group=_ENTRY_POINT_GROUP)
        )
        return examples_dir

    return _register


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path / "project" / "workspace"


@pytest.fixture
def package_folder(tmp_path: Path, workspace: Path) -> Path:
    folder = tmp_path / "my-strategy"
    create_strategy_package(folder, workspace)
    return folder


@pytest.fixture
def btc_4h_profile() -> dict[str, object]:
    return {
        "symbol": "BTC/USDC:USDC",
        "timeframe": "4h",
        "fast_ma_length": 10,
        "slow_ma_length": 30,
        "total_balance_ratio": 0.25,
    }


@pytest.fixture
def read_pyproject() -> Callable[[Path], dict]:
    def read(folder: Path) -> dict:
        return tomllib.loads((folder / "pyproject.toml").read_text(encoding="utf-8"))

    return read
