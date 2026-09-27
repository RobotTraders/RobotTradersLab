import importlib
import sys
from collections.abc import Callable
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest

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
