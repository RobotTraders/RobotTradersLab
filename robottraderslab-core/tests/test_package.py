import subprocess
import sys
from importlib.metadata import requires
from importlib.util import find_spec

import pytest
from packaging.requirements import Requirement

EXPECTED_DEPENDENCIES = {
    "httpx",
    "lazy-loader",
    "numpy",
    "pandas",
    "pydantic-core",
}

RESOLVE_EVERY_GATE_NAME = """
import sys

import robottraderslab_core
from robottraderslab_core import (
    _internal,
    engine,
    exceptions,
    exchanges,
    futures,
    ohlcv,
    strategies,
)

for gate in (
    robottraderslab_core,
    _internal,
    exceptions,
    exchanges,
    futures,
    strategies,
    ohlcv,
    engine,
):
    for name in gate.__all__:
        getattr(gate, name)

engine = [m for m in sys.modules if m.split(".")[0] == "robottraderslab"]
print(engine)
sys.exit(1 if engine else 0)
"""


IMPORT_ONE_GATE = """
import importlib
import sys

importlib.import_module("robottraderslab_core.{gate}")

loaded = [m for m in ("numpy", "pandas") if m in sys.modules]
print(loaded)
sys.exit(1 if loaded else 0)
"""


def test_the_distribution_provides_the_package():
    assert find_spec("robottraderslab_core") is not None


def test_the_distribution_depends_on_the_trading_surface_libraries_only():
    declared = requires("robottraderslab-core")

    assert declared is not None
    assert {Requirement(line).name for line in declared} == EXPECTED_DEPENDENCIES


def test_every_gate_resolves_without_loading_the_engine():
    completed = subprocess.run(
        [sys.executable, "-c", RESOLVE_EVERY_GATE_NAME],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("gate", ["strategies", "ohlcv", "engine", "exchanges"])
def test_importing_a_lazy_gate_loads_no_data_library(gate):
    completed = subprocess.run(
        [sys.executable, "-c", IMPORT_ONE_GATE.format(gate=gate)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
