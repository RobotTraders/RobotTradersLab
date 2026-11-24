import importlib
from itertools import chain
from typing import assert_type

import pytest

import robottraderslab
import robottraderslab_core
from robottraderslab import exchanges
from robottraderslab_core.exchanges import VenueParams

REEXPORTING_GATES = ("exceptions", "exchanges", "futures")


def _names_of_the_core_gate(gate_name):
    core_gate = importlib.import_module(f"robottraderslab_core.{gate_name}")
    return [(gate_name, name) for name in core_gate.__all__]


@pytest.mark.parametrize(
    ("gate_name", "name"),
    list(chain.from_iterable(map(_names_of_the_core_gate, REEXPORTING_GATES))),
)
def test_an_engine_gate_hands_out_the_core_object(gate_name, name):
    engine_gate = importlib.import_module(f"robottraderslab.{gate_name}")
    core_gate = importlib.import_module(f"robottraderslab_core.{gate_name}")

    assert getattr(engine_gate, name) is getattr(core_gate, name)


@pytest.mark.parametrize("name", ["Symbol", "TimeFrame"])
def test_the_root_hands_out_the_core_object(name):
    assert getattr(robottraderslab, name) is getattr(robottraderslab_core, name)


def test_the_exchanges_stub_types_a_name_as_the_core_defines_it():
    assert_type(exchanges.NO_VENUE_PARAMS, VenueParams)
