import logging

import pytest

from robottraderslab._internal import handlers_on_root


def test_the_handlers_sit_on_the_root_logger_for_the_block_only():
    held = logging.NullHandler()

    with handlers_on_root([held]):
        during = held in logging.getLogger().handlers
    after = held in logging.getLogger().handlers

    assert during
    assert not after


def test_the_handlers_leave_the_root_logger_when_the_block_raises():
    held = logging.NullHandler()

    with pytest.raises(RuntimeError), handlers_on_root([held]):
        raise RuntimeError("the block failed")

    assert held not in logging.getLogger().handlers
