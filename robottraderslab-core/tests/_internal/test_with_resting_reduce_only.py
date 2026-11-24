import logging
from collections.abc import Callable

import pytest

from robottraderslab_core import Symbol
from robottraderslab_core.exchanges import (
    OrderModifyRequest,
    OrderRequest,
    OrderSide,
    with_resting_reduce_only,
)

BTC = Symbol.create("BTC/USDT:USDT")


def kept_flag_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.levelno == logging.WARNING and "reduce_only" in record.getMessage()
    ]


@pytest.fixture
def make_modification() -> Callable[..., OrderModifyRequest]:
    def make(*, reduce_only: bool) -> OrderModifyRequest:
        return OrderModifyRequest(
            order_id="order-7",
            order=OrderRequest(
                symbol=BTC,
                side=OrderSide.SELL,
                quantity=1.0,
                limit_price=105.0,
                reduce_only=reduce_only,
            ),
        )

    return make


@pytest.mark.parametrize(("resting", "requested"), [(True, False), (False, True)])
def test_a_resting_flag_the_request_contradicts_replaces_it(
    make_modification, resting, requested
):
    request = make_modification(reduce_only=requested)

    replacement = with_resting_reduce_only(request, resting)

    assert replacement == OrderRequest(
        symbol=BTC,
        side=OrderSide.SELL,
        quantity=1.0,
        limit_price=105.0,
        reduce_only=resting,
    )


def test_the_contradicted_flag_is_named_at_warning_with_the_symbol_and_order(
    make_modification, caplog
):
    request = make_modification(reduce_only=False)

    with caplog.at_level(logging.WARNING):
        with_resting_reduce_only(request, True)

    (record,) = kept_flag_records(caplog)
    assert record.levelno == logging.WARNING
    assert "BTC/USDT:USDT" in record.getMessage()
    assert "order-7" in record.getMessage()
    assert "reduce_only=False" in record.getMessage()


def test_a_request_agreeing_with_the_resting_flag_is_left_alone(
    make_modification, caplog
):
    request = make_modification(reduce_only=True)

    with caplog.at_level(logging.WARNING):
        replacement = with_resting_reduce_only(request, True)

    assert replacement == request.order
    assert kept_flag_records(caplog) == []


@pytest.mark.parametrize("requested", [True, False])
def test_a_resting_order_whose_venue_states_no_flag_takes_the_requests(
    make_modification, requested, caplog
):
    request = make_modification(reduce_only=requested)

    with caplog.at_level(logging.WARNING):
        replacement = with_resting_reduce_only(request, None)

    assert replacement == request.order
    assert kept_flag_records(caplog) == []
