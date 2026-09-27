from datetime import datetime, timezone

import pytest

from robottraderslab import Symbol
from robottraderslab.exchanges import Execution, OrderSide


@pytest.fixture
def execution() -> Execution:
    return Execution(
        execution_id="execution-1",
        order_id="order-1",
        symbol=Symbol.create("BTC/USDT:USDT"),
        side=OrderSide.BUY,
        price=91213.0,
        quantity=0.1,
        timestamp=datetime(2026, 4, 24, 3, 6, 40, tzinfo=timezone.utc),
    )


class TestExecutionId:
    def test_two_executions_of_one_order_are_distinguished_by_execution_id(
        self, execution
    ):
        other_piece = Execution(
            execution_id="execution-2",
            order_id=execution.order_id,
            symbol=execution.symbol,
            side=execution.side,
            price=execution.price,
            quantity=execution.quantity,
            timestamp=execution.timestamp,
        )

        assert other_piece.order_id == execution.order_id
        assert other_piece.execution_id != execution.execution_id


class TestUnstatedFigures:
    def test_a_venue_stating_no_profit_no_fee_and_no_kind_reports_none(self, execution):
        assert execution.realised_profit is None
        assert execution.fee is None
        assert execution.kind is None

    def test_an_order_placed_outside_the_strategy_reports_no_client_order_id(
        self, execution
    ):
        assert execution.client_order_id is None
