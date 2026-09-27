import logging
from datetime import UTC, datetime
from unittest.mock import Mock

import pandas as pd
import pytest

from robottraderslab._core import (
    Balance,
    Execution,
    OHLCVProviderProtocol,
    OrderSide,
    PositionSide,
    PositionSnapshot,
    Symbol,
)
from robottraderslab.analyser import AnalysisInputs
from robottraderslab.analyser.execution_inputs import build_execution_inputs
from robottraderslab.bootstrap import ReportConfig
from robottraderslab.exceptions import StrategyCriticalError

_BTC = Symbol.create("BTC/USDT:USDT")
_ETH_USDC = Symbol.create("ETH/USDC:USDC")
_SINCE = datetime(2026, 8, 1, tzinfo=UTC)
_UNTIL = datetime(2026, 8, 3, tzinfo=UTC)
_HELD = {"USDT": Balance(locked=0.0, total=1000.0)}


def _analysis_input(**overrides: object) -> AnalysisInputs:
    arguments = {
        "executions": [],
        "balances": _HELD,
        "positions": {},
        "symbols": [_BTC],
        "since": _SINCE,
        "until": _UNTIL,
        "ohlcv_provider": Mock(spec=OHLCVProviderProtocol),
        "report_config": ReportConfig(),
    }
    arguments.update(overrides)
    return build_execution_inputs(**arguments)  # type: ignore[arg-type]


def _execution(**overrides: object) -> Execution:
    arguments = {
        "execution_id": "execution-1",
        "order_id": "order-1",
        "symbol": _BTC,
        "side": OrderSide.SELL,
        "price": 100.0,
        "quantity": 1.0,
        "timestamp": datetime(2026, 8, 2, tzinfo=UTC),
        "realised_profit": 50.0,
        "fee": 1.0,
    }
    arguments.update(overrides)
    return Execution(**arguments)  # type: ignore[arg-type]


def _position() -> PositionSnapshot:
    return PositionSnapshot(
        symbol=_BTC,
        side=PositionSide.LONG,
        quantity=1.0,
        average_entry_price=100.0,
        entry_time=datetime(2026, 8, 1, tzinfo=UTC),
        leverage=1.0,
        liquidation_price=10.0,
    )


class TestAssembly:
    def test_the_open_positions_reflect_the_venues_read(self):
        analysis_input = _analysis_input(
            positions={_BTC: _position()},
        )

        assert analysis_input.open_positions.iloc[0]["symbol"] == "BTC/USDT:USDT"

    def test_executions_are_paired_into_trades(self):
        opened = _execution(side=OrderSide.BUY, realised_profit=None, fee=None)
        closed = _execution(realised_profit=5.0, fee=1.0)

        analysis_input = _analysis_input(
            executions=[opened, closed],
        )

        assert len(analysis_input.trades) == 1

    def test_the_equity_curve_accumulates_booked_executions(self):
        booked = _execution(realised_profit=50.0, fee=1.0)

        analysis_input = _analysis_input(
            executions=[booked],
        )

        curve = analysis_input.equity_curve
        assert curve[_UNTIL] - curve[_SINCE] == pytest.approx(49.0)

    def test_the_equity_curve_holds_the_balance_with_nothing_booked(self):
        analysis_input = _analysis_input()

        assert analysis_input.equity_curve[_UNTIL] == pytest.approx(1000.0)


class TestTheDeclaredReference:
    def test_the_reference_is_priced_over_the_window(self):
        provider = Mock(spec=OHLCVProviderProtocol)
        provider.get_all_cached_ohlcv_for_symbol.return_value = pd.DataFrame(
            {"close": [100.0, 110.0]},
            index=pd.DatetimeIndex([_SINCE, _UNTIL]),
        )

        analysis_input = _analysis_input(
            ohlcv_provider=provider,
            report_config=ReportConfig(reference_symbol="BTC/USDT:USDT"),
        )

        assert list(analysis_input.reference_price) == [100.0, 110.0]
        assert analysis_input.reference_symbol_str == "BTC/USDT:USDT"

    def test_a_window_declaring_no_reference_prices_none(self):
        provider = Mock(spec=OHLCVProviderProtocol)

        analysis_input = _analysis_input(
            ohlcv_provider=provider,
        )

        assert analysis_input.reference_price is None
        provider.get_all_cached_ohlcv_for_symbol.assert_not_called()


class TestSettlementCurrency:
    def test_symbols_settling_in_one_currency_are_accepted(self):
        _analysis_input(
            symbols=[_BTC, Symbol.create("ETH/USDT:USDT")],
        )

    def test_symbols_settling_in_several_currencies_are_rejected(self):
        with pytest.raises(StrategyCriticalError, match="several currencies"):
            _analysis_input(
                symbols=[_BTC, _ETH_USDC],
            )


class TestUnpairableWarning:
    def test_an_unpairable_close_is_logged(self, caplog):
        with caplog.at_level(logging.WARNING):
            _analysis_input(
                executions=[_execution(realised_profit=5.0)],
            )

        assert "1 execution" in caplog.text

    def test_a_fully_paired_window_logs_nothing(self, caplog):
        opened = _execution(side=OrderSide.BUY, realised_profit=None, fee=None)
        closed = _execution(realised_profit=5.0, fee=1.0)

        with caplog.at_level(logging.WARNING):
            _analysis_input(
                executions=[opened, closed],
            )

        assert caplog.text == ""


class TestTheBalanceTheWindowOpenedOn:
    def test_the_windows_profit_comes_back_off_what_the_account_holds_now(self):
        booked = _execution(realised_profit=50.0, fee=1.0)

        analysis_input = _analysis_input(
            executions=[booked],
        )

        assert analysis_input.initial_balance == pytest.approx(951.0)

    def test_the_curve_opens_on_that_balance(self):
        booked = _execution(realised_profit=50.0, fee=1.0)

        analysis_input = _analysis_input(
            executions=[booked],
        )

        assert analysis_input.equity_curve[_SINCE] == pytest.approx(951.0)

    def test_the_curve_ends_on_what_the_account_holds_now(self):
        booked = _execution(realised_profit=50.0, fee=1.0)

        analysis_input = _analysis_input(
            executions=[booked],
        )

        assert analysis_input.equity_curve[_UNTIL] == pytest.approx(1000.0)

    def test_an_execution_before_the_window_is_left_out_of_the_recovery(self):
        older = _execution(
            timestamp=datetime(2026, 7, 20, tzinfo=UTC), realised_profit=300.0, fee=0.0
        )

        analysis_input = _analysis_input(
            executions=[older],
        )

        assert analysis_input.initial_balance == pytest.approx(1000.0)

    def test_the_settlement_currency_is_the_one_read(self):
        analysis_input = _analysis_input(
            balances={"USDC": Balance(locked=0.0, total=700.0), **_HELD},
        )

        assert analysis_input.initial_balance == pytest.approx(1000.0)


class TestAWindowWithNoBalanceToMeasureAgainst:
    @pytest.fixture
    def unmeasured(self) -> AnalysisInputs:
        return _analysis_input(
            executions=[_execution(realised_profit=50.0, fee=1.0)],
            balances={},
        )

    def test_no_balance_is_stated(self, unmeasured):
        assert unmeasured.initial_balance is None

    def test_the_curve_carries_what_trading_booked(self, unmeasured):
        assert unmeasured.equity_curve[_UNTIL] == pytest.approx(49.0)

    def test_the_missing_balance_is_logged(self, caplog):
        with caplog.at_level(logging.WARNING):
            _analysis_input(
                balances={},
            )

        assert "No USDT balance" in caplog.text
