from datetime import UTC, datetime

from robottraderslab._core import PositionSide, PositionSnapshot, Symbol
from robottraderslab.analyser.execution_positions import build_open_positions

_BTC = Symbol.create("BTC/USDT:USDT")


def _position(**overrides: object) -> PositionSnapshot:
    arguments = {
        "symbol": _BTC,
        "side": PositionSide.LONG,
        "quantity": 2.0,
        "average_entry_price": 100.0,
        "entry_time": datetime(2026, 8, 1, tzinfo=UTC),
        "leverage": 5.0,
        "liquidation_price": 50.0,
    }
    arguments.update(overrides)
    return PositionSnapshot(**arguments)  # type: ignore[arg-type]


class TestNoPositions:
    def test_no_positions_yields_an_empty_frame(self):
        positions = build_open_positions({})

        assert positions.empty


class TestOnePosition:
    def test_the_symbol_and_side_carry_over(self):
        positions = build_open_positions({_BTC: _position()})

        row = positions.iloc[0]
        assert row["symbol"] == "BTC/USDT:USDT"
        assert row["side"] == "long"

    def test_a_short_position_carries_its_side(self):
        positions = build_open_positions({_BTC: _position(side=PositionSide.SHORT)})

        assert positions.iloc[0]["side"] == "short"

    def test_the_entry_price_and_quantity_carry_over(self):
        positions = build_open_positions(
            {_BTC: _position(average_entry_price=123.0, quantity=4.0)}
        )

        row = positions.iloc[0]
        assert row["entry_price"] == 123.0
        assert row["net_quantity"] == 4.0
        assert row["gross_quantity"] == 4.0

    def test_the_entry_time_carries_over(self):
        entry_time = datetime(2026, 7, 15, tzinfo=UTC)

        positions = build_open_positions({_BTC: _position(entry_time=entry_time)})

        assert positions.iloc[0]["entry_time"] == entry_time

    def test_there_is_no_exit_yet(self):
        positions = build_open_positions({_BTC: _position()})

        row = positions.iloc[0]
        assert row["exit_time"] is None
        assert row["exit_price"] == 0.0
        assert row["exit_reason"] is None

    def test_the_pnl_fields_are_not_yet_known(self):
        positions = build_open_positions({_BTC: _position()})

        row = positions.iloc[0]
        assert row["gross_pnl"] == 0.0
        assert row["net_pnl"] == 0.0
        assert row["net_pnl_pct"] == 0.0

    def test_the_entry_fee_is_not_stated_by_the_venue(self):
        positions = build_open_positions({_BTC: _position()})

        assert positions.iloc[0]["entry_fee"] == 0.0


class TestSeveralPositions:
    def test_each_symbol_becomes_its_own_row(self):
        eth = Symbol.create("ETH/USDT:USDT")

        positions = build_open_positions(
            {_BTC: _position(), eth: _position(symbol=eth)}
        )

        assert set(positions["symbol"]) == {"BTC/USDT:USDT", "ETH/USDT:USDT"}
