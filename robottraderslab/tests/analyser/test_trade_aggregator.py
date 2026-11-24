import logging
from typing import Any, get_args

import numpy as np
import pandas as pd
import pytest

from robottraderslab.analyser import create_trade_aggregation
from robottraderslab.analyser.accounting_methods import AccountingMethodChoice
from robottraderslab.analyser.analysis_inputs import TRADE_COLUMNS
from robottraderslab.analyser.trade_aggregator import TradeAggregator
from robottraderslab.backtester.simulator.fill_recorder import FillType

BTC = "BTC/USDT:USDT"
ETH = "ETH/USDT:USDT"
DAY = pd.Timedelta(days=1)
START = pd.Timestamp("2024-01-01 10:00")


def _fill(
    day: int,
    fill_type: str,
    quantity: float,
    price: float,
    *,
    fee: float = 1.0,
    rate: float = 1.0,
    symbol: str = BTC,
    side: str = "long",
    reason: str | None = None,
    tag: str | None = None,
) -> dict[str, Any]:
    return {
        "timestamp": START + day * DAY,
        "symbol": symbol,
        "side": side,
        "gross_quantity": quantity,
        "net_quantity": quantity,
        "price": price,
        "fee": fee,
        "fill_type": fill_type,
        "reason": reason,
        "tag": tag,
        "rate": rate,
    }


def _fills(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows).set_index("timestamp")


def _aggregate(
    rows: list[dict[str, Any]],
    accounting_method: AccountingMethodChoice = AccountingMethodChoice.FIFO,
) -> TradeAggregator:
    return create_trade_aggregation(_fills(rows), accounting_method)


class TestFullExit:
    def test_a_long_round_trip_closes_one_trade(self):
        aggregator = _aggregate(
            [
                _fill(0, "enter_long", 2.0, 100.0, fee=0.2, reason="in", tag="a"),
                _fill(1, "exit_long", 2.0, 110.0, fee=0.22, reason="out"),
            ]
        )

        closed = aggregator.trades.iloc[0]
        assert len(aggregator.trades) == 1
        assert aggregator.open_trades.empty
        assert closed["entry_time"] == START
        assert closed["exit_time"] == START + DAY
        assert closed["net_quantity"] == 2.0
        assert closed["gross_pnl"] == pytest.approx(20.0)
        assert closed["net_pnl"] == pytest.approx(20.0 - 0.2 - 0.22)
        assert closed["entry_reason"] == "in"
        assert closed["exit_reason"] == "out"
        assert closed["tag"] == "a"

    def test_a_short_round_trip_earns_on_a_falling_price(self):
        aggregator = _aggregate(
            [
                _fill(0, "enter_short", 2.0, 100.0, side="short"),
                _fill(1, "exit_short", 2.0, 90.0, side="short"),
            ]
        )

        assert aggregator.trades.iloc[0]["gross_pnl"] == pytest.approx(20.0)

    def test_net_pnl_and_its_share_use_the_rates_of_both_legs(self):
        aggregator = _aggregate(
            [
                _fill(0, "enter_long", 1.0, 100.0, fee=1.0, rate=2.0),
                _fill(1, "exit_long", 1.0, 110.0, fee=2.0, rate=3.0),
            ]
        )

        closed = aggregator.trades.iloc[0]
        assert closed["gross_pnl"] == pytest.approx(10.0 * 3.0)
        assert closed["net_pnl"] == pytest.approx(30.0 - 1.0 - 2.0)
        assert closed["net_pnl_pct"] == pytest.approx(27.0 / (100.0 * 2.0))

    def test_a_liquidation_closes_the_whole_position_with_its_reason(self):
        aggregator = _aggregate(
            [
                _fill(0, "enter_short", 2.0, 100.0, side="short"),
                _fill(
                    1,
                    "liquidate_short",
                    2.0,
                    150.0,
                    fee=0.0,
                    side="short",
                    reason="liquidation",
                ),
            ]
        )

        closed = aggregator.trades.iloc[0]
        assert aggregator.open_trades.empty
        assert closed["exit_reason"] == "liquidation"
        assert closed["gross_pnl"] == pytest.approx(-100.0)

    def test_an_entry_without_an_exit_stays_open(self):
        aggregator = _aggregate([_fill(0, "enter_long", 2.0, 100.0)])

        assert aggregator.trades.empty
        assert aggregator.open_trades.iloc[0]["net_quantity"] == 2.0


class TestPartialExit:
    @pytest.fixture
    def entry_reduced_by_less_than_its_size(self) -> TradeAggregator:
        return _aggregate(
            [
                _fill(0, "enter_long", 1.0, 100.0, fee=0.10, reason="in", tag="a"),
                _fill(1, "reduce_long", 0.4, 110.0, fee=0.05, reason="out"),
            ]
        )

    def test_only_the_matched_part_closes(self, entry_reduced_by_less_than_its_size):
        closed = entry_reduced_by_less_than_its_size.trades.iloc[0]

        assert len(entry_reduced_by_less_than_its_size.trades) == 1
        assert closed["net_quantity"] == pytest.approx(0.4)
        assert closed["gross_quantity"] == pytest.approx(0.4)
        assert closed["gross_pnl"] == pytest.approx(4.0)

    def test_the_closed_part_carries_its_share_of_the_entry_fee(
        self, entry_reduced_by_less_than_its_size
    ):
        closed = entry_reduced_by_less_than_its_size.trades.iloc[0]

        assert closed["entry_fee"] == pytest.approx(0.04)
        assert closed["exit_fee"] == pytest.approx(0.05)

    def test_the_rest_of_the_entry_stays_open_with_the_rest_of_its_fee(
        self, entry_reduced_by_less_than_its_size
    ):
        still_open = entry_reduced_by_less_than_its_size.open_trades.iloc[0]

        assert len(entry_reduced_by_less_than_its_size.open_trades) == 1
        assert still_open["net_quantity"] == pytest.approx(0.6)
        assert still_open["gross_quantity"] == pytest.approx(0.6)
        assert still_open["entry_fee"] == pytest.approx(0.06)
        assert still_open["exit_time"] is None

    def test_both_parts_keep_the_entry_details(
        self, entry_reduced_by_less_than_its_size
    ):
        closed = entry_reduced_by_less_than_its_size.trades.iloc[0]
        still_open = entry_reduced_by_less_than_its_size.open_trades.iloc[0]

        for part in (closed, still_open):
            assert part["entry_time"] == START
            assert part["entry_price"] == 100.0
            assert part["entry_reason"] == "in"
            assert part["tag"] == "a"

    def test_the_gross_quantity_is_split_in_the_same_share_as_the_net(self):
        aggregator = _aggregate(
            [
                _fill(0, "spot_buy", 0.999, 100.0, symbol="BTC/USDT")
                | {"gross_quantity": 1.0},
                _fill(1, "spot_sell", 0.333, 110.0, symbol="BTC/USDT"),
            ]
        )

        closed = aggregator.trades.iloc[0]
        still_open = aggregator.open_trades.iloc[0]
        assert closed["gross_quantity"] == pytest.approx(1.0 / 3)
        assert still_open["gross_quantity"] == pytest.approx(2.0 / 3)
        assert still_open["net_quantity"] == pytest.approx(0.666)

    def test_a_partial_exit_converts_its_pnl_at_the_exit_rate(self):
        aggregator = _aggregate(
            [
                _fill(0, "enter_long", 1.0, 100.0, fee=0.0, rate=1.0),
                _fill(1, "reduce_long", 0.5, 110.0, fee=0.0, rate=2.0),
            ]
        )

        assert aggregator.trades.iloc[0]["gross_pnl"] == pytest.approx(5.0 * 2.0)

    def test_a_later_exit_closes_the_rest(self):
        aggregator = _aggregate(
            [
                _fill(0, "enter_long", 1.0, 100.0, fee=0.10),
                _fill(1, "reduce_long", 0.4, 110.0),
                _fill(2, "exit_long", 0.6, 120.0),
            ]
        )

        assert aggregator.open_trades.empty
        assert sorted(aggregator.trades["net_quantity"]) == pytest.approx([0.4, 0.6])
        assert sorted(aggregator.trades["entry_fee"]) == pytest.approx([0.04, 0.06])
        assert aggregator.trades["gross_pnl"].sum() == pytest.approx(4.0 + 12.0)

    def test_an_exit_spanning_two_entries_splits_only_the_last_one(self):
        aggregator = _aggregate(
            [
                _fill(0, "enter_long", 0.5, 100.0),
                _fill(1, "add_to_long", 0.6, 102.0),
                _fill(2, "reduce_long", 0.8, 110.0),
            ]
        )

        assert sorted(aggregator.trades["net_quantity"]) == pytest.approx([0.3, 0.5])
        assert aggregator.open_trades.iloc[0]["net_quantity"] == pytest.approx(0.3)
        assert aggregator.open_trades.iloc[0]["entry_price"] == 102.0

    def test_the_exit_fee_is_shared_across_the_entries_it_closes(self):
        aggregator = _aggregate(
            [
                _fill(0, "enter_long", 0.5, 100.0),
                _fill(1, "add_to_long", 0.5, 102.0),
                _fill(2, "exit_long", 1.0, 110.0, fee=0.20),
            ]
        )

        assert list(aggregator.trades["exit_fee"]) == pytest.approx([0.10, 0.10])

    def test_two_profiles_on_one_symbol_exiting_in_the_other_order(self):
        aggregator = _aggregate(
            [
                _fill(0, "enter_long", 0.5, 100.0, tag="A"),
                _fill(1, "add_to_long", 0.6, 102.0, tag="B"),
                _fill(2, "reduce_long", 0.6, 110.0, tag="B"),
                _fill(3, "exit_long", 0.5, 111.0, tag="A"),
            ]
        )

        assert aggregator.open_trades.empty
        assert aggregator.trades["net_quantity"].sum() == pytest.approx(1.1)
        assert aggregator.trades["gross_pnl"].sum() == pytest.approx(
            (110.0 - 100.0) * 0.5 + (110.0 - 102.0) * 0.1 + (111.0 - 102.0) * 0.5
        )


class TestAccountingOrder:
    @pytest.fixture
    def two_entries_and_one_exit(self) -> list[dict[str, Any]]:
        return [
            _fill(0, "enter_long", 1.0, 100.0),
            _fill(1, "add_to_long", 1.0, 200.0),
            _fill(2, "reduce_long", 1.0, 300.0),
        ]

    def test_fifo_closes_the_oldest_entry(self, two_entries_and_one_exit):
        aggregator = _aggregate(two_entries_and_one_exit, AccountingMethodChoice.FIFO)

        assert aggregator.trades.iloc[0]["entry_price"] == 100.0
        assert aggregator.open_trades.iloc[0]["entry_price"] == 200.0

    def test_lifo_closes_the_newest_entry(self, two_entries_and_one_exit):
        aggregator = _aggregate(two_entries_and_one_exit, AccountingMethodChoice.LIFO)

        assert aggregator.trades.iloc[0]["entry_price"] == 200.0
        assert aggregator.open_trades.iloc[0]["entry_price"] == 100.0

    def test_lifo_keeps_closing_the_newest_entry_exit_after_exit(self):
        aggregator = _aggregate(
            [
                _fill(0, "enter_long", 1.0, 100.0),
                _fill(1, "add_to_long", 1.0, 200.0),
                _fill(2, "add_to_long", 1.0, 300.0),
                _fill(3, "reduce_long", 1.0, 400.0),
                _fill(4, "reduce_long", 1.0, 400.0),
            ],
            AccountingMethodChoice.LIFO,
        )

        assert sorted(aggregator.trades["entry_price"]) == [200.0, 300.0]
        assert aggregator.open_trades.iloc[0]["entry_price"] == 100.0

    def test_lifo_closes_one_of_two_identical_entries_and_leaves_the_other_open(self):
        aggregator = _aggregate(
            [
                _fill(0, "enter_long", 1.0, 100.0),
                _fill(0, "add_to_long", 1.0, 100.0),
                _fill(1, "reduce_long", 1.0, 110.0),
            ],
            AccountingMethodChoice.LIFO,
        )

        still_open = aggregator.open_trades.iloc[0]
        assert len(aggregator.trades) == 1
        assert still_open["exit_time"] is None
        assert still_open["net_pnl"] == 0.0


class TestUnmatchedExit:
    def test_an_exit_with_nothing_open_is_left_out_and_reported(self, caplog):
        with caplog.at_level(logging.ERROR):
            aggregator = _aggregate([_fill(0, "exit_long", 1.0, 100.0)])

        assert aggregator.trades.empty
        assert "found no open entry" in caplog.text
        assert BTC in caplog.text

    def test_an_exit_larger_than_the_open_entries_closes_them_and_reports_the_rest(
        self, caplog
    ):
        with caplog.at_level(logging.ERROR):
            aggregator = _aggregate(
                [
                    _fill(0, "enter_long", 1.0, 100.0),
                    _fill(1, "exit_long", 1.5, 110.0),
                ]
            )

        assert aggregator.trades.iloc[0]["net_quantity"] == 1.0
        assert aggregator.open_trades.empty
        assert "0.5" in caplog.text

    def test_a_fully_matched_exit_reports_nothing(self, caplog):
        with caplog.at_level(logging.ERROR):
            _aggregate(
                [
                    _fill(0, "enter_long", 1.0, 100.0),
                    _fill(1, "exit_long", 1.0, 110.0),
                ]
            )

        assert caplog.text == ""

    def test_an_exit_two_entries_cover_between_them_leaves_nothing_behind(self, caplog):
        with caplog.at_level(logging.ERROR):
            aggregator = _aggregate(
                [
                    _fill(0, "enter_long", 0.02, 100.0),
                    _fill(1, "add_to_long", 0.04, 102.0),
                    _fill(2, "exit_long", 0.06, 110.0),
                ]
            )

        assert aggregator.open_trades.empty
        assert caplog.text == ""


class TestPositionsAreKeptApart:
    def test_a_long_and_a_short_on_one_symbol_never_match_each_other(self):
        aggregator = _aggregate(
            [
                _fill(0, "enter_long", 1.0, 100.0),
                _fill(1, "enter_short", 1.0, 100.0, side="short"),
                _fill(2, "exit_short", 1.0, 90.0, side="short"),
            ]
        )

        assert aggregator.trades.iloc[0]["side"] == "short"
        assert aggregator.open_trades.iloc[0]["side"] == "long"

    def test_symbols_never_match_each_other(self):
        aggregator = _aggregate(
            [
                _fill(0, "enter_long", 1.0, 100.0, symbol=BTC),
                _fill(1, "enter_long", 1.0, 10.0, symbol=ETH),
                _fill(2, "exit_long", 1.0, 11.0, symbol=ETH),
            ]
        )

        assert aggregator.trades.iloc[0]["symbol"] == ETH
        assert aggregator.open_trades.iloc[0]["symbol"] == BTC


class TestFillOrder:
    def test_fills_are_matched_in_time_order_whatever_their_row_order(self):
        aggregator = _aggregate(
            [
                _fill(2, "exit_long", 1.0, 120.0),
                _fill(0, "enter_long", 1.0, 100.0),
                _fill(1, "add_to_long", 1.0, 110.0),
            ]
        )

        assert aggregator.trades.iloc[0]["entry_price"] == 100.0

    def test_the_trade_table_lists_the_newest_entry_first(self):
        aggregator = _aggregate(
            [
                _fill(0, "enter_long", 1.0, 100.0),
                _fill(1, "exit_long", 1.0, 110.0),
                _fill(2, "enter_long", 1.0, 100.0),
                _fill(3, "exit_long", 1.0, 110.0),
            ]
        )

        assert list(aggregator.trades["entry_time"]) == [START + 2 * DAY, START]

    def test_a_position_opened_added_to_and_reduced_inside_one_candle(self):
        aggregator = _aggregate(
            [
                _fill(0, "enter_long", 1.0, 100.0),
                _fill(0, "add_to_long", 1.0, 101.0),
                _fill(0, "reduce_long", 1.0, 102.0),
            ]
        )

        assert aggregator.trades.iloc[0]["entry_price"] == 100.0
        assert aggregator.open_trades.iloc[0]["entry_price"] == 101.0

    def test_fills_sharing_a_timestamp_keep_their_recorded_order(self):
        symbols = [f"{name}/USDT:USDT" for name in ("UNI", "ADA", "DOGE", "DOT")]
        rows = []
        for symbol in symbols:
            rows.append(_fill(0, "enter_long", 10.0, 5.0, symbol=symbol))
            rows.append(_fill(0, "exit_long", 10.0, 5.2, symbol=symbol))

        aggregator = _aggregate(rows)

        assert len(aggregator.trades) == len(symbols)
        assert aggregator.open_trades.empty


class TestSpot:
    def test_a_buy_and_a_sell_make_one_long_trade(self):
        aggregator = _aggregate(
            [
                _fill(
                    0,
                    "spot_buy",
                    1.0,
                    40000.0,
                    symbol="BTC/USDT",
                    side="buy",
                    reason="buy",
                ),
                _fill(
                    1,
                    "spot_sell",
                    1.0,
                    42000.0,
                    symbol="BTC/USDT",
                    side="sell",
                    reason="sell",
                ),
            ]
        )

        closed = aggregator.trades.iloc[0]
        assert closed["side"] == "long"
        assert closed["gross_pnl"] == pytest.approx(2000.0)
        assert closed["entry_reason"] == "buy"
        assert closed["exit_reason"] == "sell"

    def test_selling_part_of_a_holding_leaves_the_rest_open(self):
        aggregator = _aggregate(
            [
                _fill(0, "spot_buy", 1.0, 40000.0, symbol="BTC/USDT"),
                _fill(1, "spot_sell", 0.25, 42000.0, symbol="BTC/USDT"),
            ]
        )

        assert aggregator.trades.iloc[0]["net_quantity"] == pytest.approx(0.25)
        assert aggregator.open_trades.iloc[0]["net_quantity"] == pytest.approx(0.75)


class TestQuantityIsConserved:
    """Random fill sequences that never exit more than is open."""

    @pytest.fixture(params=[AccountingMethodChoice.FIFO, AccountingMethodChoice.LIFO])
    def accounting_method(
        self, request: pytest.FixtureRequest
    ) -> AccountingMethodChoice:
        return request.param

    @pytest.fixture(params=range(20))
    def random_fills(self, request: pytest.FixtureRequest) -> list[dict[str, Any]]:
        rng = np.random.default_rng(request.param)
        rows = []
        open_quantity = {
            (symbol, side): 0.0 for symbol in (BTC, ETH) for side in ("long", "short")
        }
        for day in range(rng.integers(4, 20)):
            symbol = str(rng.choice([BTC, ETH]))
            side = str(rng.choice(["long", "short"]))
            key = (symbol, side)
            price = float(rng.uniform(90.0, 110.0))
            if open_quantity[key] == 0.0 or rng.random() < 0.5:
                quantity = float(rng.integers(1, 10))
                verb = "enter" if open_quantity[key] == 0.0 else "add_to"
                open_quantity[key] += quantity
            else:
                quantity = float(rng.integers(1, int(open_quantity[key]) + 1))
                open_quantity[key] -= quantity
                verb = "exit" if open_quantity[key] == 0.0 else "reduce"
            rows.append(
                _fill(day, f"{verb}_{side}", quantity, price, symbol=symbol, side=side)
            )
        return rows

    def test_every_exited_quantity_lands_in_a_closed_trade(
        self, random_fills, accounting_method
    ):
        fills = _fills(random_fills)
        exited = fills.loc[
            fills["fill_type"].str.startswith(("exit", "reduce")), "net_quantity"
        ].sum()
        entered = fills["net_quantity"].sum() - exited

        aggregator = create_trade_aggregation(fills, accounting_method)

        closed_quantity = aggregator.trades["net_quantity"].sum()
        open_quantity = (
            aggregator.open_trades["net_quantity"].sum()
            if not aggregator.open_trades.empty
            else 0.0
        )
        assert closed_quantity == pytest.approx(exited)
        assert open_quantity == pytest.approx(entered - exited)

    def test_gross_pnl_of_the_closed_trades_matches_the_fills(
        self, random_fills, accounting_method
    ):
        aggregator = create_trade_aggregation(_fills(random_fills), accounting_method)
        closing_everything = random_fills + [
            _fill(100, f"exit_{side}", quantity, 100.0, symbol=symbol, side=side)
            for (symbol, side), quantity in _still_open(aggregator).items()
            if quantity > 0.0
        ]

        fully_closed = create_trade_aggregation(
            _fills(closing_everything), accounting_method
        )

        assert fully_closed.open_trades.empty
        assert fully_closed.trades["gross_pnl"].sum() == pytest.approx(
            _gross_pnl_of(_fills(closing_everything))
        )


def _still_open(aggregator: TradeAggregator) -> dict[tuple[str, str], float]:
    if aggregator.open_trades.empty:
        return {}
    by_position = aggregator.open_trades.groupby(["symbol", "side"])["net_quantity"]
    return by_position.sum().to_dict()


def _gross_pnl_of(fills: pd.DataFrame) -> float:
    sign = np.where(fills["side"] == "long", 1.0, -1.0)
    is_exit = fills["fill_type"].str.startswith(("exit", "reduce"))
    direction = np.where(is_exit, 1.0, -1.0)
    return float((sign * direction * fills["price"] * fills["net_quantity"]).sum())


class TestEveryFillTypeIsRouted:
    @pytest.mark.parametrize("fill_type", get_args(FillType.__value__))
    def test_a_fill_of_this_type_opens_or_closes_a_position(self, fill_type):
        side = "short" if fill_type.endswith("short") else "long"
        opening = fill_type.startswith(("enter", "add", "spot_buy"))
        entry_type = "spot_buy" if fill_type == "spot_sell" else f"enter_{side}"
        rows = [] if opening else [_fill(0, entry_type, 1.0, 100.0, side=side)]
        rows.append(_fill(1, fill_type, 1.0, 110.0, side=side))

        aggregator = _aggregate(rows)

        assert len(aggregator.open_trades) == (1 if opening else 0)
        assert len(aggregator.trades) == (0 if opening else 1)


class TestNothingOpen:
    def test_the_open_table_keeps_the_trade_columns(self):
        aggregator = _aggregate(
            [
                _fill(0, "enter_long", 1.0, 100.0),
                _fill(1, "exit_long", 1.0, 110.0),
            ]
        )

        assert list(aggregator.open_trades.columns) == list(TRADE_COLUMNS)


class TestNoFills:
    def test_both_tables_come_back_empty(self):
        no_fills = _fills([_fill(0, "enter_long", 1.0, 1.0)]).iloc[0:0]

        aggregator = create_trade_aggregation(no_fills)

        assert aggregator.trades.empty
        assert aggregator.open_trades.empty


class TestFillsValidation:
    def test_missing_columns_are_named(self):
        with pytest.raises(ValueError, match="net_quantity"):
            TradeAggregator(pd.DataFrame({"symbol": [BTC]}))
