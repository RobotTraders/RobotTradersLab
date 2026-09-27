import pytest

from robottraderslab import Symbol
from robottraderslab._core import FeeMode, PlacementReserve
from robottraderslab.backtester.market_types import create_futures_market
from robottraderslab.bootstrap import SimulatedMarket
from robottraderslab.exchanges import Balance
from robottraderslab.strategies import AccountSnapshot
from robottraderslab.strategies.futures import TotalBalanceRatio

BTC_USDT = Symbol.create("BTC/USDT:USDT")
BALANCE = 10_000.0
TAKER_FEE_RATE = 0.0006
PRICE = 100.0
_NO_RESERVE = PlacementReserve(margin_markup=0.0, fee_markup=0.0, notional_reserve=0.0)


def _market(fee_mode: FeeMode) -> SimulatedMarket:
    return create_futures_market(
        initial_balance={"USDT": BALANCE},
        maker_fee_rate=0.0002,
        taker_fee_rate=TAKER_FEE_RATE,
        fee_mode=fee_mode,
        placement_reserve=_NO_RESERVE,
    )


def _flat_snapshot() -> AccountSnapshot:
    return AccountSnapshot(
        account_name="simulated",
        balances={"USDT": Balance(locked=0.0, total=BALANCE)},
        conversion_rates={BTC_USDT: 1.0},
    )


class TestTheFeeConventionReachesTheAccount:
    """One declaration settles what the simulator charges and what its account
    holds back when it sizes, so an entry sized from a whole balance is one the
    simulator accepts.
    """

    def test_an_exchange_backtest_sizes_with_the_fee_set_aside(self):
        market = _market("exchange")

        order = (
            market.account.long_entry(BTC_USDT)
            .size(TotalBalanceRatio(1.0), PRICE, _flat_snapshot())
            .build()
        )

        assert order.quantity == pytest.approx(BALANCE / (PRICE * (1 + TAKER_FEE_RATE)))

    def test_a_cost_backtest_sizes_on_the_whole_balance(self):
        market = _market("cost")

        order = (
            market.account.long_entry(BTC_USDT)
            .size(TotalBalanceRatio(1.0), PRICE, _flat_snapshot())
            .build()
        )

        assert order.quantity == pytest.approx(BALANCE / PRICE)
