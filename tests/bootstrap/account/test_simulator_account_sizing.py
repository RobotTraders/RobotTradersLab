import pytest

from robottraderslab import Symbol
from robottraderslab._core import PlacementReserve
from robottraderslab.bootstrap import load_single_account
from robottraderslab.exchanges import Balance
from robottraderslab.strategies import AccountSnapshot
from robottraderslab.strategies.futures import TotalBalanceRatio

BTC_USDT = Symbol.create("BTC/USDT:USDT")
BALANCE = 10_000.0
TAKER_FEE_RATE = 0.0007
PRICE = 100.0
RESERVE = PlacementReserve()
RESERVE_RATE = (
    RESERVE.margin_markup
    + TAKER_FEE_RATE * (1 + RESERVE.fee_markup)
    + RESERVE.notional_reserve
)
SIMULATOR_CONFIG = {
    "exchange": "simulator",
    "initial_balance": {"USDT": BALANCE},
    "maker_fee_rate": 0.0002,
    "taker_fee_rate": TAKER_FEE_RATE,
}


async def test_a_live_simulator_account_sizes_with_the_reserve_set_aside():
    account = await load_single_account(SIMULATOR_CONFIG, {})
    snapshot = AccountSnapshot(
        account_name=account.name,
        balances={"USDT": Balance(locked=0.0, total=BALANCE)},
        conversion_rates={BTC_USDT: 1.0},
    )

    order = (
        account.long_entry(BTC_USDT)
        .size(TotalBalanceRatio(1.0), PRICE, snapshot)
        .build()
    )

    assert order.quantity == pytest.approx(BALANCE / (PRICE * (1 + RESERVE_RATE)))
