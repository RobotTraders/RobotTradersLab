import pytest

from robottraderslab import Symbol
from robottraderslab._core import PlacementReserve
from robottraderslab.backtester.simulator import FeeModel, FeeRates, fee_model_for
from robottraderslab.exchanges import MarginMode, MarginSettings
from robottraderslab.futures import FuturesAccount
from robottraderslab.strategies import AccountSnapshot
from robottraderslab.strategies.futures import Margin

MARGIN = 40.0
PRICE = 100.0
TAKER_FEE_RATE = 0.0006


@pytest.fixture
def fee_rates() -> FeeRates:
    return FeeRates(maker=0.0002, taker=TAKER_FEE_RATE)


@pytest.fixture
def fee_model() -> FeeModel:
    return fee_model_for("exchange", PlacementReserve())


def _worth_sized_by_margin(sim, symbol: Symbol, leverage: float) -> float:
    sim.simulation_engine.set_symbol_leverage(symbol, leverage)
    snapshot = AccountSnapshot(
        account_name="simulator",
        balances=sim.simulation_engine.get_balances(),
        conversion_rates={symbol: 1.0},
        margin_settings={
            symbol: MarginSettings(leverage=leverage, margin_mode=MarginMode.ISOLATED)
        },
    )
    order = (
        FuturesAccount(sim.exchange)
        .long_entry(symbol)
        .size(Margin(MARGIN), PRICE, snapshot)
        .build()
    )
    return order.quantity * PRICE


class TestMarginAtLeverage:
    @pytest.mark.parametrize("leverage", [1.0, 5.0, 20.0])
    def test_the_venue_locks_the_margin_named(
        self, sim, btc_usdt_perp, fee_model, leverage
    ):
        worth = _worth_sized_by_margin(sim, btc_usdt_perp, leverage)

        assert fee_model.placement_requirement(
            worth, leverage, TAKER_FEE_RATE
        ) == pytest.approx(MARGIN)

    @pytest.mark.parametrize("leverage", [5.0, 20.0])
    def test_the_worth_grows_with_the_leverage(self, sim, btc_usdt_perp, leverage):
        at_one = _worth_sized_by_margin(sim, btc_usdt_perp, 1.0)

        worth = _worth_sized_by_margin(sim, btc_usdt_perp, leverage)

        assert at_one < worth < at_one * leverage
