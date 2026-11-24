import pytest
from simulation_fixture import SimulationFixture

from robottraderslab import Symbol
from robottraderslab.exchanges import (
    OrderProtocol,
    OrderSide,
    OrderType,
    TimeInForce,
    TriggerType,
    tag_of,
)
from robottraderslab.futures import FuturesAccount
from robottraderslab_core.engine import carries_this_bot

PRICE = 100.0
TAG = "1h-fast"


@pytest.fixture
async def long_position(
    sim: SimulationFixture, btc_usdt_perp: Symbol, declared_account: FuturesAccount
) -> None:
    sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=PRICE)
    await sim.exchange.place_market_order(
        symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=1.0
    )
    sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=PRICE)


async def _the_resting_order(
    sim: SimulationFixture, symbol: Symbol, kind: OrderType
) -> OrderProtocol:
    open_orders = await sim.exchange.get_open_orders([symbol])
    (resting,) = (order for order in open_orders if order.kind == kind)
    return resting


@pytest.fixture
async def resting_post_only_exit(
    sim: SimulationFixture,
    btc_usdt_perp: Symbol,
    declared_account: FuturesAccount,
    long_position: None,
) -> OrderProtocol:
    exit_order = (
        declared_account.long_exit(btc_usdt_perp, 1.0)
        .limit(120.0, time_in_force=TimeInForce.POST_ONLY)
        .build()
    )
    await declared_account.place_orders([exit_order]).execute()
    return await _the_resting_order(sim, btc_usdt_perp, OrderType.LIMIT)


class TestUpdateOrder:
    async def test_a_post_only_reduce_only_exit_moved_by_price_alone_keeps_both(
        self, sim, btc_usdt_perp, declared_account, resting_post_only_exit
    ):
        await declared_account.update_order(
            resting_post_only_exit, limit_price=130.0
        ).execute()

        moved = await _the_resting_order(sim, btc_usdt_perp, OrderType.LIMIT)
        assert moved.limit_price == 130.0
        assert moved.time_in_force == TimeInForce.POST_ONLY
        assert moved.reduce_only is True

    async def test_a_quantity_alone_keeps_the_price_and_the_reduce_only(
        self, sim, btc_usdt_perp, declared_account, resting_post_only_exit
    ):
        await declared_account.update_order(
            resting_post_only_exit, quantity=0.5
        ).execute()

        moved = await _the_resting_order(sim, btc_usdt_perp, OrderType.LIMIT)
        assert moved.quantity == 0.5
        assert moved.limit_price == 120.0
        assert moved.reduce_only is True

    async def test_a_stop_loss_moved_alone_keeps_the_trigger_types_of_a_trigger_entry(
        self, sim, btc_usdt_perp, declared_account
    ):
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=PRICE)
        entry = (
            declared_account.long_entry(btc_usdt_perp, 1.0)
            .limit(111.0)
            .trigger(110.0, trigger_type=TriggerType.MARK_PRICE)
            .stop_loss(90.0, trigger_type=TriggerType.MARK_PRICE)
            .build()
        )
        await declared_account.place_orders([entry]).execute()
        resting = await _the_resting_order(sim, btc_usdt_perp, OrderType.TRIGGER)

        await declared_account.update_order(resting, stop_loss=88.0).execute()

        moved = await _the_resting_order(sim, btc_usdt_perp, OrderType.TRIGGER)
        assert moved.stop_loss_price == 88.0
        assert moved.stop_loss_trigger_type == TriggerType.MARK_PRICE
        assert moved.trigger_price == 110.0
        assert moved.trigger_type == TriggerType.MARK_PRICE
        assert moved.limit_price == 111.0

    async def test_a_take_profit_named_on_a_limit_entry_adds_it_beside_its_stop_loss(
        self, sim, btc_usdt_perp, declared_account
    ):
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=PRICE)
        entry = (
            declared_account.long_entry(btc_usdt_perp, 1.0)
            .limit(90.0)
            .stop_loss(80.0, trigger_type=TriggerType.LAST_PRICE)
            .build()
        )
        await declared_account.place_orders([entry]).execute()
        resting = await _the_resting_order(sim, btc_usdt_perp, OrderType.LIMIT)

        await declared_account.update_order(resting, take_profit=120.0).execute()

        moved = await _the_resting_order(sim, btc_usdt_perp, OrderType.LIMIT)
        assert moved.take_profit_price == 120.0
        assert moved.stop_loss_price == 80.0
        assert moved.stop_loss_trigger_type == TriggerType.LAST_PRICE

    async def test_the_tag_of_the_client_order_id_stays_on_a_fresh_id(
        self, sim, btc_usdt_perp, declared_account
    ):
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=PRICE)
        entry = declared_account.long_entry(btc_usdt_perp, 1.0).limit(90.0).tag(TAG)
        await declared_account.place_orders([entry]).execute()
        resting = await _the_resting_order(sim, btc_usdt_perp, OrderType.LIMIT)

        await declared_account.update_order(resting, limit_price=91.0).execute()

        moved = await _the_resting_order(sim, btc_usdt_perp, OrderType.LIMIT)
        assert tag_of(moved.client_order_id) == TAG
        assert moved.client_order_id != resting.client_order_id

    async def test_an_order_placed_by_hand_without_a_tag_is_moved_without_one(
        self, sim, btc_usdt_perp, declared_account
    ):
        sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=PRICE)
        await sim.exchange.place_limit_order(
            symbol=btc_usdt_perp, side=OrderSide.BUY, quantity=1.0, price=90.0
        )
        resting = await _the_resting_order(sim, btc_usdt_perp, OrderType.LIMIT)

        await declared_account.update_order(resting, limit_price=91.0).execute()

        moved = await _the_resting_order(sim, btc_usdt_perp, OrderType.LIMIT)
        assert moved.limit_price == 91.0
        assert not carries_this_bot(moved.client_order_id)
        assert tag_of(moved.client_order_id) is None
