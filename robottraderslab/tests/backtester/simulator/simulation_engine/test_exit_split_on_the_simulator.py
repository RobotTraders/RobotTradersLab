from datetime import datetime

import pytest
from simulation_fixture import SimulationFixture

from robottraderslab import Symbol
from robottraderslab._internal import split_exits
from robottraderslab.futures import FuturesAccount
from robottraderslab.strategies import (
    BookKeeper,
    PositionSide,
    PositionTracker,
    StrategyRequirements,
    TrackedPosition,
    TrackingId,
    profile_tag,
)
from robottraderslab_core.engine import execute_trading_actions
from robottraderslab_core.exchanges import BaseExchangeAction
from robottraderslab_core.strategies import AccountSnapshot, AccountSnapshots

_PRICE = 100.0
_SINCE = datetime.min
_UNTIL = datetime.max

_FIRST_QUANTITY = 0.00001
_SECOND_QUANTITY = 0.00002
_OPPOSITE_QUANTITY = 0.00001

FIRST = TrackingId("BTC/USDT:USDT@1h-a")
SECOND = TrackingId("BTC/USDT:USDT@1h-c")
OPPOSITE = TrackingId("BTC/USDT:USDT@1h-b")


@pytest.fixture
def strategy_requirements(
    declared_account: FuturesAccount, btc_usdt_perp: Symbol
) -> StrategyRequirements:
    strategy_requirements = StrategyRequirements()
    strategy_requirements.account.add(
        declared_account, symbols=[btc_usdt_perp], positions=True
    )
    return strategy_requirements


@pytest.fixture
def position_tracker(
    strategy_requirements: StrategyRequirements,
    declared_account: FuturesAccount,
    btc_usdt_perp: Symbol,
) -> PositionTracker:
    return strategy_requirements.tracker.add(
        declared_account,
        symbols={FIRST: btc_usdt_perp, SECOND: btc_usdt_perp, OPPOSITE: btc_usdt_perp},
        tags={
            FIRST: profile_tag("1h", "a"),
            SECOND: profile_tag("1h", "c"),
            OPPOSITE: profile_tag("1h", "b"),
        },
    )


async def test_two_longs_exiting_against_a_short_send_a_plain_order_for_the_surplus(
    sim, declared_account, btc_usdt_perp, strategy_requirements, position_tracker
):
    await _stack_two_longs_and_a_short(
        strategy_requirements, sim, declared_account, btc_usdt_perp
    )
    exits = [
        declared_account.close_tracked_position(
            btc_usdt_perp, TrackedPosition(PositionSide.LONG, quantity)
        )
        .tag(tag)
        .build()
        for tag, quantity in (("1h-a", _FIRST_QUANTITY), ("1h-c", _SECOND_QUANTITY))
    ]

    sent = await _send_and_fill(
        strategy_requirements, sim, declared_account, btc_usdt_perp, *exits
    )

    held = await sim.exchange.get_open_positions([btc_usdt_perp])
    strategy_requirements.tracker.refresh_all(
        await _account_snapshots(sim, declared_account, btc_usdt_perp), _SINCE, _UNTIL
    )
    assert [(action.quantity, action.reduce_only) for action in sent] == [
        (_FIRST_QUANTITY, True),
        (_FIRST_QUANTITY, True),
        (_FIRST_QUANTITY, False),
    ]
    assert held[btc_usdt_perp].side == PositionSide.SHORT
    assert held[btc_usdt_perp].quantity == _OPPOSITE_QUANTITY
    assert position_tracker.get(OPPOSITE) == TrackedPosition(
        PositionSide.SHORT, _OPPOSITE_QUANTITY
    )
    assert position_tracker.get(FIRST) is None
    assert position_tracker.get(SECOND) is None


async def _stack_two_longs_and_a_short(
    strategy_requirements: StrategyRequirements,
    sim: SimulationFixture,
    futures_account: FuturesAccount,
    btc_usdt_perp: Symbol,
) -> None:
    await _send_and_fill(
        strategy_requirements,
        sim,
        futures_account,
        btc_usdt_perp,
        futures_account.long_entry(btc_usdt_perp, _FIRST_QUANTITY).tag("1h-a").build(),
        futures_account.long_entry(btc_usdt_perp, _SECOND_QUANTITY).tag("1h-c").build(),
    )
    await _send_and_fill(
        strategy_requirements,
        sim,
        futures_account,
        btc_usdt_perp,
        futures_account.short_entry(btc_usdt_perp, _OPPOSITE_QUANTITY)
        .tag("1h-b")
        .build(),
    )


async def _send_and_fill(
    strategy_requirements: StrategyRequirements,
    sim: SimulationFixture,
    futures_account: FuturesAccount,
    btc_usdt_perp: Symbol,
    *actions: BaseExchangeAction,
) -> list[BaseExchangeAction]:
    bookkeeper = BookKeeper()
    for action in actions:
        bookkeeper.add(action)
    await split_exits(
        strategy_requirements,
        await _account_snapshots(sim, futures_account, btc_usdt_perp),
        bookkeeper,
    )
    sent = bookkeeper.list_actions()
    await execute_trading_actions(sent)
    sim.simulate_on_current_ohlcvs(btc_usdt_perp, close=_PRICE)
    return sent


async def _account_snapshots(
    sim: SimulationFixture, futures_account: FuturesAccount, btc_usdt_perp: Symbol
) -> AccountSnapshots:
    return AccountSnapshots(
        {
            futures_account.name: AccountSnapshot(
                account_name=futures_account.name,
                positions=await sim.exchange.get_open_positions([btc_usdt_perp]),
                executions=sim.simulation_engine.get_executions_since(_SINCE),
                executions_declared=True,
            )
        }
    )
