import logging
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime, timedelta

import pytest

from robottraderslab import Symbol
from robottraderslab._core.tracked_positions import widen_tracked_positions
from robottraderslab.exchanges import PositionSide, client_order_id_carrying
from robottraderslab.strategies import (
    AccountSnapshot,
    AccountSnapshots,
    Execution,
    OrderSide,
    PositionTracker,
    StrategyRequirements,
    TrackedPosition,
    TrackingId,
    profile_tag,
)
from robottraderslab.strategies.futures import PositionSnapshot

ACCOUNT_NAME = "test"
ALPHA = TrackingId("BTC/USDT:USDT@1d#alpha")
BETA = TrackingId("ETH/USDT:USDT@1d#beta")
BTC = Symbol.create("BTC/USDT:USDT")
ETH = Symbol.create("ETH/USDT:USDT")
LONG_AGO = datetime(2026, 1, 5, tzinfo=UTC)
WITHIN_THE_READ = datetime(2026, 9, 1, tzinfo=UTC)
CANDLE_READ = datetime(2026, 8, 25, tzinfo=UTC)
CANDLE_OPEN = datetime(2026, 9, 17, tzinfo=UTC)
CANDLE_CLOSE = datetime(2026, 9, 18, tzinfo=UTC)


class _Account:
    def __init__(
        self, older_executions: Sequence[Execution], name: str = ACCOUNT_NAME
    ) -> None:
        self.name = name
        self._older_executions = older_executions
        self.reads: list[tuple[datetime, list[Symbol]]] = []

    async def _executions_since(
        self,
        since: datetime,
        symbols: Iterable[Symbol],
        *,
        max_attempts: int = 3,
        base_delay: float = 1.0,
    ) -> list[Execution]:
        self.reads.append((since, list(symbols)))
        return list(self._older_executions)


def _execution(
    execution_id: str,
    side: OrderSide,
    quantity: float,
    timestamp: datetime,
    tag: str | None = None,
    symbol: Symbol = BTC,
) -> Execution:
    return Execution(
        execution_id=execution_id,
        order_id=execution_id,
        symbol=symbol,
        side=side,
        price=100.0,
        quantity=quantity,
        timestamp=timestamp,
        client_order_id=client_order_id_carrying(tag) if tag is not None else None,
    )


def _position(symbol: Symbol, quantity: float) -> PositionSnapshot:
    return PositionSnapshot(
        symbol=symbol,
        side=PositionSide.LONG,
        quantity=quantity,
        average_entry_price=100.0,
        entry_time=LONG_AGO,
        leverage=1.0,
        liquidation_price=0.0,
    )


def _snapshots(venue_quantity: float, executions: list[Execution]) -> AccountSnapshots:
    return AccountSnapshots(
        {
            ACCOUNT_NAME: AccountSnapshot(
                account_name=ACCOUNT_NAME,
                positions={BTC: _position(BTC, venue_quantity)},
                executions=executions,
                executions_declared=True,
            )
        }
    )


async def _settle(
    requirements: StrategyRequirements,
    account_snapshots: AccountSnapshots,
    *,
    executions_since: datetime,
) -> None:
    if requirements.tracker.refresh_all(account_snapshots, CANDLE_OPEN, CANDLE_CLOSE):
        await widen_tracked_positions(
            requirements,
            account_snapshots,
            executions_since=executions_since,
            reported_since=CANDLE_OPEN,
            reported_until=CANDLE_CLOSE,
        )


@pytest.fixture
def create_run() -> Callable[
    ..., tuple[StrategyRequirements, "_Account", PositionTracker]
]:
    def _create_run(
        older_executions: Sequence[Execution] = (),
    ) -> tuple[StrategyRequirements, _Account, PositionTracker]:
        account = _Account(older_executions)
        requirements = StrategyRequirements()
        requirements.account.add(account, symbols=[BTC], positions=True)
        tracker = requirements.tracker.add(
            account=account,
            symbols={ALPHA: BTC},
            tags={ALPHA: profile_tag("1h", "a")},
        )
        return requirements, account, tracker

    return _create_run


async def test_a_position_within_the_candles_read_asks_for_no_second_read(create_run):
    requirements, account, tracker = create_run()
    opened = _execution("e1", OrderSide.BUY, 2.0, WITHIN_THE_READ, tag="1h-a")

    await _settle(requirements, _snapshots(2.0, [opened]), executions_since=CANDLE_READ)

    assert account.reads == []
    assert tracker.get(ALPHA) == TrackedPosition(PositionSide.LONG, 2.0)


async def test_a_profile_position_older_than_the_candles_read_is_still_its_own(
    create_run,
):
    opened = _execution("e1", OrderSide.BUY, 0.02, LONG_AGO, tag="1h-a")
    requirements, account, tracker = create_run([opened])

    await _settle(requirements, _snapshots(0.02, []), executions_since=CANDLE_READ)

    assert tracker.get(ALPHA) == TrackedPosition(PositionSide.LONG, 0.02)
    assert len(account.reads) == 1


async def test_the_second_read_is_scoped_to_the_symbols_that_ask_for_it():
    opened = _execution("e1", OrderSide.BUY, 0.02, LONG_AGO, tag="1h-a")
    account = _Account([opened])
    requirements = StrategyRequirements()
    requirements.account.add(account, symbols=[BTC, ETH], positions=True)
    requirements.tracker.add(
        account=account,
        symbols={ALPHA: BTC, BETA: ETH},
        tags={ALPHA: profile_tag("1h", "a"), BETA: profile_tag("1h", "b")},
    )
    explained = _execution(
        "e2", OrderSide.BUY, 3.0, WITHIN_THE_READ, tag="1h-b", symbol=ETH
    )
    snapshots = AccountSnapshots(
        {
            ACCOUNT_NAME: AccountSnapshot(
                account_name=ACCOUNT_NAME,
                positions={BTC: _position(BTC, 0.02), ETH: _position(ETH, 3.0)},
                executions=[explained],
                executions_declared=True,
            )
        }
    )

    await _settle(requirements, snapshots, executions_since=CANDLE_READ)

    assert account.reads[0][1] == [BTC]


async def test_each_account_is_read_for_the_symbols_it_holds():
    first = _Account([_execution("e1", OrderSide.BUY, 0.02, LONG_AGO, tag="1h-a")])
    second = _Account(
        [_execution("e2", OrderSide.BUY, 5.0, LONG_AGO, tag="1h-b", symbol=ETH)],
        name="second",
    )
    requirements = StrategyRequirements()
    requirements.account.add(first, symbols=[BTC], positions=True)
    requirements.account.add(second, symbols=[ETH], positions=True)
    requirements.tracker.add(
        account=first, symbols={ALPHA: BTC}, tags={ALPHA: profile_tag("1h", "a")}
    )
    requirements.tracker.add(
        account=second, symbols={BETA: ETH}, tags={BETA: profile_tag("1h", "b")}
    )
    snapshots = AccountSnapshots(
        {
            ACCOUNT_NAME: AccountSnapshot(
                account_name=ACCOUNT_NAME,
                positions={BTC: _position(BTC, 0.02)},
                executions=[],
                executions_declared=True,
            ),
            "second": AccountSnapshot(
                account_name="second",
                positions={ETH: _position(ETH, 5.0)},
                executions=[],
                executions_declared=True,
            ),
        }
    )

    await _settle(requirements, snapshots, executions_since=CANDLE_READ)

    assert first.reads[0][1] == [BTC]
    assert second.reads[0][1] == [ETH]


async def test_the_second_read_reaches_ninety_days_back(create_run):
    yesterday = datetime.now(UTC) - timedelta(days=1)
    opened = _execution("e1", OrderSide.BUY, 0.02, LONG_AGO, tag="1h-a")
    requirements, account, _ = create_run([opened])

    await _settle(requirements, _snapshots(0.02, []), executions_since=yesterday)

    assert (datetime.now(UTC) - account.reads[0][0]).days == 90


async def test_a_candle_read_reaching_further_than_the_span_keeps_its_own_start(
    create_run,
):
    long_lookback = datetime(2025, 1, 1, tzinfo=UTC)
    opened = _execution("e1", OrderSide.BUY, 0.02, LONG_AGO, tag="1h-a")
    requirements, account, _ = create_run([opened])

    await _settle(requirements, _snapshots(0.02, []), executions_since=long_lookback)

    assert account.reads[0][0] == long_lookback


async def test_a_hand_on_the_venue_before_the_candle_is_stated_and_not_warned(
    create_run, caplog
):
    by_hand = _execution("e1", OrderSide.BUY, 0.06, LONG_AGO)
    requirements, _, tracker = create_run([by_hand])

    with caplog.at_level(logging.INFO):
        await _settle(requirements, _snapshots(0.06, []), executions_since=CANDLE_READ)

    assert tracker.get(ALPHA) is None
    assert [record.levelname for record in caplog.records] == ["INFO"]
    assert "+0.06" in caplog.text


async def test_a_position_no_read_explains_names_how_far_back_it_reached(
    create_run, caplog
):
    requirements, _, _ = create_run()

    with caplog.at_level(logging.WARNING):
        await _settle(requirements, _snapshots(0.06, []), executions_since=CANDLE_READ)

    assert "+0.06" in caplog.text
    assert "left unattributed" in caplog.text
