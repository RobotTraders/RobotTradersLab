import json
import os
from collections.abc import Callable
from itertools import chain, repeat
from pathlib import Path
from typing import Any
from unittest.mock import Mock, create_autospec

import pytest
from dry_run_helpers import MESSAGE, BookedCall, PlainStrategy, ProbeStrategy
from live_cycle_helpers import BTC, ETH, market_orders

from robottraderslab import Symbol
from robottraderslab.bootstrap import BotConfig
from robottraderslab.exchanges import OrderSide, PlacedOrder, PositionSnapshot
from robottraderslab.futures import FuturesAccount
from robottraderslab.live import main
from robottraderslab.live.working_copy import write_working_copy

SOL = Symbol.create("SOL/USDT:USDT")
BOT_FOLDER = "probe-bot"
VENUE = "simulator"
FAILING_VENUE = "unreachable"
AN_OLD_MOMENT_NS = 1_000_000_000_000_000_000

BOOKED: list[BookedCall] = []
SIGNALLED: list[Symbol] = []
PROBE_STRATEGY = f"{__name__}.LiveProbe"
PLAIN_STRATEGY = f"{__name__}.PlainProbe"
POSITION_STRATEGY = f"{__name__}.PositionProbe"
SEEN_POSITIONS: list[PositionSnapshot | None] = []


class LiveProbe(ProbeStrategy):
    """The probe strategy a bot built by `bot_toml` names.

    A cycle builds it through the loader, out of the test's reach, so it
    records into `BOOKED` and `SIGNALLED`.
    """

    def __init__(self, **arguments: Any) -> None:
        super().__init__(**arguments)
        self.booked = BOOKED
        self.signalled = SIGNALLED


class PositionProbe(LiveProbe):
    """A probe strategy recording the position each booking slot sees.

    The position its snapshot holds for each profile whose booking slot runs
    goes into `SEEN_POSITIONS`.
    """

    def book_profile_actions(
        self,
        profile: Any,
        ohlcvs: Any,
        account_snapshots: Any,
        timestamp: Any,
        bookkeeper: Any,
    ) -> None:
        SEEN_POSITIONS.append(
            account_snapshots.of(self.account).position(profile.symbol)
        )
        super().book_profile_actions(
            profile, ohlcvs, account_snapshots, timestamp, bookkeeper
        )


class PlainProbe(PlainStrategy):
    """A strategy that is no profile strategy, on BTC and ETH.

    Its booking raises when `fail` is `booking`. Each one built joins
    `PLAIN_PROBES`.
    """

    def __init__(
        self, account: FuturesAccount, fail: str = "", **settings: Any
    ) -> None:
        super().__init__(account)
        self._fail = fail
        PLAIN_PROBES.append(self)

    def book_trading_actions(
        self,
        ohlcvs: Any,
        account_snapshots: Any,
        timestamp: Any,
        bookkeeper: Any,
        triggered_timeframes: Any,
    ) -> None:
        super().book_trading_actions(
            ohlcvs, account_snapshots, timestamp, bookkeeper, triggered_timeframes
        )
        if self._fail == "booking":
            raise ValueError(MESSAGE)
        for symbol in (BTC, ETH):
            bookkeeper.add(self._account.long_entry(symbol, 1.0))


PLAIN_PROBES: list[PlainProbe] = []


def section(symbol: Symbol, timeframe: str = "1d", **fields: Any) -> str:
    """A profile section of the probe strategy, as TOML.

    The field values are JSON-encodable, which is TOML-encodable for these types.
    """
    lines = [f'symbol = "{symbol}"', f'timeframe = "{timeframe}"']
    lines += [f"{key} = {json.dumps(value)}" for key, value in fields.items()]
    return "[[strategy.profiles]]\n" + "\n".join(lines) + "\n"


def bot_toml(
    *sections: str,
    strategy: str = PROBE_STRATEGY,
    strategy_keys: str = "",
    head: str = 'secrets_file = "creds.toml"\n',
    exchange: str = VENUE,
    provider_keys: str = "lookback = 10\n",
    tail: str = "",
) -> str:
    """A bot of a probe strategy on the plugged venue and mock candles, as TOML.

    Args:
        sections: The profile sections, each as `section` writes it.
        strategy: The dotted path of the strategy class.
        strategy_keys: TOML text placed under `[strategy]` as it is.
        head: TOML text placed before `[strategy]` as it is.
        exchange: The trading account's exchange.
        provider_keys: TOML text placed under `[live.ohlcv_provider]` as it is,
            after the provider's name.
        tail: TOML text placed after the `[live]` sections as it is.
    """
    return (
        f'{head}\n[strategy]\nstrategy_class = "{strategy}"\n{strategy_keys}\n'
        + "\n".join(sections)
        + f'\n[live.trading_account]\nexchange = "{exchange}"\n'
        'secret_name = "my_account"\n\n'
        '[live.ohlcv_provider]\nohlcv_provider = "mock"\n' + provider_keys + tail
    )


def save(folder: Path, text: str | bytes) -> None:
    """Save `bot.toml`, as UTF-8 when given text."""
    content = text.encode() if isinstance(text, str) else text
    (folder / "bot.toml").write_bytes(content)


def save_bot(folder: Path, *sections: str, **options: Any) -> str:
    """Save `bot.toml` as the bot `bot_toml` builds of the sections and options.

    Returns:
        The text saved.
    """
    text = bot_toml(*sections, **options)
    save(folder, text)
    return text


def write_draft(draft_path: Path, *sections: str, **options: Any) -> Path:
    """Write the bot `bot_toml` builds of the sections and options to the path.

    The folders of the path are created.

    Returns:
        The path written.
    """
    draft_path.parent.mkdir(parents=True, exist_ok=True)
    draft_path.write_text(bot_toml(*sections, **options))
    return draft_path


def saving(text: str | bytes) -> Callable[[Path], None]:
    """An edit saving `bot.toml` as the text."""
    return lambda folder: save(folder, text)


def copy_of(folder: Path) -> Path:
    """Where the bot of the folder keeps its working copy."""
    return folder / ".rtlab" / "bot.working.toml"


def read_copy(folder: Path) -> str:
    """The text of the bot's working copy."""
    return copy_of(folder).read_text()


def cycle(
    folder: Path,
    *,
    logfile_override: Path | None = None,
    logfiles_retention_days: int | None = None,
) -> int:
    """Run one live cycle of the bot, as `rtlab live` does, with the console off."""
    return main(
        folder=folder,
        logfile_override=logfile_override,
        console=False,
        logfiles_retention_days=logfiles_retention_days,
    )


def seeded(folder: Path, text: str) -> None:
    """Give the bot a working copy holding `text`, `bot.toml` saved equal to it.

    Creates the folder of the copy.
    """
    save(folder, text)
    write_working_copy(copy_of(folder), text.encode())


def seed_bot(folder: Path, *sections: str, **options: Any) -> str:
    """Save a bot as `save_bot` does and give it a working copy of the same text.

    Returns:
        The text of both files.
    """
    text = bot_toml(*sections, **options)
    seeded(folder, text)
    return text


def seed_old_copy(folder: Path) -> Path:
    """Give the bot a working copy equal to its `bot.toml`, modified long ago.

    Returns:
        The working copy's path.
    """
    seeded(folder, (folder / "bot.toml").read_text())
    copy = copy_of(folder)
    os.utime(copy, ns=(AN_OLD_MOMENT_NS, AN_OLD_MOMENT_NS))
    return copy


def bot_loaded_from(folder: Path, bot_toml_text: str) -> BotConfig:
    """The bot the text declares, as a file named `bot.toml` in the folder would."""
    return BotConfig.from_bytes(bot_toml_text.encode(), folder / "bot.toml")


def copies_at_each_call(folder: Path, method: Mock, result: object = None) -> list[str]:
    """The working copy's text as each call of the venue's method arrives.

    Sets the method's side effect, so each call answers `result`. The folder
    must hold the copy by the time the method is called.
    """
    seen: list[str] = []

    def record(*arguments: Any, **keywords: Any) -> object:
        seen.append(read_copy(folder))
        return result

    method.side_effect = record
    return seen


def copies_at_each_order(folder: Path, venue: Mock) -> list[str]:
    """The working copy's text as each market order reaches the venue.

    Sets `venue.place_market_order`'s side effect; the folder must hold the copy
    by the time an order is placed.
    """
    return copies_at_each_call(
        folder, venue.place_market_order, PlacedOrder(order_id="market-1")
    )


def refuse_copy_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every replacement of a file fail, the working copy's write included."""

    def refuse(self: Path, target: Any) -> Path:
        raise PermissionError("held by a reader")

    monkeypatch.setattr(Path, "replace", refuse)


def run_before_slot(
    monkeypatch: pytest.MonkeyPatch,
    strategy_class: type[ProbeStrategy],
    slot: str,
    before: Callable[[], object],
) -> None:
    """Call `before` each time the strategy class's slot is called, then the slot."""
    original = getattr(strategy_class, slot)

    def run(*arguments: Any) -> Any:
        before()
        return original(*arguments)

    monkeypatch.setattr(strategy_class, slot, run)


def failing_the_first_call(error: BaseException) -> Mock:
    """A call that raises the error the first time and does nothing after."""
    return create_autospec(_do_nothing, side_effect=chain([error], repeat(None)))


def entries(venue: Mock) -> list[tuple[str, float]]:
    """The symbol and quantity of each market buy the venue received."""
    return [
        (str(symbol), quantity)
        for symbol, side, quantity in market_orders(venue)
        if side is OrderSide.BUY
    ]


def entry_symbols(venue: Mock) -> set[str]:
    return {symbol for symbol, _ in entries(venue)}


def closes(venue: Mock) -> list[str]:
    """The symbol of each market sell the venue received."""
    return [
        str(symbol)
        for symbol, side, _ in market_orders(venue)
        if side is OrderSide.SELL
    ]


def logged(folder: Path, level: str) -> list[str]:
    """The lines of the bot's log at the level, as the log writes it, such as `ERROR`."""
    log = folder / "logs" / "live.log"
    if not log.is_file():
        return []
    marker = f": {level}: "
    return [line for line in log.read_text().splitlines() if marker in line]


def booked_symbols() -> list[str]:
    """The symbol of each booking slot the probe strategy ran."""
    return [str(call.symbol) for call in BOOKED]


def signalled_symbols() -> list[str]:
    """The symbol of each signal slot the probe strategy ran."""
    return [str(symbol) for symbol in SIGNALLED]


def plain_signal_runs() -> int:
    """The runs of the signal method of every plain probe built."""
    return sum(probe.signal_calls for probe in PLAIN_PROBES)


def forget_probes() -> None:
    """Empty what the probe strategies recorded."""
    BOOKED.clear()
    SIGNALLED.clear()
    SEEN_POSITIONS.clear()
    PLAIN_PROBES.clear()


def _do_nothing() -> None:
    pass


TWO = (section(BTC), section(ETH))
THREE = (section(BTC), section(ETH), section(SOL))
REFUSED_BTC = (section(BTC, refused=True), section(ETH))
DOUBLED_BTC = (section(BTC, quantity=2.0), section(ETH))
CLOSING_BTC = (section(BTC, closes=True), section(ETH))
