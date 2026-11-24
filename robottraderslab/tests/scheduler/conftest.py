from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from robottraderslab.live import working_copy_file
from robottraderslab.live.working_copy import write_working_copy


@pytest.fixture
def create_registry(tmp_path: Path) -> Callable[[str], Path]:
    def _create_registry(content: str) -> Path:
        registry_file = tmp_path / "scheduler.toml"
        registry_file.write_text(content)
        return registry_file

    return _create_registry


@pytest.fixture
def write_bot(tmp_path: Path) -> Callable[..., Path]:
    def _write_bot(
        folder: str,
        *,
        name: str | None = None,
        exchange: str | None = "bitget",
        account: str | None = None,
        secret: str | None = None,
        symbols: tuple[str, ...] = (),
    ) -> Path:
        bot_folder = tmp_path / folder if name is None else tmp_path / folder / name
        config = bot_folder / "bot.toml"
        config.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        declared = {
            "exchange": exchange,
            "account_name": account,
            "secret_name": secret,
        }
        trading_account = [
            f'{key} = "{value}"' for key, value in declared.items() if value is not None
        ]
        if trading_account:
            lines += ["[live.trading_account]", *trading_account]
        lines += [f'[[strategy.profiles]]\nsymbol = "{symbol}"' for symbol in symbols]
        config.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return config

    return _write_bot


@pytest.fixture
def keep_working_copy() -> Callable[[Path], Path]:
    def _keep_working_copy(bot_file: Path) -> Path:
        copy_file = working_copy_file(bot_file)
        write_working_copy(copy_file, bot_file.read_bytes())
        return copy_file

    return _keep_working_copy


@pytest.fixture
def keep_only_working_copy(
    keep_working_copy: Callable[[Path], Path],
) -> Callable[[Path], Path]:
    def _keep_only_working_copy(bot_file: Path) -> Path:
        copy_file = keep_working_copy(bot_file)
        bot_file.unlink()
        return copy_file

    return _keep_only_working_copy


@pytest.fixture
def write_unusable_copy() -> Callable[..., Path]:
    def _write_unusable_copy(bot_file: Path, document: bytes = b"name = [\n") -> Path:
        copy_file = working_copy_file(bot_file)
        write_working_copy(copy_file, document)
        return copy_file

    return _write_unusable_copy


@pytest.fixture
def write_bot_with_working_copy(
    write_bot: Callable[..., Path], keep_working_copy: Callable[[Path], Path]
) -> Callable[..., Path]:
    def _write_bot_with_working_copy(folder: str, **declared: Any) -> Path:
        bot_file = write_bot(folder, **declared)
        keep_working_copy(bot_file)
        return bot_file

    return _write_bot_with_working_copy


@pytest.fixture
def edit_first_bot(write_bot: Callable[..., Path]) -> Callable[..., Path]:
    def _edit_first_bot(*symbols: str) -> Path:
        return write_bot("first", name="grid", account="main", symbols=symbols)

    return _edit_first_bot


@pytest.fixture
def edit_second_bot(write_bot: Callable[..., Path]) -> Callable[..., Path]:
    def _edit_second_bot(*symbols: str) -> Path:
        return write_bot("second", name="scalp", account="main", symbols=symbols)

    return _edit_second_bot


@pytest.fixture
def first_symbol() -> str:
    return "BTC/USDT:USDT"


@pytest.fixture
def second_symbol() -> str:
    return "ETH/USDT:USDT"


@pytest.fixture
def two_bots_with_working_copies(
    write_bot_with_working_copy: Callable[..., Path],
    first_symbol: str,
    second_symbol: str,
) -> tuple[Path, Path]:
    return _two_bots(write_bot_with_working_copy, first_symbol, second_symbol)


@pytest.fixture
def two_bots_on_one_symbol(
    write_bot: Callable[..., Path], first_symbol: str
) -> tuple[Path, Path]:
    return _two_bots(write_bot, first_symbol, first_symbol)


def _two_bots(
    write: Callable[..., Path], first_symbol: str, second_symbol: str
) -> tuple[Path, Path]:
    first = write("first", name="grid", account="main", symbols=(first_symbol,))
    second = write("second", name="scalp", account="main", symbols=(second_symbol,))
    return first, second
