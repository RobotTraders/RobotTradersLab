from collections.abc import Callable
from pathlib import Path

import pytest


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
