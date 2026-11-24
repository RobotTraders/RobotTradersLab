import logging
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from robottraderslab import Symbol
from robottraderslab.cli import main
from robottraderslab.grid_search.command import main as grid_search_main
from robottraderslab.strategies import TradingMode
from robottraderslab_impulse import ImpulseStrategy

ETH = Symbol.create("ETH/USDT:USDT")
EX_CONFIG = 78

BOT_CONFIG = """\
[strategy]
strategy_class = "impulse"

[backtest]
initial_balance = { USDT = 1000.0 }
maker_fee_rate = 0.0002
taker_fee_rate = 0.0005
start_date = "2026-03-01"
end_date = "2026-03-20"

[backtest.ohlcv_provider]
ohlcv_provider = "mock"

[[strategy.profiles]]
symbol = "ETH/USDT:USDT"
timeframe = "1d"
total_balance_ratio = 0.5
trix_length = 5
signal_length = 8
trend_length = 20
stop_loss_pct = 0.05
"""

SEARCH_CONFIG = """\
max_workers = 1

[profile]
symbol = "ETH/USDT:USDT"
timeframe = "1d"

[[parameter_configs]]
parameter = "stop_loss_pct"
sampling = "linear"
min = 0.5
max = 1.5
num_samples = 2

[[parameter_configs]]
parameter = "trix_length"
sampling = "linear"
min = 5.0
max = 6.0
num_samples = 2
"""


def _section(symbol: str, **refused: Any) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "timeframe": "1d",
        "total_balance_ratio": 0.5,
        "trix_length": 9,
        "signal_length": 21,
        **refused,
    }


@pytest.fixture
def restore_logging() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers = handlers
    root.setLevel(level)


@pytest.fixture
def bot_folder(tmp_path: Path) -> Path:
    (tmp_path / "bot.toml").write_text(BOT_CONFIG, encoding="utf-8")
    return tmp_path


class TestLive:
    @pytest.mark.parametrize(
        ("refused_section", "reason"),
        [
            (
                _section(
                    "BTC/USDT:USDT", long_only=True, short_only=True, stop_loss_pct=0.1
                ),
                "long_only and short_only cannot both be true",
            ),
            (
                _section("BTC/USDT:USDT", stop_loss_pct=1.5),
                "stop_loss_pct must be in (0, 1), got 1.5",
            ),
        ],
    )
    def test_a_section_the_class_refuses_sits_out_while_the_other_symbol_trades(
        self,
        make_strategy: Callable[..., ImpulseStrategy],
        caplog,
        refused_section,
        reason,
    ):
        with caplog.at_level(logging.ERROR):
            strategy = make_strategy(
                [_section("ETH/USDT:USDT"), refused_section], TradingMode.LIVE
            )

        assert [profile.symbol for profile in strategy.profiles] == [ETH]
        (refusal,) = caplog.records
        assert refusal.levelno == logging.ERROR
        assert refusal.getMessage().startswith(
            "ImpulseStrategy: profile 2 (BTC/USDT:USDT 1d): " + reason
        )


@pytest.mark.usefixtures("restore_logging")
class TestBacktest:
    @pytest.fixture
    def run_command(self, monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
        monkeypatch.setattr(logging, "shutdown", lambda *_: None)

        def _run(*arguments: str) -> None:
            monkeypatch.setattr(sys, "argv", ["rtlab", *arguments])
            main()

        return _run

    def test_a_section_the_class_refuses_stops_the_run_naming_the_profile(
        self, bot_folder, run_command, capsys
    ):
        bot_toml = bot_folder / "bot.toml"
        bot_toml.write_text(
            BOT_CONFIG + "long_only = true\nshort_only = true\n", encoding="utf-8"
        )

        with pytest.raises(SystemExit) as exit_info:
            run_command("backtest", str(bot_folder), "--no-chart", "--no-save")

        assert exit_info.value.code == EX_CONFIG
        assert capsys.readouterr().err.strip() == (
            f"Error: `{bot_toml}`: ImpulseStrategy: profile 1 (ETH/USDT:USDT 1d): "
            "long_only and short_only cannot both be true; leave both out to "
            "trade both directions"
        )


class TestSweep:
    def test_a_point_the_class_refuses_is_recorded_as_failed_and_the_others_are_measured(
        self, bot_folder, tmp_path
    ):
        (bot_folder / "grid-search.toml").write_text(SEARCH_CONFIG, encoding="utf-8")
        exported = tmp_path / "results.csv"

        grid_search_main(bot_folder, exported)

        points = pd.read_csv(exported)
        refused = points[points["stop_loss_pct"] == 1.5]
        measured = points[points["stop_loss_pct"] == 0.5]
        assert refused["error"].str.contains("stop_loss_pct must be in").all()
        assert measured["error"].isna().all()
        assert measured["roi"].notna().all()
