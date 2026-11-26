import ast
import sys
from collections.abc import Awaitable, Callable, Iterator
from datetime import datetime, timedelta
from importlib import import_module, invalidate_caches
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from robottraderslab import Symbol
from robottraderslab.analyser import create_trade_aggregation
from robottraderslab.backtester import run_backtest
from robottraderslab.bootstrap import BotConfig
from robottraderslab.exceptions import ProfileConfigError
from robottraderslab.futures import FuturesAccount
from robottraderslab.strategies import (
    AccountSnapshot,
    AccountSnapshots,
    Balance,
    BookKeeper,
    Candles,
    OHLCVs,
    OrderSide,
    PositionSide,
    TradingMode,
    TradingSystem,
)
from robottraderslab.strategies.futures import (
    FuturesMarketOrderAction,
    MarginMode,
    MarginSettings,
    PositionSnapshot,
)

GUIDE_LINK = (
    "https://robottraders.io/lab/docs/implement-a-strategy.html#the-shape-of-a-strategy"
)
SIGN_OFF = [
    "# All the best with building and implementing your strategy. I wish you every",
    "# success with it, and thank you for using the Lab.",
    "# Happy trading, and take care.",
    "# Louis",
]
IMPULSE_FOLDER_LINK = "https://github.com/RobotTraders/RobotTradersLab/tree/main/plugins/robottraderslab-impulse"
CLOSES = [
    10.0,
    10.0,
    10.0,
    12.0,
    14.0,
    12.0,
    8.0,
    6.0,
    8.0,
    12.0,
    14.0,
    12.0,
    12.0,
    10.0,
]
FAST_BELOW_SLOW_CANDLES = [6, 7, 8, 12, 13]
FAST_ABOVE_SLOW_CANDLES = [3, 4, 5, 9, 10, 11]
AFTER_THE_CROSS_CANDLE = 7
OPEN_SIDE_CANDLE = 4
CROSSING_CANDLE = 6
CANDLE_HOURS = 4
HELD_QUANTITY = 1.5
SYMBOL = Symbol.create("BTC/USDC:USDC")
MIRRORED_CLOSES = [2 * max(CLOSES) - close for close in CLOSES]
STRATEGY_BY_MODULE_OVERLAY = """
[strategy]
strategy_class = "my_strategy.main:MyStrategy"
"""


@pytest.fixture
def generated_module(
    package_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[ModuleType]:
    monkeypatch.syspath_prepend(str(package_folder / "src"))
    yield import_module("my_strategy.main")
    sys.modules.pop("my_strategy.chart_indicators", None)
    sys.modules.pop("my_strategy.main", None)
    sys.modules.pop("my_strategy", None)


@pytest.fixture
def installed_package(
    package_folder: Path,
    generated_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fresh_entry_point_lookups: None,
    read_pyproject: Callable[[Path], dict],
) -> None:
    """importlib caches path-finder results, so the caches are invalidated once
    the distribution folder is on sys.path.
    """
    entry_points_by_group = read_pyproject(package_folder)["project"]["entry-points"]
    distribution = tmp_path / "site" / "my_strategy-0.1.0.dist-info"
    distribution.mkdir(parents=True)
    (distribution / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: my-strategy\nVersion: 0.1.0\n",
        encoding="utf-8",
    )
    (distribution / "entry_points.txt").write_text(
        "".join(
            f"[{group}]\n"
            + "".join(f"{name} = {value}\n" for name, value in registered.items())
            for group, registered in entry_points_by_group.items()
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(distribution.parent))
    invalidate_caches()


@pytest.fixture
def make_my_strategy(
    generated_module: ModuleType, make_futures_account: Callable[..., FuturesAccount]
) -> Callable[[list[dict[str, object]]], Any]:
    def make(profiles: list[dict[str, object]]) -> Any:
        return generated_module.MyStrategy(
            account=make_futures_account(),
            trading_system=TradingSystem(trading_mode=TradingMode.BACKTEST),
            config_dir=None,
            profiles=profiles,
        )

    return make


@pytest.fixture
def make_booked_orders(
    make_my_strategy: Callable[[list[dict[str, object]]], Any],
    make_dated_ohlcvs: Callable[..., OHLCVs],
    short_averages_profile: dict[str, object],
) -> Callable[..., Awaitable[list[FuturesMarketOrderAction]]]:
    async def book(
        held_side: PositionSide, closes: list[float], candle: int
    ) -> list[FuturesMarketOrderAction]:
        """Book a candle for a strategy holding a position and return its market orders.

        Args:
            held_side: The side of the position the account holds.
            closes: One close per candle.
            candle: The index of the candle booked.
        """
        ohlcvs = make_dated_ohlcvs(SYMBOL, "4h", "4h", closes)
        my_strategy = make_my_strategy([short_averages_profile])
        await my_strategy.account._read_instruments([SYMBOL])
        my_strategy.generate_trading_signals(ohlcvs)
        next(ohlcvs._iter_timeframes(start_idx=candle))
        bookkeeper = BookKeeper()

        my_strategy.book_trading_actions(
            ohlcvs,
            _snapshots_holding(my_strategy.account, held_side),
            datetime(2024, 1, 1) + timedelta(hours=CANDLE_HOURS * candle),
            bookkeeper,
            ["4h"],
        )

        return [
            action
            for action in bookkeeper.list_actions()
            if isinstance(action, FuturesMarketOrderAction)
        ]

    return book


@pytest.fixture
def short_averages_profile(btc_4h_profile: dict[str, object]) -> dict[str, object]:
    return {**btc_4h_profile, "fast_ma_length": 2, "slow_ma_length": 3}


@pytest.fixture
def strategy_source(package_folder: Path) -> str:
    return _module_source(package_folder, "main.py")


class TestStrategyTemplate:
    def test_the_comment_explains_the_profile_strategy_and_links_the_guide(
        self, strategy_source
    ):
        comment = _comment_lines(strategy_source)

        assert (
            "# This is a profile strategy: a strategy where you configure"
            in " ".join(comment)
        )
        assert f"# {GUIDE_LINK}" in comment

    def test_the_sign_off_comes_before_the_profile_strategy_paragraph(
        self, strategy_source
    ):
        lines = strategy_source.splitlines()
        paragraph_line = next(
            i
            for i, line in enumerate(lines)
            if line.startswith("# This is a profile strategy:")
        )

        assert lines[paragraph_line - len(SIGN_OFF) - 1 : paragraph_line] == [
            *SIGN_OFF,
            "#",
        ]

    def test_a_profile_whose_fast_length_is_not_below_the_slow_one_is_refused(
        self, make_my_strategy, btc_4h_profile
    ):
        with pytest.raises(
            ProfileConfigError,
            match=r"MyStrategy: profile 1 \(BTC/USDC:USDC 4h\): "
            r"fast_ma_length \(50\) must be below slow_ma_length \(30\)",
        ):
            make_my_strategy([{**btc_4h_profile, "fast_ma_length": 50}])

    def test_the_exits_are_the_states_of_the_two_averages(
        self,
        make_my_strategy,
        make_dated_ohlcvs,
        candles_where_raised,
        short_averages_profile,
    ):
        ohlcvs = make_dated_ohlcvs(SYMBOL, "4h", "4h", CLOSES)
        my_strategy = make_my_strategy([short_averages_profile])

        my_strategy.generate_profile_signals(my_strategy.profiles[0], ohlcvs)

        assert candles_where_raised(ohlcvs, SYMBOL, "4h", "long_exit") == (
            FAST_BELOW_SLOW_CANDLES
        )
        assert candles_where_raised(ohlcvs, SYMBOL, "4h", "short_exit") == (
            FAST_ABOVE_SLOW_CANDLES
        )

    @pytest.mark.parametrize(
        ("held_side", "closes"),
        [
            (PositionSide.LONG, CLOSES),
            (PositionSide.SHORT, MIRRORED_CLOSES),
        ],
    )
    async def test_a_position_on_an_open_side_books_no_exit(
        self, make_booked_orders, held_side, closes
    ):
        orders = await make_booked_orders(held_side, closes, OPEN_SIDE_CANDLE)

        assert orders == []

    @pytest.mark.parametrize(
        ("held_side", "closing_side", "closes"),
        [
            (PositionSide.LONG, OrderSide.SELL, CLOSES),
            (PositionSide.SHORT, OrderSide.BUY, MIRRORED_CLOSES),
        ],
    )
    async def test_a_position_on_a_closed_side_is_exited_on_a_candle_after_the_cross(
        self, make_booked_orders, held_side, closing_side, closes
    ):
        orders = await make_booked_orders(held_side, closes, AFTER_THE_CROSS_CANDLE)

        assert [(order.side, order.quantity) for order in orders] == [
            (closing_side, HELD_QUANTITY)
        ]

    @pytest.mark.parametrize(
        ("held_side", "closing_side", "closes"),
        [
            (PositionSide.LONG, OrderSide.SELL, CLOSES),
            (PositionSide.SHORT, OrderSide.BUY, MIRRORED_CLOSES),
        ],
    )
    async def test_a_position_on_the_side_the_cross_closes_is_exited_before_the_opposite_entry(
        self, make_booked_orders, held_side, closing_side, closes
    ):
        orders = await make_booked_orders(held_side, closes, CROSSING_CANDLE)

        assert [(order.side, order.is_exit) for order in orders] == [
            (closing_side, True),
            (closing_side, False),
        ]

    def test_the_comment_tells_what_a_refused_profile_does(self, strategy_source):
        comment = _comment_text(strategy_source)

        assert "every profile on its symbol sits out the run" in comment
        assert "the profiles of the other symbols trade" in comment
        assert "in a backtest the run stops naming the profile" in comment
        assert "A BotCriticalError raised here would stop the whole bot" in comment

    def test_the_comment_names_position_tracking_and_the_impulse_folder(
        self, strategy_source
    ):
        comment = _comment_text(strategy_source)

        assert "several timeframes or parameter sets needs position tracking" in comment
        assert IMPULSE_FOLDER_LINK in comment

    @pytest.mark.usefixtures("generated_module")
    def test_the_chart_lines_of_the_template_draw_the_two_averages_of_the_profile(
        self, btc_4h_profile
    ):
        close = np.arange(1.0, 41.0)
        candles = Candles(open=close, high=close, low=close, close=close, volume=close)

        lines = import_module("my_strategy.chart_indicators").chart_lines(
            candles, btc_4h_profile
        )

        assert [line.name for line in lines] == ["SMA 10", "SMA 30"]
        assert [line.pane for line in lines] == ["price", "price"]
        assert [line.values[-1] for line in lines] == [35.5, 25.5]
        assert [int(np.isnan(line.values).sum()) for line in lines] == [9, 29]

    def test_the_chart_template_imports_no_private_or_template_module(
        self, package_folder
    ):
        source = _module_source(package_folder, "chart_indicators.py")

        assert not [
            name
            for name in _imported_names(source)
            if name.startswith("robottraderslab.ma_strategy")
            or any(part.startswith("_") for part in name.split("."))
        ]

    @pytest.mark.usefixtures("generated_module")
    def test_the_bot_backtests_with_trades(self, workspace):
        np.random.seed(0)
        bot_config = BotConfig.from_file(
            workspace / "my-strategy" / "bot.toml", overlay=STRATEGY_BY_MODULE_OVERLAY
        )
        assert bot_config.backtest is not None
        bot_config.backtest.ohlcv_provider = {"ohlcv_provider": "mock"}

        outputs = run_backtest(bot_config)

        assert len(create_trade_aggregation(outputs.fills).trades) > 0
        assert set(outputs.fills["fill_type"]) == {
            "enter_long",
            "enter_short",
            "exit_long",
            "exit_short",
        }
        entries = outputs.fills[outputs.fills["fill_type"].str.startswith("enter")]
        assert entries["gross_quantity"].min() > entries["gross_quantity"].iloc[0] / 2

    @pytest.mark.usefixtures("installed_package")
    def test_the_backtest_report_draws_the_two_averages(self, workspace, read_payload):
        np.random.seed(0)
        bot_config = BotConfig.from_file(workspace / "my-strategy" / "bot.toml")
        assert bot_config.backtest is not None
        bot_config.backtest.ohlcv_provider = {"ohlcv_provider": "mock"}

        run_backtest(bot_config).create_analyser()

        (report,) = (workspace / "my-strategy" / "backtests").glob("*/report.html")
        (market,) = read_payload(report)["markets"]
        assert [series["name"] for series in market["series"]] == [
            "SMA 10",
            "SMA 30",
        ]


def _snapshots_holding(
    futures_account: FuturesAccount, position_side: PositionSide
) -> AccountSnapshots:
    return AccountSnapshots(
        {
            futures_account.name: AccountSnapshot(
                account_name=futures_account.name,
                positions={
                    SYMBOL: PositionSnapshot(
                        symbol=SYMBOL,
                        side=position_side,
                        quantity=HELD_QUANTITY,
                        average_entry_price=10.0,
                        entry_time=datetime(2024, 1, 1),
                        leverage=1.0,
                        liquidation_price=0.0,
                        stop_loss_price=None,
                        take_profit_price=None,
                    )
                },
                balances={"USDC": Balance(locked=0.0, total=100_000.0)},
                margin_settings={
                    SYMBOL: MarginSettings(leverage=None, margin_mode=MarginMode.CROSS)
                },
            )
        }
    )


def _module_source(package_folder: Path, module: str) -> str:
    return (package_folder / "src" / "my_strategy" / module).read_text(encoding="utf-8")


def _comment_lines(module_source: str) -> list[str]:
    return [
        line.strip()
        for line in module_source.splitlines()
        if line.strip().startswith("#")
    ]


def _comment_text(module_source: str) -> str:
    return " ".join(
        line.removeprefix("#").strip() for line in _comment_lines(module_source)
    )


def _imported_names(source: str) -> list[str]:
    names = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names += [f"{node.module}.{alias.name}" for alias in node.names]
    return names
