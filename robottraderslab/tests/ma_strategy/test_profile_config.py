import logging
from pathlib import Path

from robottraderslab import Symbol
from robottraderslab._internal import sitting_out_symbols
from robottraderslab.futures import FuturesAccount
from robottraderslab.ma_strategy.futures_ma_strategy import FuturesMAStrategy
from robottraderslab.strategies import TradingMode, TradingSystem

BTC = Symbol.create("BTC/USDT:USDT")
ETH = Symbol.create("ETH/USDT:USDT")


def _section(symbol: str, **overrides: bool) -> dict[str, str | int | float | bool]:
    return {
        "symbol": symbol,
        "timeframe": "1d",
        "fast_ma_length": 7,
        "slow_ma_length": 20,
        "available_balance_ratio": 0.25,
        **overrides,
    }


def test_a_section_with_both_direction_flags_sits_out_while_the_other_symbol_trades(
    mock_account: FuturesAccount, caplog
):
    with caplog.at_level(logging.ERROR):
        strategy = FuturesMAStrategy(
            account=mock_account,
            trading_system=TradingSystem(trading_mode=TradingMode.LIVE),
            config_dir=Path("bot-config-dir"),
            profiles=[
                _section("BTC/USDT:USDT"),
                _section("ETH/USDT:USDT", long_only=True, short_only=True),
            ],
        )

    assert [profile.symbol for profile in strategy.profiles] == [BTC]
    assert sitting_out_symbols(strategy) == {ETH}
    (refusal,) = caplog.records
    assert refusal.levelno == logging.ERROR
    assert refusal.getMessage().startswith(
        "FuturesMAStrategy: profile 2 (ETH/USDT:USDT 1d): "
        "long_only and short_only cannot both be true"
    )
