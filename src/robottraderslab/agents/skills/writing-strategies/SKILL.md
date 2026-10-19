---
name: writing-strategies
description: Writes a trading strategy for the RobotTradersLab engine as a Python package, registers it with the engine and backtests it until the run completes. Use when asked to write, create, build or change a trading strategy for the RobotTradersLab engine or its rtlab command.
license: Apache-2.0
---

# Writing a RobotTradersLab strategy

A strategy is a Python package installed beside the engine. The engine finds it through an entry point, a configuration file (TOML) names it, and `rtlab backtest` runs it. The same class runs in a backtest and live; nothing in it selects the mode. Every command below runs from the bot project, the folder whose environment holds the engine.

## Where the facts are

The installed engine is the authority on every signature, argument and configuration key, since it is the version that runs:

```bash
uv run python -m pydoc robottraderslab.strategies.ProfileStrategy
uv run python -m pydoc robottraderslab.strategies.futures.FuturesAccount
```

The engine ships a complete strategy to read, a moving-average crossover registered as `futures_ma`: `robottraderslab/ma_strategy/futures_ma_strategy.py` in the installed package. The Impulse strategy is a larger one, with an example configuration file of its own: once the user agrees to `uv add robottraderslab-impulse`, its source is under `.venv/` in `robottraderslab_impulse/`, the configuration in its `examples/`.

`robottraderslab.indicators` holds the indicators its `__all__` lists; any other indicator is computed with numpy in the signals phase.

The guides carry the concepts:

- Build a strategy: https://robottraders.io/lab/docs/build-a-strategy.html
- Ship a strategy (the package, the entry point, the editable install): https://robottraders.io/lab/docs/ship-a-strategy.html
- API reference: https://robottraders.io/lab/docs/reference.html
- Data sources (the candle providers a backtest reads): https://robottraders.io/lab/docs/data-sources.html

## Workflow

1. **Strategy package**, created beside the bot project, never inside it:
   ```bash
   uv init --package --python 3.13 ../my-strategy
   ```
   Its `pyproject.toml` declares the engine and two entry points, in place of the `[project.scripts]` placeholder uv writes:
   ```toml
   dependencies = ["robottraderslab"]

   [project.entry-points."robot_traders_lab.strategies"]
   my_strategy = "my_strategy:MyStrategy"

   [project.entry-points."robot_traders_lab.examples"]
   my_strategy = "my_strategy"
   ```
2. **Code and configuration**: the class in `src/my_strategy/__init__.py`, its configuration file in `src/my_strategy/examples/my-strategy-bot-example.toml`. The file's `[strategy]` section names the entry point's key, `strategy_class = "my_strategy"`, and carries the strategy's settings and its `[[strategy.profiles]]`; `[backtest]` and `[backtest.ohlcv_provider]` carry the balance, the fees, the dates and the candle source.
3. **Install it into the bot project**:
   ```bash
   uv add --editable ../my-strategy
   ```
   An edit to a `.py` file reaches the next run as it is; an edit to the package's `pyproject.toml` needs `uv sync --reinstall-package my-strategy`.
4. **Backtest it** with the commands `AGENTS.md` gives under Set up a workspace and Backtest, `my_strategy` being the example: the scaffold copies it to `workspace/my_strategy-bot-example/`.
5. **Fix and rerun step 4 until the backtest completes.** The error names the profile, the key or the slot at fault. The work is done when the backtest completes and its report shows the trades the strategy was meant to take.

## The contract

These rules hold for every strategy. A strategy that breaks one still looks correct, and fails at start-up, on the first candle, or by trading on data it should not see.

1. **Imports come from the public gates only**: `robottraderslab` (`Symbol`, `TimeFrame`), `robottraderslab.strategies`, `robottraderslab.strategies.futures`, `robottraderslab.indicators` and `robottraderslab.exceptions`. `robottraderslab._core` and every other subpackage are internal and change without notice.
2. **The shape is a profile class and a `ProfileStrategy` over it**: the profile is a frozen, keyword-only dataclass extending `Profile`, whose typed fields are built from each `[[strategy.profiles]]` section; a field `sizing: SizingRule` is built from the one key of the section naming a rule, such as `total_balance_ratio = 0.5`. The strategy class is `ProfileStrategy[MyProfile]` and declares `market_type: ClassVar[MarketType] = "futures"` and the annotation `account: FuturesAccount`. Every other `[strategy]` key sets the class attribute of the same name, so settings are typed class attributes with defaults and the strategy writes no `__init__`. A strategy whose logic does not split into profiles implements `StrategyProtocol` instead.
3. **`setup` declares everything read later**: one `requirements.ohlcv.add(symbol, timeframe, lookback=...)` per market, its `lookback` covering the longest indicator, and one `requirements.account.add(self.account, symbols=..., ...)` naming the state the snapshot serves (`positions`, `open_orders`, `balances`, the `sizing` rules). Reading state that was not declared raises `StrategyCriticalError` on the first candle.
4. **Signals are computed once over the whole series**: `ohlcvs.column(...)` returns the series as a numpy array and `ohlcvs.add_column(...)` stores a computed one, under a name carrying `profile.tag` so two profiles never overwrite each other.
5. **Booking reads the current candle and nothing else**: `ohlcvs.current(...)` and `ohlcvs.signal(...)` read the candle the engine stands on and take no index, which is what keeps the future out of reach. A booking slot never indexes a column by position.
6. **Booking is synchronous and performs no I/O**: no `await`, no network, no exchange call, no file read. Account state comes from `account_snapshots.of(self.account)`, and every action goes through `bookkeeper.add(...)`; the engine executes the book after the slot returns.
7. **An order is sized exactly once**: either `quantity=` on the entry or exit call, or the builder's `.size(rule, price, snapshot)` with the profile's `SizingRule`. `risk_ratio` sizes from the stop-loss, so its order carries `.stop_loss(price=...)`.
8. **Protections are restated on every candle**: a position the snapshot reports, and that the candle does not close, gets `self.account.move_stop_loss(symbol, price)` booked, since the venue drops a protection when its position closes and an operator can cancel one.
9. **State a later live cycle needs is rebuilt from the venue**: instance attributes survive from candle to candle inside one run, and a live run ends with its cycle. One profile per symbol reads its position from the snapshot; several profiles on one symbol declare a position tracker in `setup` and tag their orders with `.tag(profile.order_tag)`, since the venue nets them into one position.
10. **A broken configuration raises `StrategyCriticalError`**, imported from `robottraderslab.exceptions`; the strategy never wraps a slot in `except Exception` and never catches an exchange error: the connector classifies it, and the engine retries what is worth retrying.
11. **A profile `tag` is at most 8 bytes** and carries none of `@`, `-`, `/`, `:` or a space.
12. **An order setting goes where the engine names it**: a concept the engine knows is a builder call or its keyword, such as `.limit(price, time_in_force=...)` or `trigger_type=` on `.trigger`, `.stop_loss` and `.take_profit`, whose `DEFAULT` watches the mark price, and the last trade for a take-profit; only a field one venue alone has goes in `.params("bitget", ...)`, addressed to that venue's connector, which refuses a field it sets itself. A setting a run may change is a typed class attribute passed to its call, and a run's `[orders]` section wins over the code.
