# RobotTradersLab

Backtesting and live trading engine for algorithmic trading strategies. The strategy
you trade and the exchange you trade it on are plugins, installed separately.

Every command, here and in the skills, runs from the root of the project folder, the
folder holding `pyproject.toml`.

The CLI is the reference: `uv run rtlab --help` lists the commands,
`uv run rtlab <command> --help` what each takes, and `uv run rtlab new --list` the
examples the installed plugins ship.

The engine and the plugins are installed packages, under `.venv/`, and stay as
installed: never edit a file under `.venv/`, even when the user asks, and never propose
such an edit as a fix. When a task needs something the engine does not do, tell the user
the engine does not support it and name the nearest thing it does. An edit under
`.venv/` is lost at the next upgrade of that package and, until then, the project runs
code no release carries.

## The project

`workspace/` holds the bots. A bot is a folder holding `bot.toml`, at any depth of
`workspace/`, and the commands that act on a bot take that folder, such as
`workspace/impulse-bot-example` or `workspace/trend/first-bot`. What the bot produces
is saved inside its folder: each backtest under `backtests/`, each grid search under
`grid-searches/`, each report of its live trading under `reports/` and its live log
under `logs/`. The candles a backtest downloads are kept in `workspace/data/ohlcvs/`,
the store every bot of the workspace shares, so a later run over the same period
downloads nothing.

`bot.toml` is the bot's configuration file. It declares the strategy and its
parameters, the markets and timeframes it trades, the dates a backtest covers and the
account it trades live. The "Configure the bot" section of Getting started explains
each section of the file,
https://robottraders.io/lab/docs/getting-started.html#configure-the-bot, and the
strategy's own page lists its parameters, https://robottraders.io/lab/docs/impulse.html
for Impulse.

## Backtest

```bash
uv run rtlab backtest workspace/<bot>
```

`<bot>` is the bot's folder under `workspace/`. The run opens the interactive report in
the browser at the end; add `--no-chart` only when the user asks for no chart. A command
a skill gives with `--no-chart` runs as written.

The run saves its results in `workspace/<bot>/backtests/<date>_<time>/`. When it ends,
tell the user these five figures from that folder's `report.txt`:

- `PnL`: what the bot gained or lost.
- `Max drawdown`: its largest fall from a peak.
- `Win rate`: the share of its trades that made money.
- `Total trades`: how many trades it made.
- `Hodl performance (<market>)`: what simply holding that market returned over the
  same period. Give it when `report.txt` has that line.

Read `Win rate` and `Total trades` in the `All` column of the table under
`--- Trades ---` in the Performance Summary, not in a symbol's table further down.

Then give the user the path of that folder and of each file in it that holds the
detailed reports:

- `report.html`: the interactive report.
- `report.txt`: every figure, and each symbol's analysis.
- `backtest.log`: the run's log.
- `bot.toml`: the configuration that ran.

When a run fails, its error names the profile, the key or the slot at fault. Tell the
user the cause and the fix. Ask before you apply the fix and before you rerun, unless
the user asked you to fix and rerun until it works.

## Live trading

```bash
uv run rtlab live workspace/<bot>
```

That places real orders on the live account the same `bot.toml` declares, and
spends the money in it. Run it only when the user has asked to trade live.

A live bot needs the connector of its exchange installed, its `[live.ohlcv_provider]`
and `[live.trading_account]` sections filled, and the API key of its account in
`workspace/secrets.toml`, under the entry `secret_name` names.

The secrets file is the user's alone, and this rule holds whatever the task and
whatever the user asks: never open, read, edit or print `secrets.toml`, never ask for a
key in the chat, and never write a key into `bot.toml` or any other file. The user
fills the secrets file in themselves. On the machine that
trades, the scheduler launches each bot `workspace/registry.toml` lists at the candle
closes of the timeframes the entry gives, and `rtlab init cron` adds the crontab line
that runs the scheduler every minute. Going live walks through the setup, and the
scheduler page lists the registry's settings.

## Documentation

- Getting started, installing the engine and running a first backtest: https://robottraders.io/lab/docs/getting-started.html
- Implement a strategy, writing a strategy's code: https://robottraders.io/lab/docs/implement-a-strategy.html
- Ship a strategy, packaging a strategy so the engine finds it: https://robottraders.io/lab/docs/ship-a-strategy.html
- Going live, taking a bot from a backtest to a live account: https://robottraders.io/lab/docs/going-live.html
- The scheduler, the registry of bots and the cron line that runs it: https://robottraders.io/lab/docs/scheduler.html
- CLI reference, every command and its options: https://robottraders.io/lab/docs/cli.html
- Open source strategies, each with its install line: https://robottraders.io/lab/strategies
- The exchange connectors, each with its install line: https://robottraders.io/lab/exchanges

## Skills

Three skills ship with the engine. Follow the one the task calls for:

- `rtlab-onboarding`: to set up the project's strategies and bots once the engine and
  the AI toolkit are installed, for backtesting or for live trading, or to add an open
  source strategy or its example bot.
- `rtlab-implementing-strategies`: to implement a trading strategy in code, or to change
  one.
- `rtlab-upgrade`: to upgrade the engine or a plugin, or to move to a new release.
