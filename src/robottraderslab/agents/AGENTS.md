# RobotTradersLab

Backtesting and live trading engine for algorithmic trading strategies. The strategy
you trade and the exchange you trade it on are plugins, installed separately.

The CLI is the reference: `rtlab --help` lists the commands, `rtlab <command> --help`
what each takes, and `rtlab new --list` the examples the installed plugins ship.

## Install

Needs uv (https://docs.astral.sh/uv/), which brings Python with it.

```bash
uv sync
```

That installs what the project declares, and the engine runs nothing without a strategy.
Ask the user which one to add: https://robottraders.io/lab/strategies lists them, each
with its install line.

Then ask the user whether to add an exchange connector. A backtest needs none, so carry
on without one if the answer is no. https://robottraders.io/lab/exchanges lists them,
each with its install line.

## Set up a workspace

```bash
uv run rtlab new --list
uv run rtlab new <example> <example>-bot-example
```

The first line lists the examples the installed plugins ship; the second copies one into
`workspace/<example>-bot-example/`, the folder its last argument names. That folder is the
bot: it holds `bot.toml`, the bot configuration file, which declares the strategy and its
parameters, the markets and timeframes it trades, the dates a backtest covers and the
account it trades live. Every command takes the bot's folder, and what the bot produces
lands inside it: `backtests/`, `reports/` and `logs/`. The bot's name, carried by its
orders, its log and the scheduler's lines, is the folder's name. Its keys are documented
on the strategy's and the connector's pages at https://robottraders.io/lab.

## Backtest

```bash
uv run rtlab backtest workspace/<example>-bot-example --no-chart
```

Drop `--no-chart` to open the interactive chart of the run in a browser.

## Write a strategy

Follow the `writing-strategies` skill, which ships with the engine beside this file:
`.claude/skills/writing-strategies/SKILL.md` when `rtlab init claude` set the project up,
`agents/skills/writing-strategies/SKILL.md` when `rtlab init agents` did.

## Live trading

```bash
uv run rtlab live workspace/<example>-bot-example
```

That places real orders on the live account the same `bot.toml` declares, and
spends the money in it. Run it only when the user has asked to trade live.
