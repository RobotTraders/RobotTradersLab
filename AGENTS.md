# RobotTradersLab

Backtesting and live trading engine for algorithmic trading strategies. The strategy
you trade and the exchange you trade it on are plugins, installed separately.

The CLI is the reference: `rtlab --help` lists the commands, `rtlab <command> --help`
what each takes, and `rtlab scaffold --list` the examples the installed plugins ship.

## Install

Needs uv (https://docs.astral.sh/uv/), which brings Python with it.

```bash
uv sync
```

That installs the engine alone, and nothing runs without a strategy. Ask the user which
one to add: https://robottraders.io/lab/strategies lists them, each with its install line.

Then ask the user whether to add an exchange connector. A backtest needs none, so carry
on without one if the answer is no. https://robottraders.io/lab/exchanges lists them,
each with its install line.

## Set up a workspace

```bash
uv run rtlab scaffold --list
uv run rtlab scaffold <example>
```

The first line lists the examples the installed plugins ship; the second copies one into
`workspace/<example>/`. The folder holds the bot's settings file, `impulse-bot-example.toml`
for the Impulse example, which declares the strategy and its parameters, the markets and
timeframes it trades, the dates a backtest covers and the account it trades live. A bot's
log, its lock and the scheduler's lines carry the name of that file. Its keys are documented
on the strategy's and the connector's pages at https://robottraders.io/lab.

## Backtest

```bash
uv run rtlab backtest workspace/<example>/<bot>.toml --no-chart
```

Drop `--no-chart` to open the interactive chart of the run in a browser.

## Live trading

```bash
uv run rtlab live workspace/<example>/<bot>.toml
```

That places real orders on the live account the same settings file declares, and spends
the money in it. Run it only when the user has asked to trade live.
