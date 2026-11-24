---
name: rtlab-onboarding
description: Sets up the strategies and the bots of a RobotTradersLab project whose engine and AI toolkit are installed, for backtesting, for live trading, or both. Offers the open source strategies and installs those the user picks, installs the exchange connector a live project needs, copies the example bots the user picks and backtests them on the user's yes. Sets a live project up with the user, the live sections of each bot, its secrets entry, the key check, the registry of the scheduler and its cron line. Hands a strategy of the user's own to the rtlab-implementing-strategies skill. Use when asked to set up RobotTradersLab, to add an open source strategy or its example bot to a project, or to set a bot up for live trading.
license: Apache-2.0
---

# Setting up a RobotTradersLab project

This skill starts once the engine and the AI toolkit are installed and the agent runs inside the project folder, the one that holds `pyproject.toml`. Every command below runs from that folder.

Run each step for the user and tell them, in a sentence, what it does as you run it. Skip a step already done. Ask each question below and wait for the answer before going on.

Steps 1 to 3 are the same for every project. Step 4 runs for a project that backtests and step 5 for one that trades live; a project that does both runs both.

## 1. Pick what the project is for, and the strategies

Ask the user what the project is for: backtesting, live trading, or both. A project on a VPS is usually there to trade live, and Going live, https://robottraders.io/lab/docs/going-live.html, is the guide step 5 follows.

The engine runs no bot without a strategy. Each strategy is a package of its own. Ask the user which strategies they want. They can pick any of the open source strategies listed at https://robottraders.io/lab/strategies, or none. They can also ask for a strategy of their own.

Install each open source strategy they pick with the command its page gives, such as for Impulse:

```bash
uv add robottraderslab-impulse
```

When the user wants no strategy, stop here and tell them the project is ready.

## 2. Add an exchange connector

A live project needs the connector of the exchange its bots trade on, which places their orders and reads their accounts: ask which exchange, and install its connector with the command its page gives, https://robottraders.io/lab/exchanges lists them, such as for Kraken:

```bash
uv add --index robottraders=https://packages.robottraders.io/simple/ robottraderslab-kraken
```

A project that only backtests needs none: the engine ships a CCXT data adapter, so a bot whose `[backtest.ohlcv_provider]` names `ccxt_<exchange>`, such as the `ccxt_bitget` of Impulse's example bot, downloads that exchange's public candles with no connector and no key. Ask the user whether to add one anyway, and on a no carry on without.

## 3. Copy the example bots

Skip this step when step 1 installed no open source strategy. Each one ships at least one example bot. List them:

```bash
uv run rtlab new --list
```

Ask the user which examples to copy and what to name each bot. When they give no name, use `<example>-bot-example`, so that the copy reads as an example to work from. Copy each one:

```bash
uv run rtlab new <example> <bot>
```

The command copies the example to `workspace/<bot>/`, such as `uv run rtlab new impulse impulse-bot-example`. It refuses a folder that exists; ask the user for another name then. Then tell the user which example bot you copied and give the path of its `workspace/<bot>/bot.toml`, where they change its configuration.

## 4. Backtest them on the user's yes

Skip this step when the project only trades live, or when step 3 copied no bot. Ask the user whether to backtest the bots step 3 copied. On a yes, run and report each as the project instructions' Backtest section says.

## 5. Set the bots up for live trading

Skip this step when the project only backtests. Each part below is a section of Going live; follow that section for what it writes and why.

1. Lay out the templates, `secrets.example.toml` and `registry.example.toml`, at the top of the workspace:

   ```bash
   uv run rtlab init live
   ```

2. In each bot's `bot.toml`, write the two live sections, `[live.ohlcv_provider]` with the connector's name as `ohlcv_provider` and `[live.trading_account]` with a `secret_name` the user picks, the bot's name when they give none.

3. Copy the secrets template:

   ```bash
   cp workspace/secrets.example.toml workspace/secrets.toml
   ```

   The copy is the user's alone, as the project instructions state: never open, read, edit or print `workspace/secrets.toml`, and never ask for a key in the chat. Tell the user the file is in place and that they fill it in themselves, and give them the bot's entry to paste into it: `name`, the `secret_name` of step 2, `exchange`, the connector's name, and the key fields the connector's page lists, https://robottraders.io/lab/exchanges, each left empty, such as for Kraken:

   ```toml
   [[secrets]]
   name = "impulse-example"
   exchange = "kraken"
   api_key = ""
   api_secret = ""
   ```

   Offer the entry for another exchange when the user asks.

4. Once the user says the entry is filled in, offer to check the key with the engine's command, which reads the balances of the account, so a working key prints them and a refused one prints the exchange's answer. Run it on their yes, with a market of the exchange as `--symbol`:

   ```bash
   uv run rtlab check <entry> --symbol <symbol>
   ```

5. Ask the user whether to register the bots with the scheduler, which launches each registered bot at the candle closes of its timeframes. On a yes, copy the registry template and write one `[[bot]]` entry per bot, `config` the bot's folder relative to the registry and `timeframes` the timeframes its profiles trade, read from its `bot.toml`:

   ```bash
   cp workspace/registry.example.toml workspace/registry.toml
   ```

   Then add the line that runs the scheduler every minute to the crontab of the current user, which the command prints:

   ```bash
   uv run rtlab init cron
   ```

   The scheduler page, https://robottraders.io/lab/docs/scheduler.html, lists the registry's other settings.

6. Tell the user that each bot trades from the next candle close of its timeframes, and where to follow it: its log in `workspace/<bot>/logs/live.log` and the scheduler's in `workspace/logs/scheduler.log`.

Whatever the project is for, end by asking the user whether they want anything else.
