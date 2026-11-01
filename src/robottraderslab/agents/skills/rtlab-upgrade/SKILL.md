---
name: rtlab-upgrade
description: Upgrades the RobotTradersLab engine and its plugins in a project by applying the Upgrade section of each release's notes, showing the plan first and acting only on the user's yes. Use when asked to upgrade, update or move to a new release of the RobotTradersLab engine, a strategy or a connector, or to run uv sync --upgrade in a RobotTradersLab project.
license: Apache-2.0
---

# Upgrading a RobotTradersLab project

An upgrade moves every package the project holds, the engine, the strategies and the connectors, to the newest versions its constraints allow. A release can rename a command, a file or a key, and a bot running live can lose track of its orders when its name changes. Every release's notes end with an **Upgrade** section that lists what to do before the command, the command, and what to do after it. This skill applies that section to the project, and acts only on the user's yes. Every command below runs from the project folder.

## 1. What moves

```bash
uv pip list
uv lock --upgrade --dry-run
```

The first command gives the installed versions, the lines starting with `robottraderslab`. The second writes nothing and prints one line per package it would add, remove or update, such as `Update robottraderslab v0.8.0 -> v0.9.0`; only the `Update` lines are versions moving. When no `robottraderslab` package moves, tell the user the project is up to date and stop.

## 2. The notes

For each package that moves, read the notes of every release between the installed version and the target. The open source packages publish them as GitHub releases, tagged `v<version>`:

| Package | Releases |
|---|---|
| `robottraderslab` | https://api.github.com/repos/RobotTraders/RobotTradersLab/releases |
| `robottraderslab-impulse` | https://api.github.com/repos/RobotTraders/RobotTradersLab-Impulse/releases |
| `robottraderslab-envelope` | https://api.github.com/repos/RobotTraders/RobotTradersLab-Envelope/releases |
| `robottraderslab-discord-notifications` | https://api.github.com/repos/RobotTraders/RobotTradersLab-DiscordNotifications/releases |

The connectors (`robottraderslab-bitget`, `robottraderslab-kraken`, `robottraderslab-hyperliquid`, `robottraderslab-dukascopy`) publish no notes there: name their version change to the user and point to their page under https://robottraders.io/lab/exchanges.

A release's `body` has this shape:

```
Release X.Y.Z (beta) of the RobotTradersLab engine:

Full changelog: <url>

**New**
- ...

**Fixes**
- ...

**First install**
- `uv add <package>`

---

**Upgrade**

Before the upgrade:

1. Numbered instructions run on the installed version, before the command.

The upgrade:

- To upgrade everything: `uv sync --upgrade`
- To upgrade the engine alone: `uv sync --upgrade-package robottraderslab`, which goes through only if every installed plugin accepts the new version

After the upgrade:

1. Numbered edits, each naming the file, section, key or command and the old and new value.
N. Check: the commands that pass once the steps are done.
```

A release carrying **Upgrade** is applied by its steps. A release without it, written before this shape, is read from its prose: map each change to the project by hand, and say in the plan that you did.

## 3. The project

Find what the project holds: every `bot.toml` under `workspace/` (each folder holding one is a bot), the `grid-search.toml` beside any of them, `workspace/registry.toml`, and the scripts and notes in the project. `secrets.toml` is never read aloud or copied anywhere.

Match each Before and After step to what it touches here. A bot is live when the registry lists it or when the user says so.

## 4. The plan

Show the plan before anything runs. It opens on one sentence naming the versions the upgrade takes each package from and to, then "Here is what I'll do:", then one numbered list, numbered straight through, in the order the steps run:

1. Each Before step that applies here, with the bots it touches. A flatten names the bot, says it closes the bot's positions on the real account, and says it runs only on the user's yes for that bot.
2. "I'll upgrade" the packages: all of them, or the engine alone when the user asked for that.
3. When the notes carry the `rtlab init` rerun: "I'll refresh the AI instructions in `.claude/`" (or `agents/`), with the reason, that the release changes them, and that the user's own `CLAUDE.md` stays as it is.
4. Each other After step that applies here, naming the files it changes.
5. The Check, as what it proves: "to check" the bots work.

Each step opens on "I'll" and gives its reason when it has one. A step that touches nothing in the project is left out, and so is anything the user has no use for: the version check after the upgrade, which speaks only when it fails, and the commands themselves. A step you cannot place in the project is a question at its number in the list, never a guess. The plan ends on: "Shall I go ahead, or change any step? You can name it by its number."

A worked example, for a project with two bots, `envelope-demo`, live and listed in the registry, and `btc-test`, moving from 0.7.1 to 0.9.0, where the two releases' notes together carry the Upgrade steps below:

```
**Upgrade**

Before the upgrade:

1. Flatten every live bot whose folder's name will differ from its `name`: `uv run rtlab flatten workspace/<name>.toml --all`. A renamed bot no longer recognises its orders and positions.

The upgrade:

- To upgrade everything: `uv sync --upgrade`
- To upgrade the engine alone: `uv sync --upgrade-package robottraderslab`, which goes through only if every installed plugin accepts the new version

After the upgrade:

1. Move each `workspace/<name>.toml` to `workspace/<name>/bot.toml` and delete its `name` key.
2. In `workspace/registry.toml`, change each `config = "<name>.toml"` to `config = "<name>"`.
3. Move each `optimisation.toml` into its bot's folder as `grid-search.toml`; delete its `[strategy]`, `[backtest]` and `[live]` sections; name each swept parameter by its key inside its profile.
4. Replace `rtlab scaffold` with `rtlab new` in any script or note; the arguments are the same.
5. Rerun `uv run rtlab init claude`, or `uv run rtlab init agents`, whichever set the project up: this release changes `AGENTS.md` and the `writing-strategies` skill.
6. Check: `uv run rtlab backtest workspace/<name> --no-chart` completes for each bot, `uv run rtlab grid-search` on a bot's folder writes `grid-searches/<time>/results.csv`, and `uv run rtlab new --list` prints the installed examples.
```

The plan:

```
This upgrade takes robottraderslab from 0.7.1 to 0.9.0. Here is what I'll do:

1. I'll upgrade the package.

2. I'll refresh the AI instructions in .claude/, since this release changes
   them. Your own CLAUDE.md stays as it is.

3. I'll move workspace/envelope-demo.toml and workspace/btc-test.toml into
   their own folders as bot.toml, and update registry.toml to match.

4. I'll backtest envelope-demo and btc-test to check both bots work.

Shall I go ahead, or change any step? You can name it by its number.
```

Had envelope-demo's folder name differed from its `name`, the list would open on the flatten:

```
1. I'll flatten envelope-demo, since its name changes in this release and a
   renamed bot no longer recognises its orders. This closes its positions on
   the real account; I'll do it only if you say yes for this bot.

2. I'll upgrade the package.
```

## 5. The run

Run on the user's yes, in this order:

1. Each Before step that flattens a bot runs only on a yes for that bot.
2. The upgrade command, then `uv run rtlab --version` to confirm the engine's new version.
3. The `rtlab init` rerun, once for each kind the project holds: `uv run rtlab init claude` when it holds `.claude/rules/robottraderslab.md`, `uv run rtlab init agents` when it holds `agents/AGENTS.md`, both when it holds both. It replaces this skill with the new version's copy. A file the command refuses is a copy the user edited: show its message and run it again with `--force` only on the user's yes.
4. The other After steps, as the plan listed them.
5. The Check commands. A failing one is named with its output and fixed with the user before going on.

## 6. Report

Tell the user the versions now installed, each file changed, each bot flattened and the result of each check. A bot that ran live before the upgrade goes back live when the scheduler's next run picks it up or when the user restarts it.
