---
name: rtlab-upgrade
description: Upgrades the RobotTradersLab engine and its plugins in a project by applying the Upgrade section of every release since the installed version as one plan, showing the plan first and acting only on the user's yes. Use when asked to upgrade, update or move to a new release of the RobotTradersLab engine, a strategy or a connector, or to run uv sync --upgrade in a RobotTradersLab project.
license: Apache-2.0
---

# Upgrading a RobotTradersLab project

An upgrade moves every package the project holds, the engine, the strategies and the connectors, to the newest versions its constraints allow. A release can rename a command, a file or a key, and a bot running live can lose track of its orders when its name changes. Every release's notes end with an **Upgrade** section that lists what to do before the command, the command, and what to do after it. An upgrade can cross several releases, and this skill reads the Upgrade section of every one of them, makes one plan, applies it to the project, and acts only on the user's yes.

## 1. The notes

```bash
uv run rtlab outdated
```

The command writes nothing. It works out which `robottraderslab` packages would move, and prints the notes of every release between the installed version and the newest one the project's constraints allow. Read what it prints:

- **The table first**, in the shape of `uv pip list --outdated`, one row per package that moves, with its installed version and its latest:

  ```
  Package                   Version  Latest
  ------------------------  -------  ------
  robottraderslab           0.8.0    0.10.0
  robottraderslab-bitget    1.2.0    1.3.0
  ```

- **Then one block per release and one per connector move**, headed `## <package> <version>`, packages in the table's order and a package's releases oldest first. A release's block holds its notes from its **Upgrade** line to the end. A release written before that section existed holds the line `This release carries no Upgrade section; its notes follow.` and its whole notes. A connector's block reads `No release notes on GitHub. See <url>.`: name its version change to the user with that page, since there are no notes to apply. An open source package whose range holds no GitHub release reads `No GitHub release between <installed> and <target>.`: name its version change to the user the same way.
- **No output at all** means nothing moves: tell the user the project is up to date and stop.
- **A line starting `Error:`** is shown to the user as it stands, and the skill stops.

An **Upgrade** section has this shape:

```
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

A release carrying **Upgrade** is applied by its steps. A release without it is read from its prose: map each change to the project by hand, and say in the plan that you did.

## 2. The project

Find what the project holds: every `bot.toml` under `workspace/` (each folder holding one is a bot), the `grid-search.toml` beside any of them, `workspace/registry.toml`, and the scripts and notes in the project. `secrets.toml` is never read aloud or copied anywhere.

Match each Before and After step to what it touches here. A bot is live when the registry lists it or when the user says so.

## 3. The plan

An upgrade that crosses several releases is one plan, never one plan for each release. Merge the steps of every release the command lists:

- Every Before step of every release runs before the upgrade command, and every After step after it, each in release order.
- A step two releases repeat, the same action on the same file or command whatever its wording, is listed once, with the reasons of each release carrying it, at the position the plan's order below gives it: the `rtlab init` rerun is always the refresh step, and any other repeated step stands at the place of the earliest release carrying it.
- The Checks of every release make the one Check step.

Show the plan before anything runs. It opens on one sentence naming the versions the upgrade takes each package from and to, then "Here is what I'll do:", then one numbered list, numbered straight through, in the order the steps run:

1. Each Before step that applies here, with the bots it touches. A flatten names the bot, says it closes the bot's positions on the real account, and says it runs only on the user's yes for that bot.
2. "I'll upgrade" the packages: all of them, or the engine alone when the user asked for that.
3. When the notes carry the `rtlab init` rerun: "I'll refresh the AI instructions in `.claude/`" (or `agents/`), with the reason, that the release changes them, and that the user's own `CLAUDE.md` stays as it is.
4. Each other After step that applies here, naming the files it changes.
5. The Check, as what it proves: "to check" the bots work.

Each step opens on "I'll" and gives its reason when it has one. A step that touches nothing in the project is left out, and so is anything the user has no use for: the version check after the upgrade, which speaks only when it fails, and the commands themselves. A step you cannot place in the project is a question at its number in the list, never a guess. The plan ends on: "Shall I go ahead, or change any step? You can name it by its number."

A worked example, for a project with two bots, `envelope-demo`, live and listed in the registry, and `btc-test`, moving from 0.7.1 to 0.9.0, where the command prints two releases, each with its own Upgrade section:

```
Package          Version  Latest
---------------  -------  ------
robottraderslab  0.7.1    0.9.0

## robottraderslab 0.8.0

**Upgrade**

Before the upgrade:

1. Flatten every live bot whose folder's name will differ from its `name`: `uv run rtlab flatten workspace/<name>.toml --all`. A renamed bot no longer recognises its orders and positions.

The upgrade:

- To upgrade everything: `uv sync --upgrade`

After the upgrade:

1. Move each `workspace/<name>.toml` to `workspace/<name>/bot.toml` and delete its `name` key.
2. In `workspace/registry.toml`, change each `config = "<name>.toml"` to `config = "<name>"`.
3. Rerun `uv run rtlab init claude`, or `uv run rtlab init agents`, whichever set the project up: this release changes `AGENTS.md`.
4. Check: `uv run rtlab backtest workspace/<name> --no-chart` completes for each bot.

## robottraderslab 0.9.0

**Upgrade**

The upgrade:

- To upgrade everything: `uv sync --upgrade`

After the upgrade:

1. Replace `rtlab scaffold` with `rtlab new` in any script or note; the arguments are the same.
2. Rerun `uv run rtlab init claude`, or `uv run rtlab init agents`, whichever set the project up: this release changes the `rtlab-implementing-strategies` skill.
3. Check: `uv run rtlab new --list` prints the installed examples.
```

Both releases rerun `rtlab init`, so the plan names the rerun once, as its refresh step, with both reasons. The two Checks make one step. The plan:

```
This upgrade takes robottraderslab from 0.7.1 to 0.9.0. Here is what I'll do:

1. I'll upgrade the package.

2. I'll refresh the AI instructions in .claude/, since these releases change
   AGENTS.md and the rtlab-implementing-strategies skill. Your own CLAUDE.md
   stays as it is.

3. I'll move workspace/envelope-demo.toml and workspace/btc-test.toml into
   their own folders as bot.toml, and update registry.toml to match.

4. I'll backtest envelope-demo and btc-test, and list the installed examples,
   to check the bots and the new commands work.

Shall I go ahead, or change any step? You can name it by its number.
```

Had envelope-demo's folder name differed from its `name`, the list would open on the flatten:

```
1. I'll flatten envelope-demo, since its name changes in this release and a
   renamed bot no longer recognises its orders. This closes its positions on
   the real account; I'll do it only if you say yes for this bot.

2. I'll upgrade the package.
```

## 4. The run

Run on the user's yes, in this order:

1. Each Before step that flattens a bot runs only on a yes for that bot.
2. The upgrade command, then `uv run rtlab --version` to confirm the engine's new version.
3. The `rtlab init` rerun, once for each kind the project holds: `uv run rtlab init claude` when it holds `.claude/rules/robottraderslab.md`, `uv run rtlab init agents` when it holds `agents/AGENTS.md`, both when it holds both. It replaces this skill with the new version's copy. A file the command refuses is a copy the user edited: show its message and run it again with `--force` only on the user's yes.
4. The other After steps, as the plan listed them.
5. The Check commands. A failing one is named with its output and fixed with the user before going on.

## 5. Report

Tell the user the versions now installed, each file changed, each bot flattened and the result of each check. A bot that ran live before the upgrade goes back live when the scheduler's next run picks it up or when the user restarts it.
