# Contributing

RobotTradersLab is licensed under the [Apache License 2.0](LICENSE), and a contribution merged into this repository ships under that licence. Robot Traders OÜ sells products built on this engine alongside the open code, the exchange connectors shipped as compiled wheels under the Robot Traders Software License among them, and the [Contributor License Agreement](CLA.md) grants the Company the rights that let a contribution be part of them. Each contributor signs it once.

## Signing the agreement

The first pull request from a GitHub account triggers the agreement check. It comments on the pull request with the agreement and the sentence to post, and holds a failing check until the author posts that sentence as a comment:

```text
I have read the CLA Document and I hereby sign the CLA
```

The signature is recorded in this repository, on the `cla-signatures` branch, and every later pull request from that account passes the check without being asked. An account whose written agreement with Robot Traders OÜ already covers its work is never asked.

## Checks on a pull request

The CI pipeline runs on every pull request out of draft and checks the packages the changed paths reach: the engine for a path outside `plugins/`, a plugin for a path under its folder, and every plugin for a path under `.github/`. Each package gets the formatter and the linter, the type checker and the tests; the engine also gets a backtest and a live run of the smoke-test bot.

## Issues

Issues on this repository are for three things, each through its own form: a bug in the code, a proposal or a request for a new feature or an improvement to the code, and a mistake or gap in the documentation.

This repository takes the issues of the whole Robot Traders Lab: the engine and every plugin. The bug and feature forms ask which part your issue is about.

If you need help with installing, configuring or running your bot, please look in the [documentation](https://robottraders.io/lab/docs/contents.html) first, or use our [AI toolkit](https://robottraders.io/lab/docs/ai-agents.html), which sets up an AI agent that knows the Robot Traders Lab and can help you with your own setup. If neither answers your question, you are welcome to ask on the [Robot Traders Discord server](https://discord.com/invite/aQJ9PvC2Hz). An issue that asks for help is closed with a pointer there.

Open issues are ranked by thumbs up reactions. To show you want an issue fixed, please add a thumbs up to it instead of a comment.

## Credit

A merged contribution keeps its author in the commit history, and GitHub lists every author on the repository's contributors page.
