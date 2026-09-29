from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import pytest

from robottraderslab import Symbol
from robottraderslab.cli import app
from robottraderslab.exceptions import (
    ExchangeCriticalError,
    ExchangeRecoverableError,
)
from robottraderslab.exchanges import Balance

EXCHANGE_GROUP = "robot_traders_lab.exchanges"
SECRET_CHECK_GROUP = "robot_traders_lab.secret_checks"
VENUE = "stubvenue"
REFUSED_KEY = "refused-key-99"
WEBHOOK = "https://webhooks.example.invalid/99/token"

VENUE_ENTRY = """
[[secrets]]
name = "venue-pilot"
exchange = "stubvenue"
demo_trading = true
api_key = "key-99"
"""

WEBHOOK_ENTRY = f"""
[[secrets]]
name = "trades-channel"
webhook_url = "{WEBHOOK}"
"""


class StubAccount:
    """The account a key opens, answering the balance read."""

    def __init__(self) -> None:
        self.read_symbols: list[Symbol] = []

    async def get_balances(self, symbols: Iterable[Symbol]) -> dict[str, Balance]:
        self.read_symbols.extend(symbols)
        return {
            "USDT": Balance(total=1234.5, locked=34.5),
            "BTC": Balance(total=0.25, locked=0.0),
        }


@pytest.fixture
def workspace(project: Path) -> Path:
    return project / "workspace"


@pytest.fixture
def account() -> StubAccount:
    return StubAccount()


@pytest.fixture
def venue(
    make_registered_adapter: Callable[..., None], account: StubAccount
) -> dict[str, Any]:
    built: dict[str, Any] = {}

    async def create(*, api_key: str, demo_trading: bool = False) -> StubAccount:
        if api_key == REFUSED_KEY:
            raise ExchangeCriticalError("the venue refuses the API key")
        built.update(api_key=api_key, demo_trading=demo_trading)
        return account

    make_registered_adapter(create, VENUE, group=EXCHANGE_GROUP)
    return built


def write_secrets(workspace: Path, text: str, name: str = "secrets.toml") -> None:
    (workspace / name).write_text(text)


def test_a_venue_key_prints_each_balance(cli, workspace, venue):
    write_secrets(workspace, VENUE_ENTRY)

    checked = cli.invoke(app, ["check", "venue-pilot"])

    assert checked.exit_code == 0
    assert checked.output == (
        f"`venue-pilot` is a {VENUE} key, read on the account of BTC/USDT:USDT\n"
        "BTC: total 0.25, locked 0\n"
        "USDT: total 1,234.5, locked 34.5\n"
    )


def test_run_from_a_folder_inside_the_project(cli, workspace, venue, monkeypatch):
    write_secrets(workspace, VENUE_ENTRY)
    monkeypatch.chdir(workspace)

    checked = cli.invoke(app, ["check", "venue-pilot"])

    assert checked.exit_code == 0


def test_the_entry_settings_reach_the_connector(cli, workspace, venue):
    write_secrets(workspace, VENUE_ENTRY)

    cli.invoke(app, ["check", "venue-pilot"])

    assert venue == {"api_key": "key-99", "demo_trading": True}


def test_the_symbol_names_the_account_read(cli, workspace, venue, account):
    write_secrets(workspace, VENUE_ENTRY)

    cli.invoke(app, ["check", "venue-pilot", "--symbol", "ETH/USDC:USDC"])

    assert account.read_symbols == [Symbol.create("ETH/USDC:USDC")]


def test_another_secrets_file_under_the_workspace(cli, workspace, venue):
    write_secrets(workspace, VENUE_ENTRY, name="secrets-demo.toml")

    checked = cli.invoke(
        app, ["check", "venue-pilot", "--secrets", "secrets-demo.toml"]
    )

    assert checked.exit_code == 0


def test_a_key_the_venue_refuses(cli, workspace, venue):
    write_secrets(workspace, VENUE_ENTRY.replace("key-99", REFUSED_KEY))

    refused = cli.invoke(app, ["check", "venue-pilot"])

    assert refused.exit_code == 1
    assert refused.stderr == "Error: the venue refuses the API key\n"


def test_an_entry_the_file_lacks(cli, workspace, venue):
    write_secrets(workspace, VENUE_ENTRY)

    refused = cli.invoke(app, ["check", "kraken-main"])

    assert refused.exit_code == 1
    assert "holds no `[[secrets]]` entry named `kraken-main`" in refused.stderr


def test_an_entry_of_no_known_kind(cli, workspace):
    write_secrets(workspace, '[[secrets]]\nname = "loose"\napi_key = "key-99"\n')

    refused = cli.invoke(app, ["check", "loose"])

    assert refused.exit_code == 1
    assert "names no `exchange` and carries no `webhook_url`" in refused.stderr


def test_a_secrets_file_that_is_not_there(cli, workspace):
    refused = cli.invoke(app, ["check", "venue-pilot"])

    assert refused.exit_code == 1
    assert "there is no secrets file at" in refused.stderr


def test_the_connector_account_check_reports_before_the_balances(
    cli, workspace, venue, make_registered_adapter
):
    async def check(**arguments: Any) -> list[str]:
        return ["Position mode: one-way"]

    make_registered_adapter(check, VENUE, group=SECRET_CHECK_GROUP)
    write_secrets(workspace, VENUE_ENTRY)

    checked = cli.invoke(app, ["check", "venue-pilot"])

    assert checked.output.splitlines()[1:] == [
        "Position mode: one-way",
        "BTC: total 0.25, locked 0",
        "USDT: total 1,234.5, locked 34.5",
    ]


def test_the_connector_account_check_receives_the_entry_and_the_symbol(
    cli, workspace, venue, make_registered_adapter
):
    checks: list[dict[str, Any]] = []

    async def check(**arguments: Any) -> list[str]:
        checks.append(arguments)
        return []

    make_registered_adapter(check, VENUE, group=SECRET_CHECK_GROUP)
    write_secrets(workspace, VENUE_ENTRY)

    cli.invoke(app, ["check", "venue-pilot"])

    assert checks == [
        {
            "entry": "venue-pilot",
            "symbol": Symbol.create("BTC/USDT:USDT"),
            "api_key": "key-99",
            "demo_trading": True,
        }
    ]


def test_a_balance_read_the_venue_rejects(cli, workspace, venue, account):
    async def rejected(symbols: Iterable[Symbol]) -> dict[str, Balance]:
        raise ExchangeRecoverableError("the venue rejects the read")

    account.get_balances = rejected
    write_secrets(workspace, VENUE_ENTRY)

    refused = cli.invoke(app, ["check", "venue-pilot"])

    assert refused.exit_code == 1
    assert refused.stderr == "Error: the venue rejects the read\n"


def test_a_secrets_file_naming_one_entry_twice(cli, workspace):
    write_secrets(workspace, VENUE_ENTRY + VENUE_ENTRY)

    refused = cli.invoke(app, ["check", "venue-pilot"])

    assert refused.exit_code == 1
    assert "Duplicate secret name found: `venue-pilot`" in refused.stderr


def test_a_webhook_goes_to_the_plugin_registered_for_its_field(
    cli, workspace, make_registered_adapter
):
    checks: list[dict[str, Any]] = []

    async def check(**arguments: Any) -> list[str]:
        checks.append(arguments)
        return ["posted one message"]

    make_registered_adapter(check, "webhook_url", group=SECRET_CHECK_GROUP)
    write_secrets(workspace, WEBHOOK_ENTRY)

    checked = cli.invoke(app, ["check", "trades-channel"])

    assert checked.exit_code == 0
    assert checked.output == "posted one message\n"
    assert checks == [{"entry": "trades-channel", "webhook_url": WEBHOOK}]


def test_a_webhook_the_plugin_refuses(cli, workspace, make_registered_adapter):
    async def check(**arguments: Any) -> list[str]:
        raise RuntimeError("the webhook refuses the post")

    make_registered_adapter(check, "webhook_url", group=SECRET_CHECK_GROUP)
    write_secrets(workspace, WEBHOOK_ENTRY)

    refused = cli.invoke(app, ["check", "trades-channel"])

    assert refused.exit_code == 1
    assert refused.stderr == "Error: the webhook refuses the post\n"


@pytest.mark.usefixtures("make_registered_adapter")
def test_a_webhook_with_no_plugin_installed(cli, workspace):
    write_secrets(workspace, WEBHOOK_ENTRY)

    refused = cli.invoke(app, ["check", "trades-channel"])

    assert refused.exit_code == 1
    assert refused.stderr == (
        "Error: `trades-channel` carries a `webhook_url`, and the Discord "
        "notifications plugin, which checks one, is not installed\n"
    )
