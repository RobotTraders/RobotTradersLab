import pytest

from robottraderslab.bootstrap import resolve_secret_references
from robottraderslab.exceptions import StrategyCriticalError


def test_block_without_secret_name_is_unchanged():
    configs = {"discord": {"webhook_url": "https://inline"}}

    resolved = resolve_secret_references(configs, {})

    assert resolved == {"discord": {"webhook_url": "https://inline"}}


def test_secret_fields_are_merged_into_block():
    configs = {"discord": {"bot_name": "Bot", "secret_name": "discord_main"}}
    secrets = {"discord_main": {"webhook_url": "https://from-secret"}}

    resolved = resolve_secret_references(configs, secrets)

    assert resolved == {
        "discord": {"bot_name": "Bot", "webhook_url": "https://from-secret"}
    }


def test_secret_value_wins_over_inline_value():
    configs = {"discord": {"webhook_url": "https://inline", "secret_name": "main"}}
    secrets = {"main": {"webhook_url": "https://from-secret"}}

    resolved = resolve_secret_references(configs, secrets)

    assert resolved["discord"]["webhook_url"] == "https://from-secret"


def test_unknown_secret_name_raises_with_block_and_secret_names():
    configs = {"discord": {"secret_name": "missing"}}

    with pytest.raises(StrategyCriticalError, match="`discord`.*`missing`"):
        resolve_secret_references(configs, {})


def test_original_config_is_not_mutated():
    configs = {"discord": {"secret_name": "main"}}
    secrets = {"main": {"webhook_url": "https://from-secret"}}

    resolve_secret_references(configs, secrets)

    assert configs == {"discord": {"secret_name": "main"}}
