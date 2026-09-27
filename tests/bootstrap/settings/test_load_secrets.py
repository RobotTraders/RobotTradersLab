import json
import logging
from pathlib import Path

import pytest

from robottraderslab.bootstrap import load_secrets
from robottraderslab.exceptions import StrategyCriticalError

_TEST_DIR = Path(__file__).parent


def test_load_secrets():
    secrets = load_secrets(_TEST_DIR / "secrets.toml")

    assert len(secrets) == 2
    assert secrets["secret-1"]["anything"].get_secret_value() == "value-1"
    assert secrets["secret-2"]["anythingelse"].get_secret_value() == "value-2"


def test_a_loaded_secret_rendered_anywhere():
    secrets = load_secrets(_TEST_DIR / "secrets.toml")

    secret = secrets["secret-1"]["anything"]

    assert str(secret) == "**********"
    assert "value-1" not in repr(secrets)
    assert "value-1" not in json.dumps(secrets, default=str)


def test_load_secrets_with_duplicates():
    with pytest.raises(
        StrategyCriticalError, match="Duplicate secret name found: `secret-1`"
    ):
        load_secrets(_TEST_DIR / "secrets_with_duplicates.toml")


def test_load_secret_no_name():
    with pytest.raises(
        StrategyCriticalError,
        match="Secret without a name found \\(missing or empty\\)",
    ):
        load_secrets(_TEST_DIR / "secret_no_name.toml")


def test_the_exchange_a_secret_carries_is_not_masked(tmp_path):
    toml_file = tmp_path / "secrets.toml"
    toml_file.write_text(
        '[[secrets]]\nname = "acct"\nexchange = "bitget"\napi_key = "a-real-key"\n'
    )

    secrets = load_secrets(toml_file)

    assert secrets["acct"]["exchange"] == "bitget"
    assert secrets["acct"]["api_key"].get_secret_value() == "a-real-key"


def test_a_named_but_missing_secrets_file_warns_and_yields_nothing(tmp_path, caplog):
    missing = tmp_path / "absent-secrets.toml"

    with caplog.at_level(logging.WARNING):
        secrets = load_secrets(missing)

    assert secrets == {}
    assert "not found" in caplog.text
