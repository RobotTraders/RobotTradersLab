import logging
import tomllib
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from robottraderslab.exceptions import StrategyCriticalError

type SecretsByName = dict[str, dict[str, Any]]

logger = logging.getLogger(__name__)

SECRET_REFERENCE_KEY = "secret_name"  # nosec B105
_EXCHANGE_KEY = "exchange"


def load_secrets(secrets_file: str | Path | None) -> SecretsByName:
    """Read the credentials a bot keeps out of its config file.

    A copy of the example `rtlab init live` laid out, with no entry in it yet,
    starts a bot with no credentials, the same as a file never written.
    """
    if secrets_file is None:
        return {}

    logger.debug(f"Loading secrets from `{secrets_file}`")

    secrets_path = Path(secrets_file)
    if not secrets_path.exists():
        logger.warning(f"Secrets file {secrets_file} not found.")
        return {}

    with open(secrets_path, "rb") as f:
        raw_secrets = tomllib.load(f)

    if not raw_secrets.get("secrets"):
        logger.debug("No secrets declared in `%s`", secrets_file)
        return {}

    secrets_by_name: dict[str, dict] = {}
    for secret in raw_secrets["secrets"]:
        name = secret.get("name")
        if not name:
            raise StrategyCriticalError(
                "Secret without a name found (missing or empty). "
                "Please ensure all secrets have a unique `name` field."
            )
        secret_copy = {
            key: SecretStr(value)
            if isinstance(value, str) and key != _EXCHANGE_KEY
            else value
            for key, value in secret.items()
            if key != "name"
        }
        if secrets_by_name.setdefault(name, secret_copy) is not secret_copy:
            raise StrategyCriticalError(
                f"Duplicate secret name found: `{name}`. "
                "Please ensure all secrets have a unique `name` field."
            )

    logger.debug("Secrets declared: %s", sorted(secrets_by_name))

    return secrets_by_name


def reveal_secrets(config: dict[str, Any]) -> dict[str, Any]:
    """Read a masked secret back for the plugin being built.

    A secret stays masked everywhere else, so a log line, a report or a
    traceback cannot render one, and this is the only place a value is read.
    """
    return {
        key: value.get_secret_value() if isinstance(value, SecretStr) else value
        for key, value in config.items()
    }


def resolve_secret_references(
    configs: dict[str, dict[str, Any]], secrets_by_name: SecretsByName
) -> dict[str, dict[str, Any]]:
    """Merge referenced secrets into config blocks.

    A block opts in by setting ``secret_name`` to the name of a ``[[secrets]]``
    entry; that entry's fields are merged in and win on conflict. Blocks without
    ``secret_name`` are returned unchanged.

    Args:
        configs: Mapping of block name to its raw config.

    Raises:
        StrategyCriticalError: If a block references a secret name that is not defined.
    """
    return {
        name: _merge_referenced_secret(name, config, secrets_by_name)
        for name, config in configs.items()
    }


def _merge_referenced_secret(
    name: str, config: dict[str, Any], secrets_by_name: SecretsByName
) -> dict[str, Any]:
    secret_name = config.get(SECRET_REFERENCE_KEY)
    if secret_name is None:
        return dict(config)

    if secret_name not in secrets_by_name:
        raise StrategyCriticalError(
            f"`{name}` references secret `{secret_name}`, "
            "which is not defined in the secrets file."
        )

    referenced = {
        key: value for key, value in config.items() if key != SECRET_REFERENCE_KEY
    }
    return {**referenced, **secrets_by_name[secret_name]}
