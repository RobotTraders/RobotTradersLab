import logging
from collections.abc import Iterator
from unittest.mock import Mock, patch

import pytest

from robottraderslab._core import LogLevel
from robottraderslab.exceptions import StrategyCriticalError
from robottraderslab.scheduler.command import main


@pytest.fixture(autouse=True)
def setup_logging() -> Iterator[Mock]:
    with patch(
        "robottraderslab.scheduler.command.setup_live_logging", autospec=True
    ) as configured:
        yield configured


@pytest.fixture(autouse=True)
def run_scheduler() -> Iterator[Mock]:
    with patch(
        "robottraderslab.scheduler.command.run_scheduler",
        autospec=True,
        return_value=0,
    ) as ran:
        yield ran


def test_logs_rotate_next_to_the_registry_file(setup_logging, create_registry):
    registry_file = create_registry('[[bot]]\nconfig = "bot.toml"\n')

    main(registry_file)

    setup_logging.assert_called_once()
    logging_kwargs = setup_logging.call_args.kwargs
    assert logging_kwargs["config_dir"] == registry_file.parent
    assert logging_kwargs["log_name"] == "scheduler"


def test_the_scheduler_runs_against_the_given_registry(run_scheduler, create_registry):
    registry_file = create_registry('[[bot]]\nconfig = "bot.toml"\n')

    main(registry_file)

    run_scheduler.assert_called_once_with(registry_file)


def test_a_bot_that_failed(run_scheduler, create_registry):
    run_scheduler.return_value = 2
    registry_file = create_registry('[[bot]]\nconfig = "bot.toml"\n')

    failed = main(registry_file)

    assert failed == 2


def test_logging_settings_come_from_the_registry(setup_logging, create_registry):
    registry_file = create_registry(
        '[logging]\nfile_level = "WARNING"\nenable_console = true\nretention_days = 7\n'
    )

    main(registry_file)

    setup_logging.assert_called_once()
    logging_kwargs = setup_logging.call_args.kwargs
    assert logging_kwargs["file_level"] == LogLevel.WARNING
    assert logging_kwargs["enable_console"] is True
    assert logging_kwargs["retention_days"] == 7


def test_a_registry_without_logging_settings_uses_the_defaults(
    setup_logging, create_registry
):
    registry_file = create_registry('[[bot]]\nconfig = "bot.toml"\n')

    main(registry_file)

    setup_logging.assert_called_once()
    logging_kwargs = setup_logging.call_args.kwargs
    assert logging_kwargs["file_level"] == LogLevel.INFO
    assert logging_kwargs["enable_console"] is False
    assert logging_kwargs["enable_file"] is True
    assert logging_kwargs["retention_days"] == 60


def test_custom_handlers_reach_the_logging_setup(
    setup_logging, create_registry, tmp_path
):
    secrets_file = tmp_path / "secrets.toml"
    secrets_file.write_text(
        '[[secrets]]\nname = "scheduler_discord"\nwebhook_url = "https://discord.test/hook"\n'
    )
    registry_file = create_registry(
        f'secrets_file = "{secrets_file.as_posix()}"\n'
        "[notifier.discord.log]\n"
        'secret_name = "scheduler_discord"\n'
        'level = "WARNING"\n'
    )

    with patch(
        "robottraderslab.bootstrap.notifier_loader.load_class", return_value=object
    ):
        main(registry_file)

    setup_logging.assert_called_once()
    custom_handlers = setup_logging.call_args.kwargs["custom_handlers"]
    assert custom_handlers["discord"]["level"] == "WARNING"


def test_a_plugin_setting_coexists_with_the_log_occasion(
    setup_logging, create_registry, tmp_path
):
    secrets_file = tmp_path / "secrets.toml"
    secrets_file.write_text(
        '[[secrets]]\nname = "scheduler_discord"\nwebhook_url = "https://discord.test/hook"\n'
    )
    registry_file = create_registry(
        f'secrets_file = "{secrets_file.as_posix()}"\n'
        "[notifier.discord]\n"
        "yellow = 0xF1C40F\n"
        "[notifier.discord.log]\n"
        'secret_name = "scheduler_discord"\n'
        'level = "WARNING"\n'
    )

    with patch(
        "robottraderslab.bootstrap.notifier_loader.load_class", return_value=object
    ):
        main(registry_file)

    setup_logging.assert_called_once()
    custom_handlers = setup_logging.call_args.kwargs["custom_handlers"]
    assert custom_handlers["discord"]["level"] == "WARNING"


def test_webhook_resolves_from_the_secrets_file(
    setup_logging, create_registry, tmp_path
):
    secrets_file = tmp_path / "secrets.toml"
    secrets_file.write_text(
        '[[secrets]]\nname = "scheduler_discord"\nwebhook_url = "https://secret.test"\n'
    )
    registry_file = create_registry(
        f'secrets_file = "{secrets_file.as_posix()}"\n'
        "[notifier.discord.log]\n"
        'secret_name = "scheduler_discord"\n'
    )

    with patch(
        "robottraderslab.bootstrap.notifier_loader.load_class", return_value=object
    ):
        main(registry_file)

    setup_logging.assert_called_once()
    custom_handlers = setup_logging.call_args.kwargs["custom_handlers"]
    assert custom_handlers["discord"]["webhook_url"] == "https://secret.test"


def test_the_secrets_file_is_found_beside_the_registry(
    setup_logging, create_registry, tmp_path, monkeypatch
):
    secrets_file = tmp_path / "secrets.toml"
    secrets_file.write_text(
        '[[secrets]]\nname = "scheduler_discord"\nwebhook_url = "https://beside"\n'
    )
    registry_file = create_registry(
        'secrets_file = "secrets.toml"\n'
        "[notifier.discord.log]\n"
        'secret_name = "scheduler_discord"\n'
    )
    monkeypatch.chdir(tmp_path.parent)

    with patch(
        "robottraderslab.bootstrap.notifier_loader.load_class", return_value=object
    ):
        main(registry_file)

    setup_logging.assert_called_once()
    custom_handlers = setup_logging.call_args.kwargs["custom_handlers"]
    assert custom_handlers["discord"]["webhook_url"] == "https://beside"


def test_a_malformed_registry(run_scheduler, create_registry):
    registry_file = create_registry("[[bots]\nconfig = 'bot.toml'\n")

    with pytest.raises(StrategyCriticalError, match="cannot start the scheduler"):
        main(registry_file)

    run_scheduler.assert_not_called()


def test_a_malformed_registry_still_configures_logging(setup_logging, create_registry):
    registry_file = create_registry("[[bots]\nconfig = 'bot.toml'\n")

    with pytest.raises(StrategyCriticalError):
        main(registry_file)

    setup_logging.assert_called_once()
    logging_kwargs = setup_logging.call_args.kwargs
    assert logging_kwargs["config_dir"] == registry_file.parent
    assert logging_kwargs["log_name"] == "scheduler"


def test_an_unusable_logging_level(run_scheduler, create_registry):
    registry_file = create_registry('[logging]\nfile_level = "warn"\n')

    with pytest.raises(StrategyCriticalError, match="cannot start the scheduler"):
        main(registry_file)

    run_scheduler.assert_not_called()


def test_a_secret_reference_that_is_not_defined(run_scheduler, create_registry):
    registry_file = create_registry(
        'secrets_file = "absent.toml"\n'
        "[notifier.discord.log]\n"
        'secret_name = "nowhere"\n'
        '[[bot]]\nconfig = "bot.toml"\n'
    )

    main(registry_file)

    run_scheduler.assert_called_once_with(registry_file)


def test_a_notifier_plugin_that_is_not_installed(
    run_scheduler, setup_logging, create_registry, caplog
):
    registry_file = create_registry(
        '[notifier.nosuchplugin.log]\n[[bot]]\nconfig = "bot.toml"\n'
    )

    with caplog.at_level(logging.ERROR):
        main(registry_file)

    run_scheduler.assert_called_once_with(registry_file)
    assert setup_logging.call_args.kwargs["custom_handlers"] == {}
    assert "Running without the `nosuchplugin` notifier" in caplog.text


def test_a_bare_webhook_url_in_the_registry(run_scheduler, create_registry):
    registry_file = create_registry(
        "[notifier.discord.log]\n"
        'webhook_url = "https://discord.test/hook"\n'
        '[[bot]]\nconfig = "bot.toml"\n'
    )

    with pytest.raises(StrategyCriticalError, match="notifier.discord.log.webhook_url"):
        main(registry_file)

    run_scheduler.assert_not_called()


def test_a_refused_credential_never_reaches_the_fallback_logging(
    setup_logging, create_registry
):
    registry_file = create_registry(
        "[notifier.discord.log]\n"
        'webhook_url = "https://discord.test/hook"\n'
        '[[bot]]\nconfig = "bot.toml"\n'
    )

    with pytest.raises(StrategyCriticalError):
        main(registry_file)

    setup_logging.assert_called_once()
    assert setup_logging.call_args.kwargs["custom_handlers"] == {}


def test_a_secret_name_only_registry_notifier_reaches_the_scheduler(
    run_scheduler, create_registry, tmp_path
):
    secrets_file = tmp_path / "secrets.toml"
    secrets_file.write_text(
        '[[secrets]]\nname = "scheduler_discord"\nwebhook_url = "https://secret.test"\n'
    )
    registry_file = create_registry(
        f'secrets_file = "{secrets_file.as_posix()}"\n'
        "[notifier.discord.log]\n"
        'secret_name = "scheduler_discord"\n'
        '[[bot]]\nconfig = "bot.toml"\n'
    )

    main(registry_file)

    run_scheduler.assert_called_once_with(registry_file)
