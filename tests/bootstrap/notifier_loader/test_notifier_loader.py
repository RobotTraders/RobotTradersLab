import logging
from unittest.mock import AsyncMock, patch

import pytest

from robottraderslab.bootstrap import (
    load_log_handlers,
    load_notifiers,
    report_unavailable_notifiers,
)
from robottraderslab.exceptions import StrategyCriticalError

PLUGIN_FAILURES = pytest.mark.parametrize(
    "failure_class", [ValueError, StrategyCriticalError], ids=["exception", "critical"]
)


def test_load_notifiers_with_empty_config():
    on_fill, on_placement = load_notifiers({}, {})

    assert on_fill == []
    assert on_placement == []


def test_a_fills_occasion_is_subscribed_to_fills():
    mock_notifier = AsyncMock()
    mock_factory = lambda **kwargs: mock_notifier  # noqa: E731

    with patch(
        "robottraderslab.bootstrap.notifier_loader.load_class",
        return_value=mock_factory,
    ):
        on_fill, on_placement = load_notifiers(
            {"discord": {"fills": {"webhook_url": "https://discord.test/hook"}}}, {}
        )

    assert on_fill == [mock_notifier]
    assert on_placement == []


def test_a_placements_occasion_is_subscribed_to_placements():
    class Notifier:
        async def notify_placement(self, placement): ...

    notifier = Notifier()
    mock_factory = lambda **kwargs: notifier  # noqa: E731

    with patch(
        "robottraderslab.bootstrap.notifier_loader.load_class",
        return_value=mock_factory,
    ):
        on_fill, on_placement = load_notifiers(
            {"discord": {"placements": {"webhook_url": "https://discord.test/hook"}}},
            {},
        )

    assert on_fill == []
    assert on_placement == [notifier.notify_placement]


def test_fills_and_placements_construct_separate_notifiers():
    built_kwargs = []

    def factory(**kwargs):
        built_kwargs.append(kwargs)

        class Notifier:
            async def notify_placement(self, placement): ...

        return Notifier()

    with patch(
        "robottraderslab.bootstrap.notifier_loader.load_class", return_value=factory
    ):
        load_notifiers(
            {
                "discord": {
                    "fills": {"webhook_url": "https://fills.test"},
                    "placements": {"webhook_url": "https://placements.test"},
                }
            },
            {},
        )

    assert built_kwargs == [
        {"webhook_url": "https://fills.test"},
        {"webhook_url": "https://placements.test"},
    ]


def test_load_notifiers_resolves_entry_point_by_plugin_name():
    mock_factory = lambda **kwargs: AsyncMock()  # noqa: E731

    with patch(
        "robottraderslab.bootstrap.notifier_loader.load_class",
        return_value=mock_factory,
    ) as mock_load_class:
        load_notifiers(
            {"discord": {"fills": {"webhook_url": "https://url"}}},
            {},
        )

    mock_load_class.assert_called_once_with("discord", "robot_traders_lab.notifiers")


def test_load_notifiers_resolves_webhook_url_from_secret():
    captured_kwargs = {}
    mock_factory = lambda **kwargs: captured_kwargs.update(kwargs)  # noqa: E731

    with patch(
        "robottraderslab.bootstrap.notifier_loader.load_class",
        return_value=mock_factory,
    ):
        load_notifiers(
            {"discord": {"fills": {"secret_name": "discord_main"}}},
            {"discord_main": {"webhook_url": "https://from-secret"}},
        )

    assert captured_kwargs == {"webhook_url": "https://from-secret"}


def test_load_notifiers_for_an_unknown_secret(caplog):
    with caplog.at_level(logging.ERROR):
        on_fill, on_placement = load_notifiers(
            {"discord": {"fills": {"secret_name": "discord_main"}}}, {}
        )

    assert on_fill == []
    assert on_placement == []
    assert "Running without the `discord` notifier" in caplog.text


@PLUGIN_FAILURES
def test_a_notifier_whose_plugin_raises_leaves_the_others_loaded(caplog, failure_class):
    telegram_notifier = AsyncMock()

    def load_plugin(plugin_name, group):
        if plugin_name == "discord":
            raise failure_class("DISCORD_GREEN=notahex is not a colour")
        return lambda **kwargs: telegram_notifier

    with (
        patch(
            "robottraderslab.bootstrap.notifier_loader.load_class",
            side_effect=load_plugin,
        ),
        caplog.at_level(logging.ERROR),
    ):
        on_fill, on_placement = load_notifiers(
            {"discord": {"fills": {}}, "telegram": {"fills": {}}}, {}
        )

    assert on_fill == [telegram_notifier]
    assert on_placement == []
    assert "Running without the `discord` notifier" in caplog.text


def test_an_unknown_occasion_names_the_valid_ones(caplog):
    with caplog.at_level(logging.ERROR):
        load_notifiers({"discord": {"typo": {}}}, {})

    assert "fills" in caplog.text
    assert "placements" in caplog.text
    assert "log" in caplog.text


def test_a_placements_notifier_without_notify_placement(caplog):
    mock_factory = lambda **kwargs: object()  # noqa: E731

    with (
        patch(
            "robottraderslab.bootstrap.notifier_loader.load_class",
            return_value=mock_factory,
        ),
        caplog.at_level(logging.ERROR),
    ):
        on_fill, on_placement = load_notifiers({"discord": {"placements": {}}}, {})

    assert on_fill == []
    assert on_placement == []
    assert "notify_placement" in caplog.text


def test_a_sync_notify_placement(caplog):
    class SyncPlacement:
        def notify_placement(self, placement): ...

    mock_factory = lambda **kwargs: SyncPlacement()  # noqa: E731

    with (
        patch(
            "robottraderslab.bootstrap.notifier_loader.load_class",
            return_value=mock_factory,
        ),
        caplog.at_level(logging.ERROR),
    ):
        on_fill, on_placement = load_notifiers({"discord": {"placements": {}}}, {})

    assert on_fill == []
    assert on_placement == []
    assert "notify_placement" in caplog.text


def test_load_notifiers_skips_the_log_occasion():
    with patch(
        "robottraderslab.bootstrap.notifier_loader.load_class"
    ) as mock_load_class:
        on_fill, on_placement = load_notifiers(
            {"discord": {"log": {"webhook_url": "https://discord.test/hook"}}}, {}
        )

    assert on_fill == []
    assert on_placement == []
    mock_load_class.assert_not_called()


def test_load_log_handlers_with_empty_config():
    assert load_log_handlers({}, {}) == ({}, {})


def test_a_log_occasion_resolves_through_the_logging_handler_group():
    class FakeHandlerClass:
        pass

    with patch(
        "robottraderslab.bootstrap.notifier_loader.load_class",
        return_value=FakeHandlerClass,
    ) as mock_load_class:
        handlers_config, _ = load_log_handlers(
            {"discord": {"log": {"webhook_url": "https://discord.test/hook"}}}, {}
        )

    mock_load_class.assert_called_once_with(
        "discord", "robot_traders_lab.logging_handlers"
    )
    assert handlers_config == {
        "discord": {
            "()": FakeHandlerClass,
            "webhook_url": "https://discord.test/hook",
        }
    }


def test_a_log_occasion_forwards_level_formatter_and_filters_untouched():
    class FakeHandlerClass:
        pass

    with patch(
        "robottraderslab.bootstrap.notifier_loader.load_class",
        return_value=FakeHandlerClass,
    ):
        handlers_config, _ = load_log_handlers(
            {
                "discord": {
                    "log": {
                        "webhook_url": "https://discord.test/hook",
                        "level": "INFO",
                        "formatter": "default",
                        "filters": [],
                    }
                }
            },
            {},
        )

    assert handlers_config["discord"]["level"] == "INFO"
    assert handlers_config["discord"]["formatter"] == "default"
    assert handlers_config["discord"]["filters"] == []


def test_load_log_handlers_skips_fills_and_placements():
    with patch(
        "robottraderslab.bootstrap.notifier_loader.load_class"
    ) as mock_load_class:
        handlers_config, _ = load_log_handlers(
            {
                "discord": {
                    "fills": {"webhook_url": "https://fills.test"},
                    "placements": {"webhook_url": "https://placements.test"},
                }
            },
            {},
        )

    assert handlers_config == {}
    mock_load_class.assert_not_called()


def test_load_log_handlers_resolves_webhook_url_from_secret():
    class FakeHandlerClass:
        pass

    with patch(
        "robottraderslab.bootstrap.notifier_loader.load_class",
        return_value=FakeHandlerClass,
    ):
        handlers_config, _ = load_log_handlers(
            {"discord": {"log": {"secret_name": "discord_main"}}},
            {"discord_main": {"webhook_url": "https://from-secret"}},
        )

    assert handlers_config["discord"]["webhook_url"] == "https://from-secret"


@PLUGIN_FAILURES
def test_a_log_handler_whose_plugin_raises_leaves_the_others_resolved(failure_class):
    class FakeHandlerClass:
        pass

    def load_plugin(plugin_name, group):
        if plugin_name == "discord":
            raise failure_class("DISCORD_GREEN=notahex is not a colour")
        return FakeHandlerClass

    with patch(
        "robottraderslab.bootstrap.notifier_loader.load_class",
        side_effect=load_plugin,
    ):
        handlers, unavailable = load_log_handlers(
            {"discord": {"log": {}}, "telegram": {"log": {}}}, {}
        )

    assert list(handlers) == ["telegram"]
    assert "DISCORD_GREEN=notahex" in unavailable["discord"]


def test_load_handlers_for_an_unknown_occasion():
    handlers, unavailable = load_log_handlers({"discord": {"typo": {}}}, {})

    assert handlers == {}
    assert "typo" in unavailable["discord"]


def test_an_absent_notifier_is_reported_once_logging_can_carry_it(caplog):
    with caplog.at_level(logging.ERROR):
        report_unavailable_notifiers({"discord": "no such entry point"})

    assert "Running without the `discord` notifier" in caplog.text
    assert "no such entry point" in caplog.text
