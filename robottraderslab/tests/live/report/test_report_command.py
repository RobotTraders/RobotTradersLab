from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import ANY, Mock

import pytest
from live_bot_helpers import (
    AN_OLD_MOMENT_NS,
    FAILING_VENUE,
    THREE,
    TWO,
    bot_toml,
    copy_of,
    save,
    save_bot,
    seed_old_copy,
    seeded,
)

from robottraderslab.exceptions import BotCriticalError
from robottraderslab.live.report.command import main
from robottraderslab_core.exchanges import Balance

_NO_LOOKBACK = ""
_MARKER = "held-by-this-file"
_MARKER_LINE = f"# {_MARKER}\n"
_NO_PROFILES = '[strategy]\nstrategy_class = "impulse"\n'
_THE_WINDOW_OPENS = datetime(2025, 12, 25, tzinfo=UTC)


@pytest.fixture
def reporting_venue(venue: Mock) -> Mock:
    venue.get_executions_since.return_value = []
    venue.get_balances.return_value = {"USDT": Balance(locked=0.0, total=1000.0)}
    return venue


def _bot_toml(*sections: str, **options: str) -> str:
    return bot_toml(*sections, provider_keys=_NO_LOOKBACK, **options)


def _saved_page(folder: Path) -> str:
    (page,) = (folder / "reports").glob("*/report.html")
    return page.read_text(encoding="utf-8")


def test_the_page_shows_the_bot_toml_of_the_given_folder(
    reporting_venue, browser_openings, bot_folder
):
    save(bot_folder, _bot_toml(*TWO, tail=_MARKER_LINE))

    main(folder=bot_folder, days=7, save=True)

    assert _MARKER in _saved_page(bot_folder)


def test_the_window_starts_the_requested_days_back(
    reporting_venue, browser_openings, bot_folder
):
    save(bot_folder, _bot_toml(*TWO))

    main(folder=bot_folder, days=7, save=False)

    reporting_venue.get_executions_since.assert_called_once_with(_THE_WINDOW_OPENS, ANY)


def test_an_unsaved_report_is_opened_and_kept_in_no_reports_folder(
    reporting_venue, browser_openings, bot_folder
):
    save(bot_folder, _bot_toml(*TWO))

    main(folder=bot_folder, days=7, save=False)

    assert len(browser_openings) == 1
    assert not (bot_folder / "reports").exists()


def test_a_bot_with_no_config_file_is_refused(reporting_venue, bot_folder):
    with pytest.raises(BotCriticalError, match="There is no `bot.toml`"):
        main(folder=bot_folder, days=7, save=False)

    reporting_venue.get_executions_since.assert_not_called()


def test_the_report_reads_the_working_copy_over_a_different_bot_toml(
    reporting_venue, browser_openings, bot_folder
):
    seeded(bot_folder, _bot_toml(*TWO, tail=_MARKER_LINE))
    save(bot_folder, _bot_toml(*THREE, exchange=FAILING_VENUE))

    main(folder=bot_folder, days=7, save=True)

    assert _MARKER in _saved_page(bot_folder)


def test_a_copy_declaring_no_profile_is_refused_naming_the_copy(
    reporting_venue, bot_folder
):
    seeded(bot_folder, _NO_PROFILES)
    save(bot_folder, _bot_toml(*TWO))

    with pytest.raises(BotCriticalError, match=r"bot\.working\.toml.*profiles"):
        main(folder=bot_folder, days=7, save=False)


def test_the_report_leaves_the_working_copy_untouched(
    reporting_venue, browser_openings, bot_folder
):
    save_bot(bot_folder, *TWO, provider_keys=_NO_LOOKBACK)
    seed_old_copy(bot_folder)
    save(bot_folder, _bot_toml(*THREE))

    main(folder=bot_folder, days=7, save=True)

    assert copy_of(bot_folder).stat().st_mtime_ns == AN_OLD_MOMENT_NS
