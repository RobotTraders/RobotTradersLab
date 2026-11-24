from collections.abc import Iterable
from typing import Any

from robottraderslab.outdated.moves import Move
from robottraderslab.outdated.notes import render_report

NO_SECTION = "This release carries no Upgrade section; its notes follow."


def recorded_body(page: Iterable[dict[str, Any]], version: str) -> str:
    entry = next(e for e in page if e["tag_name"] == f"v{version}")
    body: str = entry["body"]
    return body


def test_a_recorded_release_with_an_upgrade_section_prints_from_its_line(
    engine_releases_page, make_release
):
    moves = [Move("robottraderslab", "0.9.0", "0.10.0")]
    body = recorded_body(engine_releases_page, "0.10.0")
    releases = {"robottraderslab": [make_release("0.10.0", body)]}

    report = render_report(moves, releases)

    assert (
        "## robottraderslab 0.10.0\n\n**Upgrade**\n\nBefore the upgrade:\n\n" in report
    )
    assert "`get_quote` reports the last traded price" not in report


def test_a_recorded_release_without_an_upgrade_section_prints_its_whole_body(
    engine_releases_page, make_release
):
    moves = [Move("robottraderslab", "0.8.0", "0.9.0")]
    body = recorded_body(engine_releases_page, "0.9.0")
    releases = {"robottraderslab": [make_release("0.9.0", body)]}

    report = render_report(moves, releases)

    assert (
        "\n\n## robottraderslab 0.9.0\n\n"
        f"{NO_SECTION}\n\n"
        "Release 0.9.0 (beta) of the RobotTradersLab engine:\n\n"
        "**Before you upgrade**\n"
    ) in report
    assert report.endswith("compare/v0.8.0...v0.9.0")


def test_a_body_with_windows_line_endings_prints_its_upgrade_section_with_unix_ones(
    make_release,
):
    body = "Release 1.0.0\r\n\r\n**New**\r\n\r\n- a thing\r\n\r\n---\r\n\r\n**Upgrade**\r\n\r\n1. Do it.\r\n"
    moves = [Move("robottraderslab", "0.9.0", "1.0.0")]

    report = render_report(moves, {"robottraderslab": [make_release("1.0.0", body)]})

    assert report == (
        "Package          Version  Latest\n"
        "---------------  -------  ------\n"
        "robottraderslab  0.9.0    1.0.0\n\n"
        "## robottraderslab 1.0.0\n\n"
        "**Upgrade**\n\n1. Do it."
    )


def test_the_columns_are_padded_to_their_widest_cell_whatever_the_name_length(
    make_release,
):
    moves = [
        Move("robottraderslab", "0.9.0", "0.10.0"),
        Move("robottraderslab-discord-notifications", "10.0.0", "10.1.0"),
    ]
    releases = {
        "robottraderslab": [make_release("0.10.0")],
        "robottraderslab-discord-notifications": [make_release("10.1.0")],
    }

    report = render_report(moves, releases)

    assert report.splitlines()[:4] == [
        "Package                                Version  Latest",
        "-------------------------------------  -------  ------",
        "robottraderslab                        0.9.0    0.10.0",
        "robottraderslab-discord-notifications  10.0.0   10.1.0",
    ]


def test_the_table_comes_first_then_the_blocks_package_by_package(make_release):
    moves = [
        Move("robottraderslab", "0.9.0", "0.10.0"),
        Move("robottraderslab-envelope", "0.3.1", "0.4.0"),
        Move("robottraderslab-bitget", "1.2.0", "1.3.0"),
    ]
    releases = {
        "robottraderslab": [make_release("0.10.0", "**Upgrade**\n\nengine")],
        "robottraderslab-envelope": [
            make_release("0.4.0", "**Upgrade**\n\nenvelope"),
        ],
    }

    report = render_report(moves, releases)

    assert report.splitlines() == [
        "Package                   Version  Latest",
        "------------------------  -------  ------",
        "robottraderslab           0.9.0    0.10.0",
        "robottraderslab-envelope  0.3.1    0.4.0",
        "robottraderslab-bitget    1.2.0    1.3.0",
        "",
        "## robottraderslab 0.10.0",
        "",
        "**Upgrade**",
        "",
        "engine",
        "",
        "## robottraderslab-envelope 0.4.0",
        "",
        "**Upgrade**",
        "",
        "envelope",
        "",
        "## robottraderslab-bitget 1.3.0",
        "",
        "No release notes on GitHub. See https://robottraders.io/lab/exchanges.",
    ]


def test_a_moving_package_with_no_release_between_its_versions_says_so():
    moves = [Move("robottraderslab", "0.9.0", "0.10.0")]

    report = render_report(moves, {"robottraderslab": []})

    assert report.endswith(
        "## robottraderslab 0.10.0\n\nNo GitHub release between 0.9.0 and 0.10.0."
    )


def test_the_releases_of_a_package_come_out_in_the_order_given(
    make_release, headings_of
):
    moves = [Move("robottraderslab", "0.8.0", "0.10.0")]
    releases = {
        "robottraderslab": [
            make_release("0.10.0", "**Upgrade**\n\nfirst"),
            make_release("0.9.0", "**Upgrade**\n\nsecond"),
        ]
    }

    report = render_report(moves, releases)

    headings = headings_of(report)
    assert headings == ["## robottraderslab 0.10.0", "## robottraderslab 0.9.0"]
