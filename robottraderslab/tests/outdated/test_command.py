import pytest

from robottraderslab.outdated.command import main

ENGINE_URL = "https://api.github.com/repos/RobotTraders/RobotTradersLab/releases"
ENVELOPE_URL = (
    "https://api.github.com/repos/RobotTraders/RobotTradersLab-Envelope/releases"
)
IMPULSE_URL = (
    "https://api.github.com/repos/RobotTraders/RobotTradersLab-Impulse/releases"
)
DISCORD_URL = (
    "https://api.github.com/repos/RobotTraders/RobotTradersLab-DiscordNotifications"
    "/releases"
)

UV_DRY_RUN = (
    "Update robottraderslab v0.8.0 -> v0.10.0\n"
    "Update robottraderslab-envelope v0.3.1 -> v0.4.0\n"
    "Update robottraderslab-bitget v1.2.0 -> v1.3.0\n"
)


def test_the_upgrade_lists_each_release_between_the_versions_and_none_outside(
    locked_project,
    make_uv_answer,
    respx_mock,
    engine_releases_page,
    capsys,
    headings_of,
):
    make_uv_answer(UV_DRY_RUN)
    respx_mock.get(ENGINE_URL).respond(json=engine_releases_page)
    respx_mock.get(ENVELOPE_URL).respond(
        json=[
            {"tag_name": "v0.5.0", "body": "**Upgrade**\n\nnewer than the target"},
            {"tag_name": "v0.4.0", "body": "**Upgrade**\n\nenvelope step"},
            {"tag_name": "v0.3.1", "body": "**Upgrade**\n\nalready installed"},
        ]
    )

    main(locked_project)

    printed = capsys.readouterr().out
    headings = headings_of(printed)
    assert headings == [
        "## robottraderslab 0.9.0",
        "## robottraderslab 0.10.0",
        "## robottraderslab-envelope 0.4.0",
        "## robottraderslab-bitget 1.3.0",
    ]


@pytest.mark.parametrize(
    ("package", "prefix", "legacy_url"),
    [
        ("robottraderslab-impulse", "robottraderslab-impulse-v", IMPULSE_URL),
        ("robottraderslab-envelope", "robottraderslab-envelope-v", ENVELOPE_URL),
        (
            "robottraderslab-discord-notifications",
            "robottraderslab-discord-notifications-v",
            DISCORD_URL,
        ),
    ],
)
def test_a_plugin_lists_the_notes_of_the_public_repository_and_of_its_legacy_one(
    locked_project,
    make_uv_answer,
    respx_mock,
    capsys,
    headings_of,
    package,
    prefix,
    legacy_url,
):
    make_uv_answer(f"Update {package} v0.2.0 -> v0.3.0\n")
    respx_mock.get(ENGINE_URL).respond(
        json=[
            {"tag_name": f"{prefix}0.3.0", "body": "**Upgrade**\n\nprefixed step"},
            {"tag_name": "v0.2.5", "body": "**Upgrade**\n\nengine step"},
        ]
    )
    respx_mock.get(legacy_url).respond(
        json=[{"tag_name": "v0.2.1", "body": "**Upgrade**\n\nlegacy step"}]
    )

    main(locked_project)

    printed = capsys.readouterr().out
    headings = headings_of(printed)
    assert headings == [f"## {package} 0.2.1", f"## {package} 0.3.0"]
    assert "legacy step" in printed
    assert "prefixed step" in printed
    assert "engine step" not in printed


@pytest.fixture
def public_release_page() -> list[dict[str, str]]:
    return [
        {"tag_name": "robottraderslab-v0.12.0", "body": "**Upgrade**\n\nengine 0.12"},
        {"tag_name": "robottraderslab-v0.11.0", "body": "**Upgrade**\n\nengine 0.11"},
        {
            "tag_name": "robottraderslab-impulse-v0.3.0",
            "body": "**Upgrade**\n\nimpulse",
        },
        {"tag_name": "v0.10.0", "body": "**Upgrade**\n\nengine 0.10"},
    ]


def test_an_engine_across_the_change_lists_each_release_once_and_no_plugin_release(
    locked_project,
    make_uv_answer,
    respx_mock,
    public_release_page,
    capsys,
    headings_of,
):
    make_uv_answer("Update robottraderslab v0.10.0 -> v0.12.0\n")
    respx_mock.get(ENGINE_URL).respond(json=public_release_page)

    main(locked_project)

    printed = capsys.readouterr().out
    headings = headings_of(printed)
    assert headings == ["## robottraderslab 0.11.0", "## robottraderslab 0.12.0"]
    assert "impulse" not in printed


def test_the_core_lists_its_releases_of_the_public_repository_and_no_engine_release(
    locked_project, make_uv_answer, respx_mock, capsys, headings_of
):
    make_uv_answer(
        "Update robottraderslab v0.10.0 -> v0.12.0\n"
        "Update robottraderslab-core v0.1.0 -> v0.3.0\n"
    )
    respx_mock.get(ENGINE_URL).respond(
        json=[
            {
                "tag_name": "robottraderslab-v0.12.0",
                "body": "**Upgrade**\n\nengine 0.12",
            },
            {
                "tag_name": "robottraderslab-core-v0.3.0",
                "body": "**Upgrade**\n\ncore 0.3",
            },
            {
                "tag_name": "robottraderslab-core-v0.2.0",
                "body": "**Upgrade**\n\ncore 0.2",
            },
            {
                "tag_name": "robottraderslab-v0.2.5",
                "body": "**Upgrade**\n\nengine 0.2.5",
            },
            {"tag_name": "v0.2.6", "body": "**Upgrade**\n\nlegacy engine 0.2.6"},
        ]
    )

    main(locked_project)

    printed = capsys.readouterr().out
    headings = headings_of(printed)
    assert headings == [
        "## robottraderslab 0.12.0",
        "## robottraderslab-core 0.2.0",
        "## robottraderslab-core 0.3.0",
    ]
    assert "engine 0.2.5" not in printed
    assert "legacy engine" not in printed


def test_a_plugin_lists_its_release_of_the_public_repository_with_those_of_its_own(
    locked_project,
    make_uv_answer,
    respx_mock,
    public_release_page,
    capsys,
    headings_of,
):
    make_uv_answer("Update robottraderslab-impulse v0.2.0 -> v0.3.0\n")
    respx_mock.get(ENGINE_URL).respond(json=public_release_page)
    respx_mock.get(IMPULSE_URL).respond(
        json=[{"tag_name": "v0.2.1", "body": "**Upgrade**\n\nown repository step"}]
    )

    main(locked_project)

    headings = headings_of(capsys.readouterr().out)
    assert headings == [
        "## robottraderslab-impulse 0.2.1",
        "## robottraderslab-impulse 0.3.0",
    ]


def test_each_repository_is_read_once_for_the_engine_and_the_three_plugins(
    locked_project,
    make_uv_answer,
    respx_mock,
    engine_releases_page,
    capsys,
    headings_of,
):
    make_uv_answer(
        "Update robottraderslab v0.8.0 -> v0.10.0\n"
        "Update robottraderslab-impulse v0.2.0 -> v0.3.0\n"
        "Update robottraderslab-envelope v0.3.1 -> v0.4.0\n"
        "Update robottraderslab-discord-notifications v1.0.0 -> v1.1.0\n"
    )
    engine = respx_mock.get(ENGINE_URL).respond(json=engine_releases_page)
    impulse = respx_mock.get(IMPULSE_URL).respond(
        json=[{"tag_name": "v0.3.0", "body": "**Upgrade**\n\nimpulse step"}]
    )
    envelope = respx_mock.get(ENVELOPE_URL).respond(
        json=[{"tag_name": "v0.4.0", "body": "**Upgrade**\n\nenvelope step"}]
    )
    discord = respx_mock.get(DISCORD_URL).respond(
        json=[{"tag_name": "v1.1.0", "body": "**Upgrade**\n\ndiscord step"}]
    )

    main(locked_project)

    assert engine.call_count == 1
    assert impulse.call_count == 1
    assert envelope.call_count == 1
    assert discord.call_count == 1
    assert respx_mock.calls.call_count == 4
    printed = capsys.readouterr().out
    headings = headings_of(printed)
    assert headings == [
        "## robottraderslab 0.9.0",
        "## robottraderslab 0.10.0",
        "## robottraderslab-impulse 0.3.0",
        "## robottraderslab-envelope 0.4.0",
        "## robottraderslab-discord-notifications 1.1.0",
    ]


def test_a_move_with_no_release_in_range_prints_its_row_and_no_block(
    locked_project, make_uv_answer, respx_mock, engine_releases_page, capsys
):
    make_uv_answer("Update robottraderslab v0.10.0 -> v0.10.0\n")
    respx_mock.get(ENGINE_URL).respond(json=engine_releases_page)

    main(locked_project)

    assert capsys.readouterr().out == (
        "Package          Version  Latest\n"
        "---------------  -------  ------\n"
        "robottraderslab  0.10.0   0.10.0\n"
    )


def test_a_connector_alone_asks_github_for_nothing(
    locked_project, make_uv_answer, respx_mock, capsys, headings_of
):
    make_uv_answer("Update robottraderslab-bitget v1.2.0 -> v1.3.0\n")

    main(locked_project)

    assert respx_mock.calls.call_count == 0
    printed = capsys.readouterr().out
    headings = headings_of(printed)
    assert headings == ["## robottraderslab-bitget 1.3.0"]


def test_a_project_already_at_the_newest_versions_prints_nothing(
    locked_project, make_uv_answer, respx_mock, capsys
):
    make_uv_answer("Update pandas v2.3.1 -> v2.3.3\n")

    main(locked_project)

    assert respx_mock.calls.call_count == 0
    assert capsys.readouterr().out == ""
