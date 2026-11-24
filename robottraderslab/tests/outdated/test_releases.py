from collections.abc import Iterable, Iterator

import httpx
import pytest

from robottraderslab.outdated import OutdatedError
from robottraderslab.outdated.releases import (
    Release,
    TaggedRelease,
    fetch_releases,
    github_client,
    releases_between,
    releases_tagged,
)

PAGE_SIZE = 100
ENGINE_REPOSITORY = "RobotTraders/RobotTradersLab"
ENGINE_URL = f"https://api.github.com/repos/{ENGINE_REPOSITORY}/releases"


@pytest.fixture
def client() -> Iterator[httpx.Client]:
    with github_client() as opened:
        yield opened


def versions(releases: Iterable[Release]) -> list[str]:
    return [str(release.version) for release in releases]


def entries(count: int) -> list[dict[str, str]]:
    return [{"tag_name": f"v1.0.{index}", "body": "notes"} for index in range(count)]


def test_a_page_of_the_engine_releases_is_read_newest_first(
    respx_mock, engine_releases_page, client
):
    page = respx_mock.get(ENGINE_URL, params={"per_page": "100", "page": "1"}).respond(
        json=engine_releases_page
    )

    releases = fetch_releases(client, ENGINE_REPOSITORY)

    assert page.call_count == 1
    assert [release.tag for release in releases[:3]] == ["v0.10.0", "v0.9.0", "v0.8.0"]
    assert releases[0].body == engine_releases_page[0]["body"]
    assert len(releases) == len(engine_releases_page)


def test_a_full_page_asks_for_the_next_one(respx_mock, client):
    respx_mock.get(ENGINE_URL, params={"per_page": "100", "page": "1"}).respond(
        json=entries(PAGE_SIZE)
    )
    second = respx_mock.get(
        ENGINE_URL, params={"per_page": "100", "page": "2"}
    ).respond(json=entries(1))

    releases = fetch_releases(client, ENGINE_REPOSITORY)

    assert second.call_count == 1
    assert len(releases) == PAGE_SIZE + 1


def test_every_entry_is_read_whatever_its_tag(respx_mock, client):
    respx_mock.get(ENGINE_URL).respond(
        json=[
            {"tag_name": "robottraderslab-impulse-v0.3.0", "body": "plugin"},
            {"tag_name": "v-next", "body": "not a version"},
            {"tag_name": "v0.11.0", "body": "engine"},
        ]
    )

    releases = fetch_releases(client, ENGINE_REPOSITORY)

    assert releases == [
        TaggedRelease("robottraderslab-impulse-v0.3.0", "plugin"),
        TaggedRelease("v-next", "not a version"),
        TaggedRelease("v0.11.0", "engine"),
    ]


def test_a_release_with_no_body_has_an_empty_one(respx_mock, client):
    respx_mock.get(ENGINE_URL).respond(json=[{"tag_name": "v1.0.0", "body": None}])

    releases = fetch_releases(client, ENGINE_REPOSITORY)

    assert releases == [TaggedRelease("v1.0.0", "")]


@pytest.mark.parametrize(
    ("prefix", "dropped_tags", "kept_tag"),
    [
        ("v", ["1.0.0", "v-next", "vv1.0.0"], "v1.0.0"),
        (
            "robottraderslab-impulse-v",
            [
                "robottraderslab-impulse-1.0.0",
                "robottraderslab-impulse-v-next",
                "robottraderslab-impulse-vv1.0.0",
            ],
            "robottraderslab-impulse-v1.0.0",
        ),
    ],
)
def test_a_tag_that_is_not_the_prefix_followed_by_a_version_is_no_release(
    make_release, prefix, dropped_tags, kept_tag
):
    entries = [TaggedRelease(tag, "dropped") for tag in dropped_tags]
    entries.append(TaggedRelease(kept_tag, "kept"))

    releases = releases_tagged(entries, prefix)

    assert releases == [make_release("1.0.0", "kept")]


def test_a_pre_release_tag_is_a_release():
    entries = [TaggedRelease("v0.10.0rc1", "candidate")]

    releases = releases_tagged(entries, "v")

    assert versions(releases) == ["0.10.0rc1"]


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        ("v", ["0.10.0"]),
        ("robottraderslab-v", ["0.11.0"]),
        ("robottraderslab-impulse-v", ["0.3.0"]),
    ],
)
def test_a_tag_is_read_as_a_release_of_its_own_prefix_alone(
    respx_mock, client, prefix, expected
):
    respx_mock.get(ENGINE_URL).respond(
        json=[
            {"tag_name": "v0.10.0", "body": "legacy engine"},
            {"tag_name": "robottraderslab-v0.11.0", "body": "engine"},
            {"tag_name": "robottraderslab-impulse-v0.3.0", "body": "impulse"},
            {"tag_name": "robottraderslab-envelope-v0.4.0", "body": "envelope"},
        ]
    )

    releases = releases_tagged(fetch_releases(client, ENGINE_REPOSITORY), prefix)

    assert versions(releases) == expected


def test_a_refusal_names_the_url_asked(respx_mock, client):
    respx_mock.get(ENGINE_URL).respond(403)

    with pytest.raises(OutdatedError, match=r"releases\?per_page=100&page=1"):
        fetch_releases(client, ENGINE_REPOSITORY)


def test_an_unreachable_github_names_the_url_asked(respx_mock, client):
    respx_mock.get(ENGINE_URL).mock(side_effect=httpx.ConnectError("no route"))

    with pytest.raises(OutdatedError, match="repos/RobotTraders/RobotTradersLab/"):
        fetch_releases(client, ENGINE_REPOSITORY)


def test_an_answer_that_is_not_json_names_the_url_asked(respx_mock, client):
    respx_mock.get(ENGINE_URL).respond(text="<html>maintenance</html>")

    with pytest.raises(OutdatedError, match=r"releases\?per_page=100&page=1.*JSON"):
        fetch_releases(client, ENGINE_REPOSITORY)


def test_an_answer_that_is_not_a_list_names_the_url_asked(respx_mock, client):
    respx_mock.get(ENGINE_URL).respond(json={"message": "Not Found"})

    with pytest.raises(OutdatedError, match=r"releases\?per_page=100&page=1.*list"):
        fetch_releases(client, ENGINE_REPOSITORY)


def test_the_releases_two_apart_come_out_in_ascending_order(make_release):
    releases = [
        make_release(version)
        for version in ["0.10.0", "0.9.0", "0.8.0", "0.7.1", "0.7.0"]
    ]

    between = releases_between(releases, "0.7.1", "0.10.0")

    assert versions(between) == ["0.8.0", "0.9.0", "0.10.0"]


def test_a_release_beyond_the_target_is_left_out(make_release):
    releases = [make_release("0.9.0"), make_release("0.10.0")]

    between = releases_between(releases, "0.8.0", "0.9.0")

    assert versions(between) == ["0.9.0"]


def test_a_pre_release_installed_takes_the_releases_after_it(make_release):
    releases = [make_release("0.10.0rc1"), make_release("0.10.0")]

    between = releases_between(releases, "0.10.0rc1", "0.10.0")

    assert versions(between) == ["0.10.0"]


def test_a_pre_release_target_takes_the_releases_up_to_it(make_release):
    releases = [
        make_release("0.9.0"),
        make_release("0.10.0rc1"),
        make_release("0.10.0"),
    ]

    between = releases_between(releases, "0.9.0", "0.10.0rc1")

    assert versions(between) == ["0.10.0rc1"]
