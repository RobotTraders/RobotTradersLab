import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import httpx
from packaging.version import VERSION_PATTERN, Version

from .errors import OutdatedError

_API = "https://api.github.com"
_PAGE_SIZE = 100
_TIMEOUT_SECONDS = 30.0
_HEADERS = {"Accept": "application/vnd.github+json"}


@dataclass(frozen=True, slots=True)
class Release:
    """A release of one version, with its notes."""

    version: Version
    body: str


@dataclass(frozen=True, slots=True)
class TaggedRelease:
    """A release entry as GitHub lists it, before any tag is read as a version."""

    tag: str
    body: str


@dataclass(frozen=True, slots=True)
class ReleaseSource:
    """Where a package's releases live: a repository and the prefix of its tags.

    `tag_prefix` carries the `v` that precedes the version: `v` for `v0.10.0`,
    `robottraderslab-impulse-v` for `robottraderslab-impulse-v0.3.0`.
    """

    repository: str
    tag_prefix: str


def github_client() -> httpx.Client:
    """The one client a run opens, shared by every `fetch_releases` call."""
    return httpx.Client(headers=_HEADERS, timeout=_TIMEOUT_SECONDS)


def fetch_releases(client: httpx.Client, repository: str) -> list[TaggedRelease]:
    """GitHub's list of releases takes no filter, so every page is read.

    Returns:
        Every release entry of the repository, whatever its tag, in the order
        GitHub lists them, newest first.

    Raises:
        OutdatedError: If GitHub cannot be reached or answers with an error.
    """
    releases: list[TaggedRelease] = []
    page = 1
    while True:
        entries = _read_page(client, repository, page)
        releases.extend(
            TaggedRelease(entry["tag_name"], entry["body"] or "") for entry in entries
        )
        if len(entries) < _PAGE_SIZE:
            return releases
        page += 1


def releases_tagged(releases: Iterable[TaggedRelease], prefix: str) -> list[Release]:
    """The entries whose tag is `prefix` followed by a PEP 440 version.

    A tag of another prefix, and a tag with a second `v` after the prefix, is no
    release of this package.

    Returns:
        The matching releases, in the order given.
    """
    pattern = re.compile(
        rf"^{re.escape(prefix)}(?![vV])(?:{VERSION_PATTERN})$",
        re.VERBOSE | re.IGNORECASE,
    )
    return [
        release_of(release.tag[len(prefix) :], release.body)
        for release in releases
        if pattern.match(release.tag) is not None
    ]


def release_of(version: str, body: str) -> Release:
    """A release of the given PEP 440 version and notes.

    Raises:
        InvalidVersion: If `version` is not a PEP 440 version.
    """
    return Release(Version(version), body)


def releases_between(
    releases: Iterable[Release], installed: str, target: str
) -> list[Release]:
    """The releases above `installed` and up to `target`, oldest first.

    Raises:
        InvalidVersion: If `installed` or `target` is not a PEP 440 version.
    """
    lowest = Version(installed)
    highest = Version(target)
    inside = [r for r in releases if lowest < r.version <= highest]
    return sorted(inside, key=lambda release: release.version)


def _read_page(
    client: httpx.Client, repository: str, page: int
) -> list[dict[str, Any]]:
    url = f"{_API}/repos/{repository}/releases?per_page={_PAGE_SIZE}&page={page}"
    try:
        response = client.get(url)
        response.raise_for_status()
    except httpx.HTTPError as e:
        raise OutdatedError(f"cannot read the releases at {url}: {e}") from e
    try:
        entries: list[dict[str, Any]] = response.json()
    except ValueError as e:
        raise OutdatedError(f"the answer at {url} is not JSON: {e}") from e
    if not isinstance(entries, list):
        raise OutdatedError(f"the answer at {url} is not a list of releases")
    return entries
