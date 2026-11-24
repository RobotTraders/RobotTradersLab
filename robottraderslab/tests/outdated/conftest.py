import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from robottraderslab.outdated.releases import Release, release_of

type ReleaseFactory = Callable[[str, str], Release]
type HeadingsReader = Callable[[str], list[str]]


@pytest.fixture
def make_release() -> ReleaseFactory:
    def make(version: str, body: str = "") -> Release:
        return release_of(version, body)

    return make


@pytest.fixture
def headings_of() -> HeadingsReader:
    def read(report: str) -> list[str]:
        return [line for line in report.splitlines() if line.startswith("## ")]

    return read


@pytest.fixture
def engine_releases_page() -> list[dict[str, Any]]:
    """The first page GitHub answered for the engine's releases, `tag_name` and
    `body` of each entry, v0.10.0 newest.
    """
    recorded = Path(__file__).parent / "releases_page.json"
    return json.loads(recorded.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
