from collections.abc import Sequence
from itertools import chain
from pathlib import Path

import httpx

from .moves import Move, project_moves
from .notes import render_report
from .releases import (
    Release,
    ReleaseSource,
    fetch_releases,
    github_client,
    releases_between,
    releases_tagged,
)

_PUBLIC_REPOSITORY = "RobotTraders/RobotTradersLab"
_SOURCES = {
    "robottraderslab": (
        ReleaseSource(_PUBLIC_REPOSITORY, "robottraderslab-v"),
        ReleaseSource(_PUBLIC_REPOSITORY, "v"),
    ),
    "robottraderslab-core": (
        ReleaseSource(_PUBLIC_REPOSITORY, "robottraderslab-core-v"),
    ),
    "robottraderslab-impulse": (
        ReleaseSource(_PUBLIC_REPOSITORY, "robottraderslab-impulse-v"),
        ReleaseSource("RobotTraders/RobotTradersLab-Impulse", "v"),
    ),
    "robottraderslab-envelope": (
        ReleaseSource(_PUBLIC_REPOSITORY, "robottraderslab-envelope-v"),
        ReleaseSource("RobotTraders/RobotTradersLab-Envelope", "v"),
    ),
    "robottraderslab-discord-notifications": (
        ReleaseSource(_PUBLIC_REPOSITORY, "robottraderslab-discord-notifications-v"),
        ReleaseSource("RobotTraders/RobotTradersLab-DiscordNotifications", "v"),
    ),
}


def main(project: Path) -> None:
    """GitHub limits a read without a token to 60 requests an hour per address.

    Raises:
        OutdatedError: If the upgrade cannot be planned or GitHub cannot be
            read.
    """
    moves = project_moves(project)
    if moves:
        print(_report(moves))


def _report(moves: Sequence[Move]) -> str:
    with github_client() as client:
        releases_by_package = _releases_by_package(client, moves)
    return render_report(moves, releases_by_package)


def _releases_by_package(
    client: httpx.Client, moves: Sequence[Move]
) -> dict[str, list[Release]]:
    sourced = [move for move in moves if move.package in _SOURCES]
    repositories = dict.fromkeys(
        source.repository
        for source in chain.from_iterable(_SOURCES[move.package] for move in sourced)
    )
    releases_by_repository = {
        repository: fetch_releases(client, repository) for repository in repositories
    }
    return {
        move.package: releases_between(
            chain.from_iterable(
                releases_tagged(
                    releases_by_repository[source.repository], source.tag_prefix
                )
                for source in _SOURCES[move.package]
            ),
            move.installed,
            move.target,
        )
        for move in sourced
    }
