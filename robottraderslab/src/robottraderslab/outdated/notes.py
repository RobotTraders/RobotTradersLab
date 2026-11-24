import re
from collections.abc import Mapping, Sequence

from packaging.version import Version

from .moves import Move
from .releases import Release

_UPGRADE_LINE = re.compile(r"^[ \t]*\*\*Upgrade\*\*[ \t]*$", re.MULTILINE)
_NO_SECTION = "This release carries no Upgrade section; its notes follow."
_NO_RELEASE_NOTES = (
    "No release notes on GitHub. See https://robottraders.io/lab/exchanges."
)
_NO_RELEASE_IN_RANGE = "No GitHub release between {installed} and {target}."
_HEADER = ("Package", "Version", "Latest")
_COLUMN_GAP = "  "


def render_report(
    moves: Sequence[Move], releases_by_package: Mapping[str, Sequence[Release]]
) -> str:
    """A package absent from `releases_by_package` publishes no release notes on
    GitHub, so it gets a block saying so.

    Args:
        releases_by_package: The releases between the installed and the target
            version of each package that publishes notes, by package name.
    """
    blocks = [
        block
        for move in moves
        for block in _blocks(move, releases_by_package.get(move.package))
    ]
    return "\n\n".join([_table(moves), *blocks])


def _blocks(move: Move, releases: Sequence[Release] | None) -> list[str]:
    if releases is None:
        return [_block(move.package, move.target, _NO_RELEASE_NOTES)]
    if not releases and move.installed != move.target:
        absent = _NO_RELEASE_IN_RANGE.format(
            installed=move.installed, target=move.target
        )
        return [_block(move.package, move.target, absent)]
    return [
        _block(move.package, release.version, _release_notes(release.body))
        for release in releases
    ]


def _block(package: str, version: str | Version, text: str) -> str:
    return f"## {package} {version}\n\n{text}"


def _release_notes(body: str) -> str:
    text = body.replace("\r\n", "\n").strip()
    heading = _UPGRADE_LINE.search(text)
    if heading is None:
        return f"{_NO_SECTION}\n\n{text}".strip()
    return text[heading.start() :].lstrip()


def _table(moves: Sequence[Move]) -> str:
    rows = [_HEADER, *((move.package, move.installed, move.target) for move in moves)]
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    rule = tuple("-" * width for width in widths)
    return "\n".join(
        _COLUMN_GAP.join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip()
        for row in (rows[0], rule, *rows[1:])
    )
