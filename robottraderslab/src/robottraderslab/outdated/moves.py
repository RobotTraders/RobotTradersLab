import re
from dataclasses import dataclass
from pathlib import Path

from robottraderslab.scaffold import UvError, run_uv

from .errors import OutdatedError

_LOCK_FILE = "uv.lock"
_COMMAND = ["lock", "--upgrade", "--dry-run", "--color", "never"]
_ENGINE_FAMILY = "robottraderslab"
_MOVE_LINE = re.compile(rf"Update ({_ENGINE_FAMILY}\S*) v(\S+) -> v(\S+)")


@dataclass(frozen=True, slots=True)
class Move:
    package: str
    installed: str
    target: str


def project_moves(project: Path) -> list[Move]:
    """The dry run writes nothing, so the project's lock stays as it is.

    Raises:
        OutdatedError: If the folder holds no lock, or uv cannot be found
            or run, or cannot resolve the project.
    """
    if not (project / _LOCK_FILE).is_file():
        raise OutdatedError(
            f"'{project}' holds no `{_LOCK_FILE}`; run the command from the "
            f"project folder that holds it"
        )
    try:
        completed = run_uv(_COMMAND, project, capture=True)
    except UvError as e:
        raise OutdatedError(f"cannot plan the upgrade of '{project}': {e}") from e
    return parse_moves(completed.stderr)


def parse_moves(uv_output: str) -> list[Move]:
    """
    Args:
        uv_output: What `uv lock --upgrade --dry-run` printed.
    """
    matches = (_MOVE_LINE.fullmatch(line.strip()) for line in uv_output.splitlines())
    return [Move(*match.groups()) for match in matches if match is not None]
