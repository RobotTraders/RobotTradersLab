import os
import shutil
import subprocess  # nosec B404
from collections.abc import Sequence
from pathlib import Path

_UV_VARIABLE = "UV"
_UV = "uv"


class UvError(Exception):
    pass


def find_executable(name: str, refused: type[Exception], reason: str = "") -> str:
    """
    Args:
        reason: Appended as it is after the name, so it starts with the
            punctuation the sentence needs, such as `, so cron is not installed`.

    Raises:
        refused: If PATH holds no executable of that name.
    """
    path = shutil.which(name)
    if path is None:
        raise refused(f"`{name}` is not on PATH{reason}")
    return path


def run_command(
    command: Sequence[str],
    refused: type[Exception],
    failure: str,
    *,
    cwd: Path | None = None,
    stdin_text: str | None = None,
    capture: bool = False,
    encoding: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Every failure of the command, started or not, is raised as `refused` so the
    caller words one refusal for it.

    Args:
        encoding: Left None, the streams use the locale's encoding, which the
            crontab round trip depends on.

    Raises:
        refused: If the command cannot start or exits non-zero.
    """
    completed = start_command(
        command,
        refused,
        cwd=cwd,
        stdin_text=stdin_text,
        capture=capture,
        encoding=encoding,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() if capture else ""
        raise refused(failure.format(returncode=completed.returncode, stderr=stderr))
    return completed


def start_command(
    command: Sequence[str],
    refused: type[Exception],
    *,
    cwd: Path | None = None,
    stdin_text: str | None = None,
    capture: bool = False,
    encoding: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """
    Raises:
        refused: If the command cannot start.
    """
    try:
        return subprocess.run(  # nosec B603
            command,
            cwd=cwd,
            input=stdin_text,
            capture_output=capture,
            text=True,
            encoding=encoding,
        )
    except OSError as e:
        raise refused(f"cannot run `{command[0]}`: {e}") from e


def uv_executable() -> str:
    """uv hands every process it runs the path it was invoked by, so a command
    run through `uv run` calls the same uv.

    Raises:
        UvError: If neither that path nor PATH names uv.
    """
    return os.environ.get(_UV_VARIABLE) or find_executable(_UV, UvError)


def run_uv(
    arguments: Sequence[str], cwd: Path, *, capture: bool = False
) -> subprocess.CompletedProcess[str]:
    """
    Raises:
        UvError: If uv cannot be found or started, or exits non-zero.
    """
    failure = f"`{_UV} {arguments[0]}` failed with exit code {{returncode}}"
    if capture:
        failure += ": {stderr}"
    return run_command(
        [uv_executable(), *arguments],
        UvError,
        failure,
        cwd=cwd,
        capture=capture,
        encoding="utf-8",
    )
