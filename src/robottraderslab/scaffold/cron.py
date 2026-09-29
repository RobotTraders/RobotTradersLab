import os
import shlex
import shutil
import subprocess  # nosec B404
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_CRONTAB = "crontab"
_LIST = "-l"
_FROM_STDIN = "-"
_NO_CRONTAB = "no crontab for"
_EVERY_MINUTE = "* * * * *"
_UV_VARIABLE = "UV"
_UV = "uv"
_SCHEDULER = "scheduler"
_RUN = "run"
_RTLAB = "rtlab"
_CONSOLE_SCRIPTS = frozenset({_RTLAB, "robottraderslab"})
_MODULE_FLAG = "-m"
_MODULE = "robottraderslab"
_DIRECTORY_FLAG = "--directory"
_ASSIGN = "="
_CHANGE_DIRECTORY = "cd"
_COMMENT = "#"
_ALREADY_THERE = "The crontab already holds the scheduler line:"
_ADDED = "Added to the crontab:"
_REPLACED = "Replaced in the crontab:"
_BY = "by:"
_DUPLICATE = "Removed a duplicate scheduler line:"
_OTHERS = "Other scheduler lines left in place:"


class CrontabError(Exception):
    """Raised when the scheduler line cannot be placed in the crontab."""


def main(project: Path, registry: Path) -> None:
    """
    Args:
        project: The folder the cron line runs the scheduler in.
        registry: The registry the scheduler reads, as a path from the project
            folder.

    Raises:
        CrontabError: If cron or uv cannot be found, or the crontab cannot be
            read or written.
    """
    crontab = _crontab_command()
    wanted = _Scheduler(project, project / registry)
    line = _scheduler_line(_uv_path(), project, registry)
    lines = _read_crontab(crontab).splitlines()
    others = _other_schedulers(lines, wanted)
    placement = _place(lines, _standing(lines, wanted), line)
    if placement.changed:
        _write_crontab(crontab, placement.lines)
    print("\n".join(placement.report))
    if others:
        print("\n".join([_OTHERS, *others]))


def _crontab_command() -> str:
    crontab = shutil.which(_CRONTAB)
    if crontab is None:
        raise CrontabError(
            f"`{_CRONTAB}` is not on PATH, so cron is not installed on this machine"
        )
    return crontab


def _uv_path() -> str:
    """uv hands every process it runs the path it was invoked by, which is the
    one cron has to call since cron's PATH does not hold uv's folder.
    """
    uv = os.environ.get(_UV_VARIABLE) or shutil.which(_UV)
    if uv is None:
        raise CrontabError(
            f"`{_UV}` is not on PATH; run the command through `{_UV} run`"
        )
    return Path(uv).as_posix()


def _scheduler_line(uv: str, project: Path, registry: Path) -> str:
    command = [uv, _RUN, _DIRECTORY_FLAG, project.as_posix(), _RTLAB, _SCHEDULER]
    return f"{_EVERY_MINUTE} {shlex.join([*command, registry.as_posix()])}"


def _read_crontab(crontab: str) -> str:
    """A user with no crontab yet is answered on the error stream, so that
    answer reads as an empty crontab.
    """
    listed = subprocess.run(  # nosec B603
        [crontab, _LIST], capture_output=True, text=True
    )
    if listed.returncode == 0:
        return listed.stdout
    if _NO_CRONTAB in listed.stderr:
        return ""
    raise CrontabError(f"cannot read the crontab: {listed.stderr.strip()}")


@dataclass(frozen=True, slots=True)
class _Scheduler:
    project: Path
    registry: Path


@dataclass(frozen=True, slots=True)
class _Placement:
    lines: list[str]
    report: list[str]
    changed: bool


def _other_schedulers(lines: Sequence[str], wanted: _Scheduler) -> list[str]:
    return [
        line
        for line in lines
        if (scheduler := _scheduler_of(line)) is not None and scheduler != wanted
    ]


def _standing(lines: Sequence[str], wanted: _Scheduler) -> list[int]:
    return [index for index, line in enumerate(lines) if _scheduler_of(line) == wanted]


def _scheduler_of(line: str) -> _Scheduler | None:
    """cron starts a line in the user's home folder, so a relative or `~`
    project folder is read from there, and a relative registry from the
    folder the line runs in.
    """
    tokens = _tokens(line)
    registry = _registry_of(tokens)
    project = _project_of(tokens)
    if registry is None or project is None:
        return None
    folder = Path.home() / Path(project).expanduser()
    return _Scheduler(folder, folder / registry)


def _tokens(line: str) -> list[str]:
    """The shell's own operators are words of their own, so `cd … &&` names
    its folder.
    """
    if line.lstrip().startswith(_COMMENT):
        return []
    lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    return list(lexer)


def _registry_of(tokens: Sequence[str]) -> str | None:
    for index in range(1, len(tokens) - 1):
        if tokens[index] == _SCHEDULER and _calls_the_engine(tokens[:index]):
            return tokens[index + 1]
    return None


def _calls_the_engine(before: Sequence[str]) -> bool:
    return Path(before[-1]).name in _CONSOLE_SCRIPTS or list(before[-2:]) == [
        _MODULE_FLAG,
        _MODULE,
    ]


def _project_of(tokens: Sequence[str]) -> str | None:
    for index, token in enumerate(tokens):
        if token.startswith(f"{_DIRECTORY_FLAG}{_ASSIGN}"):
            return token.partition(_ASSIGN)[2]
        if token in (_DIRECTORY_FLAG, _CHANGE_DIRECTORY) and index + 1 < len(tokens):
            return tokens[index + 1]
    return None


def _place(lines: Sequence[str], standing: Sequence[int], line: str) -> _Placement:
    """A second line standing for this project and registry would launch the
    same scheduler twice a minute, so only the first is kept.
    """
    if not standing:
        return _Placement([*lines, line], [_ADDED, line], changed=True)
    first, *duplicates = standing
    kept = lines[first] == line
    report = [_ALREADY_THERE, line] if kept else [_REPLACED, lines[first], _BY, line]
    for index in duplicates:
        report += [_DUPLICATE, lines[index]]
    placed = [
        line if index == first else existing
        for index, existing in enumerate(lines)
        if index not in duplicates
    ]
    return _Placement(placed, report, changed=not kept or bool(duplicates))


def _write_crontab(crontab: str, lines: Sequence[str]) -> None:
    """cron reads a crontab up to its last newline, so the file always ends on
    one.
    """
    written = subprocess.run(  # nosec B603
        [crontab, _FROM_STDIN],
        input="".join(f"{line}\n" for line in lines),
        capture_output=True,
        text=True,
    )
    if written.returncode != 0:
        raise CrontabError(f"cannot write the crontab: {written.stderr.strip()}")
