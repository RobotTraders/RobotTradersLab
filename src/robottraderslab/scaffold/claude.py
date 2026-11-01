from pathlib import Path

from .agents import MANIFEST_NAME, copy_agent_files, report_agent_files

_RULES_FILE = Path(".claude", "rules", "robottraderslab.md")
_SKILLS_DIR = Path(".claude", "skills")
_MANIFEST = _SKILLS_DIR / MANIFEST_NAME


def install_for_claude(project: Path, force: bool) -> tuple[list[Path], list[Path]]:
    """Claude loads a rule file whatever instruction files the project
    already holds.

    Raises:
        EditedCopiesError: If an edited copy is found and force is not set.
    """
    return copy_agent_files(
        project,
        project / _RULES_FILE,
        project / _SKILLS_DIR,
        project / _MANIFEST,
        force=force,
    )


def main(project: Path, force: bool) -> None:
    written, removed = install_for_claude(project, force)
    report_agent_files(written, removed)
