from pathlib import Path

from .agents import copy_agent_files

_RULES_FILE = Path(".claude", "rules", "robottraderslab.md")
_SKILLS_DIR = Path(".claude", "skills")


def install_for_claude(project: Path) -> list[Path]:
    """Claude loads a rule file whatever instruction files the project
    already holds, so the project's own `CLAUDE.md` stays untouched.
    """
    return copy_agent_files(project / _RULES_FILE, project / _SKILLS_DIR)


def main(project: Path) -> None:
    for path in install_for_claude(project):
        print(f"Wrote {path}")
