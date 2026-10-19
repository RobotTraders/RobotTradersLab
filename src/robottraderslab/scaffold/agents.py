from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path

_SHIPPED = files("robottraderslab") / "agents"
_INSTRUCTIONS = "AGENTS.md"
_SKILLS = "skills"
_AGENTS_DIR = Path("agents")


def lay_out_agent_files(project: Path) -> list[Path]:
    """The folder keeps the layout the engine ships the files in, so any AI
    agent reads them there or takes them into its own layout.
    """
    folder = project / _AGENTS_DIR
    return copy_agent_files(folder / _INSTRUCTIONS, folder / _SKILLS)


def copy_agent_files(instructions: Path, skills: Path) -> list[Path]:
    """A copy already in place is overwritten, so the project carries the
    version of the engine installed.
    """
    return [
        _copy(_SHIPPED / _INSTRUCTIONS, instructions),
        *(
            _copy(file, skills / skill.name / file.name)
            for skill in (_SHIPPED / _SKILLS).iterdir()
            for file in skill.iterdir()
        ),
    ]


def main(project: Path) -> None:
    for path in lay_out_agent_files(project):
        print(f"Wrote {path}")


def _copy(source: Traversable, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return target
