import keyword
import re
from importlib.metadata import entry_points, version
from importlib.util import find_spec
from pathlib import Path
from string import Template

from .report import report_written
from .template_files import read_template
from .uv import UvError, run_uv
from .workspace import mark_workspace

_ENGINE_DISTRIBUTION = "robottraderslab"
_TEMPLATES_DIR = "strategy"
_ENTRY_POINT_GROUP = "robot_traders_lab.strategies"
_PACKAGE_NAME = re.compile(r"[a-z][a-z0-9]*([_-][a-z0-9]+)*")
_STRATEGY_SUFFIX = "Strategy"
_PROFILE_SUFFIX = "Profile"
_ENGINE_CLASS = "ProfileStrategy"
_UV = "uv"
_SOURCE_DIR = "src"
_MAIN_MODULE = "main.py"
_CHART_MODULE = "chart_indicators.py"
_INIT_MODULE = "__init__.py"
_BOT_FILE = "bot.toml"
_GITIGNORE = ".gitignore"
_ADD = "add"
_EDITABLE = "--editable"
_SPACE = " "
_QUOTE = '"'


class StrategyPackageError(Exception):
    pass


def create_strategy_package(folder: Path, workspace: Path) -> list[Path]:
    """The entry point key and the bot's strategy_class are both the module
    name, so the bot resolves the strategy the package registers.

    Args:
        folder: The package's folder, which must not exist yet.
        workspace: The workspace that receives the bot's folder.

    Returns:
        The files written, in the order they were written.

    Raises:
        StrategyPackageError: If the package's folder or the bot's exists, or
            the folder's name cannot be a Python package that the engine can
            register.
    """
    bot_folder = workspace / folder.name
    for existing in (folder, bot_folder):
        if existing.exists():
            raise StrategyPackageError(f"'{existing}' already exists; not overwriting")
    template_values = _template_values(folder.name)
    module_dir = folder / _SOURCE_DIR / template_values["module"]
    written = [
        _write(folder / "pyproject.toml", "pyproject.toml.template", template_values),
        _write(folder / _GITIGNORE, "gitignore", template_values),
        _write_text(module_dir / _INIT_MODULE, ""),
        _write(module_dir / _MAIN_MODULE, "strategy.py.template", template_values),
        _write(
            module_dir / _CHART_MODULE, "chart_indicators.py.template", template_values
        ),
        _write(bot_folder / _BOT_FILE, "bot.toml.template", template_values),
    ]
    mark_workspace(workspace)
    return written


def install_package(folder: Path, project: Path) -> None:
    """Install the package in the project, editable, as `uv add --editable`
    does from the project's folder.

    Raises:
        StrategyPackageError: If uv cannot be found or the install fails; the
            message names the line to run by hand.
    """
    by_hand = f"run `{_UV} {_ADD} {_EDITABLE} {_argument(folder)}` in '{project}'"
    try:
        run_uv([_ADD, _EDITABLE, str(folder)], project)
    except UvError as e:
        raise StrategyPackageError(
            f"{e}; {by_hand}; the package stays at '{folder}'"
        ) from e


def main(folder: Path, project: Path, workspace: Path) -> None:
    """
    Args:
        folder: The package's folder, which must not exist yet, relative to
            the project.
        project: The bot project the strategy is installed in.
        workspace: The workspace that receives the bot's folder.

    Raises:
        StrategyPackageError: If the package cannot be created or installed.
    """
    package = (project / folder).resolve()
    report_written(create_strategy_package(package, workspace))
    install_package(package, project)
    print(f"Installed {package.name} in {project}")


def _template_values(name: str) -> dict[str, str]:
    module = _module_name(name)
    class_name = _class_name(module)
    return {
        "name": name,
        "module": module,
        "class_name": class_name,
        "profile_class": class_name + _PROFILE_SUFFIX,
        "engine_version": version(_ENGINE_DISTRIBUTION),
    }


def _module_name(name: str) -> str:
    if not _PACKAGE_NAME.fullmatch(name) or keyword.iskeyword(name):
        raise StrategyPackageError(
            f"'{name}' cannot be a Python package name: start with a lowercase "
            "letter, then use lowercase letters and digits, joined by single "
            "dashes or underscores, and no Python keyword"
        )
    module = name.replace("-", "_")
    if find_spec(module) is not None:
        raise StrategyPackageError(
            f"'{module}' is already an importable module, so a package of that "
            "name could not be imported; pick another name"
        )
    if module in entry_points(group=_ENTRY_POINT_GROUP).names:
        raise StrategyPackageError(
            f"'{module}' is already registered under `{_ENTRY_POINT_GROUP}` by an "
            "installed package; pick another name"
        )
    return module


def _class_name(module: str) -> str:
    class_name = "".join(part.capitalize() for part in module.split("_"))
    if not class_name.endswith(_STRATEGY_SUFFIX):
        class_name += _STRATEGY_SUFFIX
    if class_name == _ENGINE_CLASS:
        raise StrategyPackageError(
            f"'{module}' would name the class `{_ENGINE_CLASS}`, which the engine's "
            "own class holds; pick another name"
        )
    return class_name


def _write(target: Path, template: str, template_values: dict[str, str]) -> Path:
    return _write_text(
        target,
        Template(read_template(_TEMPLATES_DIR, template)).substitute(template_values),
    )


def _write_text(target: Path, text: str) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def _argument(folder: Path) -> str:
    absolute = str(folder.resolve())
    return f"{_QUOTE}{absolute}{_QUOTE}" if _SPACE in absolute else absolute
