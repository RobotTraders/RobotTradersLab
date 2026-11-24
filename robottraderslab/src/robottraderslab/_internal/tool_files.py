from pathlib import Path

_TOOL_DIR_NAME = ".rtlab"


def tool_directory(folder: Path) -> Path:
    """Return the `.rtlab` folder the engine keeps its files in for the folder."""
    return folder / _TOOL_DIR_NAME


def tool_file(owner: Path, suffix: str) -> Path:
    """Return the path of a file kept about `owner` in the `.rtlab` folder beside it.

    Deleting the owner's folder deletes the file with it.

    Args:
        owner: A path naming a file, so that its name carries the suffix being
            replaced.
        suffix: The new suffix, starting with a dot, such as `.lock`.
    """
    resolved = owner.resolve()
    return tool_directory(resolved.parent) / resolved.with_suffix(suffix).name


def scheduler_file(registry: Path, suffix: str) -> Path:
    """Return the path of a file the scheduler keeps about its registry.

    The file sits in the nearest `.rtlab` at or above the registry's folder, or
    in a `.rtlab` in the registry's own folder when none exists. Under it the file sits at the
    registry's path below that folder's parent, so registries named alike in two
    folders of one workspace keep two files.

    Args:
        registry: A path naming the registry file, so that its name carries the
            suffix being replaced.
        suffix: The new suffix, starting with a dot, such as `.lock`.
    """
    resolved = registry.resolve()
    directory = next(
        (
            tool_directory(folder)
            for folder in resolved.parents
            if tool_directory(folder).is_dir()
        ),
        tool_directory(resolved.parent),
    )
    return directory / resolved.relative_to(directory.parent).with_suffix(suffix)


def workspace_root(owner: Path) -> Path:
    """Find the workspace of a bot, the owner's folder when no folder above holds `.rtlab`.

    The workspace is the nearest folder above the owner's own folder that holds
    `.rtlab` and no file named like the owner, since a folder holding one is the
    folder of another bot. When every folder above that holds `.rtlab` holds such
    a file too, the workspace is the topmost of them. The answer is the same
    before and after the owner's first run.
    """
    resolved = owner.resolve()
    folder = resolved.parent
    marked = [
        directory for directory in folder.parents if tool_directory(directory).is_dir()
    ]
    for directory in marked:
        if not (directory / resolved.name).is_file():
            return directory
    return marked[-1] if marked else folder


def mark_root(directory: Path) -> None:
    """Make the directory a workspace root.

    It is the workspace of the bots below it that no nearer `.rtlab` folder claims.
    """
    tool_directory(directory).mkdir(parents=True, exist_ok=True)
