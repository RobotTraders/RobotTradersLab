from pathlib import Path

from .examples import copy_example, list_examples


def main(name: str | None, folder: Path | None, workspace: Path) -> None:
    """Copy a packaged example into the workspace, or list what is installed.

    Args:
        name: The example to copy; when it or `folder` is None, the installed
            examples are listed.
        folder: The bot's folder the example is copied into, relative to the
            workspace.
        workspace: The workspace the folder is copied into.

    Raises:
        ExampleError: If the example cannot be copied.
    """
    if name is None or folder is None:
        installed = list_examples()
        print("\n".join(installed) if installed else "No examples installed.")
        return
    print(f"Example copied to {copy_example(name, workspace, folder)}")
