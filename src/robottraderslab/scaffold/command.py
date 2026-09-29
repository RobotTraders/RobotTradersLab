from pathlib import Path

from .examples import copy_example, list_examples


def main(name: str | None, workspace: Path) -> None:
    """Copy a packaged example into the workspace, or list what is installed.

    Args:
        name: The example to copy, None to list the installed ones.

    Raises:
        ExampleError: If the example cannot be copied.
    """
    if name is None:
        installed = list_examples()
        print("\n".join(installed) if installed else "No examples installed.")
        return
    print(f"Example copied to {copy_example(name, workspace)}")
