from pathlib import Path

from .examples import copy_example, list_examples

_DESTINATION = Path("workspace")


def main(name: str | None) -> None:
    """Copy a packaged example into the workspace, or list what is installed.

    Raises:
        ExampleError: If the example cannot be copied.
    """
    if name is None:
        installed = list_examples()
        print("\n".join(installed) if installed else "No examples installed.")
        return
    print(f"Example copied to {copy_example(name, _DESTINATION)}")
