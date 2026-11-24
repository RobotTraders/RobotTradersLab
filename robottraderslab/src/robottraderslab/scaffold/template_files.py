from importlib.resources import files

_TEMPLATES_DIR_NAME = "templates"


def read_template(*path: str) -> str:
    """Return the text of a file shipped under `templates/`, given its path
    below that folder.
    """
    return (
        files(__package__)
        .joinpath(_TEMPLATES_DIR_NAME, *path)
        .read_text(encoding="utf-8")
    )
