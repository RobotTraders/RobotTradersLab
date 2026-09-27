from .formatters import get_formatter_config
from .handler_levels import lowest_handler_level
from .log_level import LogLevel
from .notebook_logging import (
    auto_configure_notebook_logging,
    is_notebook,
    setup_notebook_logging,
)

__all__ = [
    "LogLevel",
    "auto_configure_notebook_logging",
    "get_formatter_config",
    "is_notebook",
    "lowest_handler_level",
    "setup_notebook_logging",
]

# Imported here rather than at the top so processes outside notebooks never
# pay for matplotlib and IPython; both are guaranteed present in a notebook.
if is_notebook():
    import matplotlib
    from IPython import get_ipython

    matplotlib.rcParams["savefig.bbox"] = None
    shell = get_ipython()
    if shell is not None:
        shell.config.InlineBackend.print_figure_kwargs = {"bbox_inches": None}
