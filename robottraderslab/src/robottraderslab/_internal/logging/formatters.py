def get_default_format() -> str:
    """Get the default log format string.

    Returns:
        Log format with timestamp, level, and message
    """
    return "%(asctime)s: %(levelname)s: %(message)s"


def get_formatter_config() -> dict:
    """Get the formatter configuration for logging.dictConfig.

    Returns:
        Dictionary with 'default' formatter configuration
    """
    return {
        "default": {
            "format": get_default_format(),
        },
    }
