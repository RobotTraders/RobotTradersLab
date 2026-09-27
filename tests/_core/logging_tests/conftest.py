import logging
import logging.config

import pytest


@pytest.fixture(autouse=True)
def _reset_logging():
    yield
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "handlers": {},
            "root": {
                "level": "DEBUG",
                "handlers": [],
            },
        }
    )
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(logger_name)
        logger.handlers = []
        logger.setLevel(logging.NOTSET)
        logger.propagate = True
