import logging
import os
from pathlib import Path

DEFAULT_LOG_DIR = Path.home() / ".cache" / "april-set" / "logs"

class AprilLogger:
    _logger = None

    @classmethod
    def setup(cls, verbose: bool = False, debug: bool = False):
        if cls._logger is not None:
            return cls._logger

        log_dir = DEFAULT_LOG_DIR
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "april.log"

        logger = logging.getLogger("april")
        logger.setLevel(logging.DEBUG)

        logger.handlers = []

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d) - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        if debug:
            console_handler.setLevel(logging.DEBUG)
        elif verbose:
            console_handler.setLevel(logging.INFO)
        else:
            console_handler.setLevel(logging.WARNING)

        console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        cls._logger = logger
        return logger

    @classmethod
    def get_logger(cls):
        if cls._logger is None:
            return cls.setup()
        return cls._logger
