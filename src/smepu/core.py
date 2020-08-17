"""Placeholder."""
import logging
import os
import sys
from pathlib import Path
from typing import Union


def is_on_sagemaker() -> bool:
    """Guess whether we are running on SageMaker."""
    return "SM_HOSTS" in os.environ


def mkdir(path: Path, parents=True, exist_ok=True, **kwargs) -> Path:
    """Create a directory."""
    path = pathify(path)
    path.mkdir(parents=parents, exist_ok=exist_ok, **kwargs)
    return path


def setup_opinionated_logger(name: str, level: int = logging.INFO):
    """Configure a very opinionated logger that works on and outside SageMaker.

    On SageMaker (particularly training), root logger may have no handler despite basicConfig(...). Hence, force add
    stdout handler to let all log messages end up at CloudWatch.

    Reason to use stdout on SageMaker: with certain container e.g., xgboost, script mode swallows stderr which means
    nothing shipped to CloudWatch.

    When run outside SageMaker (i.e., from your shell on your workstation), typically the root logger will be configured
    to stderr, hence we don't add anymore handler to stdout (otherwise, double print log messages).
    """
    fmt = "%(asctime)s [%(levelname)s] %(name)s %(message)s"
    datefmt = "[%Y-%m-%d %H:%M:%S]"
    logging.basicConfig(
        level=level, format=fmt, datefmt=datefmt,
    )

    if not logger_has_stdeo(logging.root):
        # Training: if no logging handler to stdout or stderr, setup one to stdout.
        # Reason to use stdout: xgboost script mode swallows stderr.
        print("0000: Root logger has no handler. Likely this is training on SageMaker.")
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        logging.getLogger().addHandler(ch)
        print("1000: added stdout handler to root logger")

    def print_logging_setup(logger):
        """Walkthrough logger hierarchy and print details of each logger.

        Print to stdout to make sure CloudWatch pick it up, regardless of how logger handler is setup.
        """
        lgr = logging.getLogger(name)
        while lgr is not None:
            print("level: {}, name: {}, handlers: {}".format(lgr.level, lgr.name, lgr.handlers))
            lgr = lgr.parent

    logger = logging.getLogger(name)
    logger.setLevel(level)
    print_logging_setup(logger)

    return logger


def pathify(path: Union[str, Path, os.PathLike]) -> Path:
    """Convert path-like argument to ``pathlib.Path``."""
    if isinstance(path, Path):
        return path
    else:
        return Path(path)


def logger_has_stdeo(logger: logging.Logger) -> bool:
    """Check whether logger has stdout or stderr in its handlers."""
    for handler in logger.handlers:
        if hasattr(handler, "stream") and handler.stream in (sys.stdout, sys.stderr):  # type: ignore
            return True
    return False
