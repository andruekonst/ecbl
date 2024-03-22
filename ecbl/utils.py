import logging
from time import perf_counter
from contextlib import contextmanager


@contextmanager
def log_duration(name, message=None, logger=None):
    """Duration logging context manager.

    Prints the message, then measures duration of `with` block contents execution and logs it.

    Args:
        name: Timer name.
        message: Message to print before.
        logger: Logger instance, if None the default logger will be used.

    """
    if logger is None or isinstance(logger, str):
        logger = logging.getLogger(logger)
    if message:
        logger.info(message)
    logger.info(f'{name} start')
    start = perf_counter()
    yield
    end = perf_counter()
    logger.info(f'{name} finished')
    duration = end - start
    logger.info(f'{name} duration = {duration} s')
