"""
**logger** module just exposes a ``setup_logger`` function to quickly configure the python logger.
"""
import logging
import os


# WIP - Prateek: TODO: pickup filenames dynamically
# TODO: Improve log_dir path
def setup_logger(log_file_name, level=logging.INFO):
    """Set up a logger with a specific log file name."""
    logger = logging.getLogger(log_file_name)

    if not logger.hasHandlers():
        logger.setLevel(level)

        log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(log_dir, log_file_name)
        file_handler = logging.FileHandler(log_file)

        stream_handler = logging.StreamHandler()

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger
