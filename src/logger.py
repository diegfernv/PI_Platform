import logging
from colorama import Fore, Style
from constants import COLORS

class CustomFormatter(logging.Formatter):


    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: COLORS.DEBUG + format + COLORS.RESET,
        logging.INFO: COLORS.INFO + format + COLORS.RESET,
        logging.WARNING: COLORS.WARNING + format + COLORS.RESET,
        logging.ERROR: COLORS.ERROR + format + COLORS.RESET,
        logging.CRITICAL: COLORS.CRITICAL + format + COLORS.RESET
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def get_logger(name="app_logger"):
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)

        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(CustomFormatter())

        logger.handlers = [handler]
    return logger
