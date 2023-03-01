import logging
import logging.handlers
from config.config import LogConfig


LOGGER_NAME = 'abb_search_logger'


def setup_logger():
    config = LogConfig()
    root = logging.getLogger(LOGGER_NAME)
    formatter = "%(levelname)s  %(asctime)s  %(message)s"
    log_file = config.get_loging_file()
    h = logging.FileHandler(log_file, 'a')
    f = logging.Formatter(formatter)
    h.setFormatter(f)
    root.addHandler(h)
    root.setLevel(logging.INFO)
    return h


def remove_handler(h):
    root = logging.getLogger(LOGGER_NAME)
    root.removeHandler(h)


def log_message(message):
    logger = logging.getLogger(LOGGER_NAME)
    logger.info(message)


def log_error(message):
    logger = logging.getLogger(LOGGER_NAME)
    logger.critical(message)

def listener_configure():
    root = logging.getLogger()
    config = LogConfig()
    formatter = config.log_format
    log_file = config.get_loging_file()
    h = logging.handlers.RotatingFileHandler(log_file, 'a',
                                             backupCount=config.backupCount)
    f = logging.Formatter(formatter)
    h.setFormatter(f)
    root.addHandler(h)


def listener_process(queue, configure):
    configure()
    while True:
        try:
            record = queue.get()
            if record is None:
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)
        except Exception:
            pass


def logger_configure(queue):
    h = logging.handlers.QueueHandler(queue)
    root = logging.getLogger()
    root.addHandler(h)
    root.setLevel(logging.INFO)


def task_logger_configure(queue):
    h = logging.handlers.QueueHandler(queue)
    root = logging.getLogger()
    root.addHandler(h)
    root.setLevel(logging.INFO)
