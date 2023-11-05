from loguru import logger
import logging
import sys


def configure_loguru_logging():
    config = {
        "handlers": [
            {"sink": sys.stdout, "level": "DEBUG"},
        ],
    }

    logger.configure(**config)

    def clear_loggers(source_id: str):
        class InterceptHandler(logging.Handler):
            def emit(self, record):
                logger_opt = logger.opt(exception=record.exc_info, depth=6)
                logger_opt.log(record.levelname, record.getMessage())

        logging.getLogger(source_id).handlers.clear()
        logging.getLogger(source_id).addHandler(InterceptHandler())
        logging.getLogger(source_id).propagate = False

    clear_loggers(None)
    clear_loggers("flwr")
    clear_loggers("ray")
