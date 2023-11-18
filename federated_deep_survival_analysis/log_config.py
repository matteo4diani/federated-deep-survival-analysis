from loguru import logger
import logging
import sys


def setup_loguru_logging(source_id: str):
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            logger_opt = logger.opt(
                exception=record.exc_info, depth=6
            )
            logger_opt.log(
                record.levelname, record.getMessage()
            )

    logging.getLogger(source_id).handlers.clear()
    logging.getLogger(source_id).addHandler(
        InterceptHandler()
    )
    logging.getLogger(source_id).propagate = False


def configure_loguru_logging(
    logging_libs=["flwr", "ray", "tqdm"],
    loguru_libs=["auton_survival"],
    level="INFO",
):
    config = {
        "handlers": [
            {"sink": sys.stdout, "level": level},
        ],
    }

    logger.configure(**config)

    for lib in loguru_libs:
        logger.enable(lib)

    setup_loguru_logging(None)

    for lib in logging_libs:
        setup_loguru_logging(lib)
