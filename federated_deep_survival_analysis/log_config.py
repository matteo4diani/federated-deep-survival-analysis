from loguru import logger
import logging
import sys
import warnings


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
    logging_libs=["flwr", "ray", "tqdm", "auton_survival"],
    level="INFO",
):
    logging.basicConfig(level=level)

    config = {
        "handlers": [
            {"sink": sys.stdout, "level": level},
        ],
    }

    logger.configure(**config)

    setup_loguru_logging(None)

    for lib in logging_libs:
        setup_loguru_logging(lib)

    showwarning_ = warnings.showwarning

    def showwarning(message, *args, **kwargs):
        logger.warning(message)
        showwarning_(message, *args, **kwargs)

    warnings.showwarning = showwarning
