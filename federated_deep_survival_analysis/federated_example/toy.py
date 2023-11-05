import pickle
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf

import flwr as fl

from dataset import prepare_dataset
from client import generate_client_fn
from log_config import configure_loguru_logging
from server import get_on_fit_config, get_evaluate_fn


from loguru import logger
import sys


def function_test(x, y):
    result = x + y
    logger.info(f"{result = }")
    return result


def function_prod(x, y):
    result = x * y
    logger.info(f"result = {result}")
    return result


class MyClass:
    def __init__(self, x) -> None:
        self.x = x

    def print_x_squared(self):
        logger.info(f"{self.x**2 = }")


class MyCompositeClass:
    def __init__(self, my_object: MyClass) -> None:
        self.obj = my_object
        
    def init_obj(self, obj: MyClass):
        self.obj = obj


@hydra.main(config_path="config", config_name="toy", version_base=None)
def main(config: DictConfig):
    configure_loguru_logging()

    logger.info(config.bar.more.blabla)

    output = call(config.my_func)
    logger.info(f"{output = }")

    output = call(config.my_func, y=100)

    partial_fn = call(config.my_partial_func)

    output = partial_fn(y=1000)

    logger.info(f"partial {output = }")

    obj: MyClass = instantiate(config.my_object)
    obj.print_x_squared()

    composite_obj: MyCompositeClass = instantiate(config.my_composite_object)
    composite_obj.obj.print_x_squared()
    
    partial_obj: MyCompositeClass = instantiate(config.my_partial_object)
    partial_obj.init_obj(obj)
    partial_obj.obj.print_x_squared()

    model = instantiate(config.toy_model)
    logger.info(model)
if __name__ == "__main__":
    main()
