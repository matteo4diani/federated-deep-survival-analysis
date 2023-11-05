import pickle
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import flwr as fl

from dataset import prepare_dataset
from client import generate_client_fn
from log_config import configure_loguru_logging
from server import get_on_fit_config, get_evaluate_fn



from loguru import logger
import sys


# A decorator for Hydra. This tells hydra to by default load the config in config/base.yaml
@hydra.main(config_path="config", config_name="toy", version_base=None)
def main(config: DictConfig):
    configure_loguru_logging()
    ## 1. Parse config & get experiment output dir
    logger.info(f"\n{OmegaConf.to_yaml(config)}")
    logger.info(config.bar.more)
    logger.info(config.bar.more.blabla)
    
    
if __name__ == "__main__":
    main()
