import pickle
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf

import flwr as fl

from dataset import prepare_dataset
from client import generate_client_fn
from federated_deep_survival_analysis.log_config import configure_loguru_logging
from server import get_on_fit_config, get_evaluate_fn


from loguru import logger
import sys


@hydra.main(config_path="config", config_name="base", version_base=None)
def main(config: DictConfig):
    if config.loguru:
        configure_loguru_logging()

    ## 1. Parse config & get experiment output dir
    logger.info(f"\n{OmegaConf.to_yaml(config)}")

    save_path = HydraConfig.get().runtime.output_dir

    ## 2. Prepare your dataset
    trainloaders, validationloaders, testloader = prepare_dataset(
        config.num_clients, config.batch_size
    )

    ## 3. Define your clients
    client_fn = generate_client_fn(
        trainloaders, validationloaders, config.model
    )

    ## 4. Define your strategy
    strategy = instantiate(
        config.strategy,
        evaluate_fn=get_evaluate_fn(config.model, testloader),
    )

    ## 5. Start Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config.num_clients,
        config=fl.server.ServerConfig(num_rounds=config.num_rounds),
        strategy=strategy,
        client_resources={
            "num_cpus": 2,
            "num_gpus": 0.0,
        },
    )

    ## 6. Save your results
    results_path = Path(save_path) / "results.pkl"

    results = {"history": history, "anythingelse": "here"}

    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
