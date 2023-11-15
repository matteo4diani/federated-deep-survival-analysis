import pickle
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf

import flwr as fl

from dcph_dataset import prepare_support_dataset
from dcph_client import get_client_fn, get_model_fn
from federated_deep_survival_analysis.log_config import configure_loguru_logging
from dcph_server import get_fit_config_fn, get_evaluate_fn


from loguru import logger
import sys


@hydra.main(config_path="config", config_name="base", version_base=None)
def main(config: DictConfig):
    configure_loguru_logging(level="TRACE")

    # 1. Parse config & get experiment output dir
    logger.info(f"\n{OmegaConf.to_yaml(config)}")

    save_path = HydraConfig.get().runtime.output_dir

    # 2. Prepare your dataset
    trainloaders, valloaders, testloader = prepare_support_dataset(
        config.num_clients, config.batch_size
    )

    # 3. Define your clients
    client_fn = get_client_fn(
        trainloaders=trainloaders,
        valloaders=valloaders,
        config=config,
    )

    # 4. Define your strategy
    strategy = instantiate(
        config.strategy,
        evaluate_fn=get_evaluate_fn(
            model_fn=get_model_fn(config, input_dim=testloader.features.shape[-1]), testloader=testloader
        ),
    )

    # 5. Start Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config.num_clients,
        config=fl.server.ServerConfig(num_rounds=config.federated_epochs),
        strategy=strategy,
        client_resources={
            "num_cpus": 2,
            "num_gpus": 0.0,
        },
    )

    # 6. Save your results
    results_path = Path(save_path) / "results.pkl"

    results = {"history": history, "anythingelse": "here"}

    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
