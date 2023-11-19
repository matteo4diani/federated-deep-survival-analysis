import pickle
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call, instantiate
from omegaconf import DictConfig, OmegaConf
from tempfile import TemporaryDirectory
from flwr.server.utils import tensorboard
import flwr as fl


from federated_deep_survival_analysis.utils.log_config import (
    configure_loguru_logging,
)

from loguru import logger


@hydra.main(
    config_path="config",
    config_name="base",
    version_base=None,
)
def main(config: DictConfig):
    if config.utils.logs.loguru:
        configure_loguru_logging(
            level=config.utils.logs.log_level
        )

    # 1. Parse config & get experiment output dir
    logger.info(
        f"\n{OmegaConf.to_yaml(config, resolve=True)}"
    )

    save_path: str = HydraConfig.get().runtime.output_dir

    # 2. Prepare your dataset
    (
        trainloaders,
        valloaders,
        testloader,
        input_dim,
    ) = call(
        config.dataset_fn,
        num_partitions=config.num_clients,
        server_test_size=config.server_validation_size,
        test_size=config.config_fit.validation_size,
        random_state=config.random_seed,
        split_type=config.data.split_type,
    )

    # 3. Define your clients
    model_fn = call(config.model_fn, input_dim=input_dim)

    client_fn = call(
        config.client_fn,
        trainloaders=trainloaders,
        valloaders=valloaders,
        model_fn=model_fn,
    )

    strategy = tensorboard(logdir=save_path)(
        fl.server.strategy.FedAvg
    )(
        fraction_fit=config.strategy.fraction_fit,
        fraction_evaluate=config.strategy.fraction_evaluate,
        min_fit_clients=config.strategy.min_fit_clients,
        min_evaluate_clients=config.strategy.min_evaluate_clients,
        min_available_clients=config.strategy.min_available_clients,
        on_fit_config_fn=call(config.fit_config_fn),
        evaluate_fn=call(
            config.evaluate_fn,
            testloader=testloader,
            model_fn=model_fn,
        ),
        initial_parameters=call(
            config.init_params, model_fn=model_fn
        ),
        evaluate_metrics_aggregation_fn=call(
            config.metrics_fn,
            metric_names=["test_cic", "test_loss"],
        ),
        fit_metrics_aggregation_fn=call(
            config.metrics_fn,
            metric_names=["train_cic", "train_loss"],
        ),
    )

    # 5. Start Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config.num_clients,
        config=fl.server.ServerConfig(
            num_rounds=config.federated_epochs
        ),
        strategy=strategy,
        client_resources={
            "num_cpus": 2,
            "num_gpus": 0.0,
        },
    )

    # 6. Save your results
    results_path = Path(save_path) / "results.pkl"

    results = {
        "config": OmegaConf.to_container(
            config, resolve=True
        ),
        "history": history,
    }

    with open(str(results_path), "wb") as results_file:
        pickle.dump(
            results,
            results_file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


if __name__ == "__main__":
    main()
