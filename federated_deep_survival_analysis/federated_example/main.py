import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from federated_deep_survival_analysis.federated_example.dataset import (
    get_dataset,
)
from auton_survival import enable_auton_logger
import logging


def log(obj: any):
    logger.info("\n{}", obj)


@hydra.main(
    config_path="config",
    config_name="base",
    version_base=None,
)
def main(config: DictConfig):
    enable_auton_logger(
        add_logger=True, capture_warnings=True
    )

    # 1. Parse config and get experiment output dir
    log(OmegaConf.to_yaml(config))

    # 2. Prepare dataset
    (
        train_dataloaders,
        validation_dataloaders,
        test_dataloader,
    ) = get_dataset(
        num_partitions=config.num_clients,
        batch_size=config.batch_size,
    )

    # 3. Define clients

    # 4. Define strategy

    # 5. Start simulation

    # 6. Save results


if __name__ == "__main__":
    main()
