import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from dataset import get_dataset
from auton_survival import enable_auton_logger


@hydra.main(config_path="config", config_name="base", version_base=None)
def main(config: DictConfig):
    enable_auton_logger(add_logger=True, capture_warnings=True)

    # 1. Parse config and get experiment output dir
    logger.info("\n{}", OmegaConf.to_yaml(config))

    # 2. Prepare dataset
    trainloaders, validationloaders, testloaders = get_dataset()

    # 3. Define clients

    # 4. Define strategy

    # 5. Start simulation

    # 6. Save results

    pass


if __name__ == "__main__":
    main()
