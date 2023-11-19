from collections import OrderedDict
from auton_survival import DeepCoxPH
from omegaconf import DictConfig

import torch

from .dcph_model import test


def get_fit_config_fn(config_fit: DictConfig):
    """Return function that prepares config to send to clients."""

    def fit_config_fn(server_round: int):
        # This function will be executed by the strategy in its
        # `configure_fit()` method.

        # Here we are returning the same config on each round but
        # here you might use the `server_round` input argument to
        # adapt over time these settings so clients. For example, you
        # might want clients to use a different learning rate at later
        # stages in the FL process (e.g. smaller lr after N rounds)

        return {
            "lr": config_fit.lr,
            "momentum": config_fit.momentum,
            "local_epochs": config_fit.local_epochs,
            "patience": config_fit.patience,
            "validation_size": config_fit.validation_size,
            "batch_size": config_fit.batch_size,
            "weight_decay": config_fit.weight_decay,
        }

    return fit_config_fn


def get_evaluate_fn(testloader=None, model_fn=None):
    """Define function for global evaluation on the server."""

    def evaluate_fn(
        server_round: int, parameters, config_
    ):
        model: DeepCoxPH = model_fn()

        params_dict = zip(
            model.torch_module.state_dict().keys(),
            parameters,
        )
        state_dict = OrderedDict(
            {k: torch.Tensor(v) for k, v in params_dict}
        )
        model.torch_module.load_state_dict(
            state_dict, strict=True
        )

        loss, concordance_index = test(model, testloader)

        return loss, {
            "concordance_index": concordance_index[0]
        }

    return evaluate_fn
