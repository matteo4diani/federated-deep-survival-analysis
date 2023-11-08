from collections import OrderedDict
from typing import Any, Dict, List, Tuple, Union
from auton_survival import metrics, DeepCoxPH
import numpy as np
from omegaconf import DictConfig
import pandas as pd
import torch
import hydra
import auton_survival
import flwr as fl


class DeepCoxPHClient(fl.client.NumPyClient):
    def __init__(
        self,
        trainloader: pd.DataFrame,
        valloader: pd.DataFrame,
        config: DictConfig,
    ) -> None:
        model_fn = get_model_fn(config)
        self.model = model_fn()
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        pass

    def set_parameters(self, parameters):
        pass

    def fit(self, parameters, config):
        pass

    def evaluate(self, parameters, config):
        pass


def get_client_fn(trainloaders, valloaders, config):
    """Return a function that can be used by the VirtualClientEngine.

    to spawn a FlowerClient with client id `cid`.
    """

    def client_fn(cid: str):
        return DeepCoxPHClient(
            trainloader=trainloaders[int(cid)],
            valloader=valloaders[int(cid)],
            config=config,
        )

    # return the function to spawn client
    return client_fn


def get_model_fn(config):
    def model_fn():
        return DeepCoxPH(layers=config.model.layers)

    return model_fn
