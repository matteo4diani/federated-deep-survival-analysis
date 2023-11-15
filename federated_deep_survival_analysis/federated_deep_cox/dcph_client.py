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
from federated_deep_survival_analysis.federated_deep_cox.dcph_dataset import SurvivalDataset

from federated_deep_survival_analysis.federated_deep_cox.dcph_model import test, train


class DeepCoxPHClient(fl.client.NumPyClient):
    def __init__(
        self,
        trainloader: SurvivalDataset,
        valloader: SurvivalDataset,
        config: DictConfig,
    ) -> None:
        model_fn = get_model_fn(config, input_dim=trainloader.features.shape[-1])
        self.model = model_fn()
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.torch_module.state_dict().items()]


    def set_parameters(self, parameters):
        params_dict = zip(self.model.torch_module.state_dict().keys(), parameters)

        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

        self.model.torch_module.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        train(self.model, self.trainloader, config)
        
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        loss, concordance_index = test(self.model, self.valloader)
        
        return float(loss), len(self.valloader), {"concordance_index": concordance_index}


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


def get_model_fn(config, input_dim):
    def model_fn():
        model = DeepCoxPH(layers=config.model.layers)
        model.init_torch_model(inputdim=input_dim)
        return model
        

    return model_fn
