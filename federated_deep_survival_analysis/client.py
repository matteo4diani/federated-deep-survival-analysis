from collections import OrderedDict
from typing import Dict, List, Tuple, Union
from auton_survival import metrics
import numpy as np
import pandas as pd
import torch

import auton_survival
import flwr as fl

class DeepCoxPHClient(fl.client.NumPyClient):
    """Flower client implementing CIFAR-10 image classification using
    PyTorch."""

    def __init__(
        self,
        model: auton_survival.DeepCoxPH,
        train: dict[str, np.ndarray],
        test: dict[str, np.ndarray],
        num_examples: dict
    ) -> None:
        self.model = model
        self.trainloader = train
        self.testloader = test
        self.num_examples = num_examples

    def get_parameters(self, config) -> List[np.ndarray]:
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        self.model.fit(self.trainloader['features'], 
                       self.trainloader['times'], 
                       self.trainloader['events'],
                       iters=1)
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss = self.model.test(self.model, self.testloader)
        accuracy = metrics.survival_regression_metric('ctd', self.testloader)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}