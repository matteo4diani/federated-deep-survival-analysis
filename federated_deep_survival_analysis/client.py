from collections import OrderedDict
from typing import Dict, List, Tuple, Union
from auton_survival import metrics
import numpy as np
import pandas as pd
import torch

import auton_survival
import flwr as fl


class DeepCoxPHClient(fl.client.NumPyClient):
    def __init__(
        self, model: auton_survival.DeepCoxPH, train, test, num_examples
    ) -> None:
        self.model = model
        self.trainloader = train
        self.testloader = test
        self.num_examples = num_examples

    def get_parameters(self, config):
        pass

    def set_parameters(self, parameters):
        pass

    def fit(self, parameters, config):
        pass

    def evaluate(self, parameters, config):
        pass
