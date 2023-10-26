from collections import OrderedDict
import flwr as fl
import torch
from torch.utils.data import DataLoader

from model import Net, train, test


class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        num_classes: int,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

        self.model: torch.nn.Module = Net(
            num_classes=num_classes
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def set_parameters(self, parameters):
        params_dict = zip(
            self.model.state_dict().keys(), parameters
        )

        state_dict = OrderedDict(
            {
                key: torch.Tensor(value)
                for key, value in params_dict
            }
        )

        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        lr = config["lr"]
        momentum = config["momentum"]

        optim = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
        )

        epochs = config["local_epochs"]

        # train locally
        train(
            self.model,
            trainloader=self.train_dataloader,
            optimizer=optim,
            epochs=epochs,
            device=self.device,
        )
