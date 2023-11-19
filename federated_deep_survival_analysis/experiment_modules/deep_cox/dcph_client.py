from collections import OrderedDict
import torch
import flwr as fl
from auton_survival import DeepCoxPH

from federated_deep_survival_analysis.utils.tensorboard_writer import (
    ClientWriter,
)

from .dcph_dataset import (
    SurvivalDataset,
)

from .dcph_model import (
    test,
    train,
)


class DeepCoxPHClient(fl.client.NumPyClient):
    def __init__(
        self,
        trainloader: SurvivalDataset,
        valloader: SurvivalDataset,
        model_fn,
        client_id=None,
        save_path=None,
    ) -> None:
        self.model: DeepCoxPH = model_fn()
        self.trainloader = trainloader
        self.valloader = valloader
        self.client_id = client_id
        self.save_path = save_path

    def get_parameters(self, config={}):
        return [
            val.cpu().numpy()
            for _, val in self.model.torch_module.state_dict().items()
        ]

    def set_parameters(self, parameters):
        params_dict = zip(
            self.model.torch_module.state_dict().keys(),
            parameters,
        )

        state_dict = OrderedDict(
            {k: torch.Tensor(v) for k, v in params_dict}
        )

        self.model.torch_module.load_state_dict(
            state_dict, strict=True
        )

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        train(self.model, self.trainloader, config)

        loss, concordance_index = test(
            self.model, self.trainloader
        )

        metrics = {
            "train_loss": float(loss),
            "train_concordance_index": float(
                concordance_index[0]
            ),
        }

        writer = ClientWriter(
            self.save_path, self.client_id
        )
        writer.write(metrics, config["server_round"])

        return (
            self.get_parameters(),
            len(self.trainloader),
            metrics,
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        loss, concordance_index = test(
            self.model, self.valloader
        )

        metrics = {
            "test_loss": float(loss),
            "test_concordance_index": float(
                concordance_index[0]
            ),
        }

        return (
            float(loss),
            len(self.valloader),
            metrics,
        )


def get_client_fn(
    trainloaders=None,
    valloaders=None,
    model_fn=None,
    save_path=None,
):
    """Return a function that can be used by the VirtualClientEngine.

    to spawn a FlowerClient with client id `cid`.
    """

    def client_fn(cid: str):
        return DeepCoxPHClient(
            trainloader=trainloaders[int(cid)],
            valloader=valloaders[int(cid)],
            model_fn=model_fn,
            client_id=int(cid),
            save_path=save_path,
        )

    # return the function to spawn client
    return client_fn
