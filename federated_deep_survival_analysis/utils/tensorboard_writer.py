from typing import Dict
from torch.utils.tensorboard import SummaryWriter


class ClientWriter:
    def __init__(self, save_path: str, client_id: int):
        super(ClientWriter, self).__init__()
        self.save_path = save_path
        self.writer = SummaryWriter(
            f"{save_path}/custom/clients/{client_id}"
        )

    def write(self, metrics: Dict, epoch: int):
        for name, metric in metrics.items():
            self.writer.add_scalar(
                f"clients/{name}", metric, epoch
            )

        self.writer.flush()


class ServerWriter:
    def __init__(self, save_path: str):
        super(ServerWriter, self).__init__()
        self.save_path = save_path
        self.writer = SummaryWriter(
            f"{save_path}/custom/server"
        )

    def write(self, metrics: Dict, epoch: int):
        for name, metric in metrics.items():
            self.writer.add_scalar(
                f"server/{name}", metric, epoch
            )

        self.writer.flush()
