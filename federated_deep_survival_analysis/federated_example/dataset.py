import torch
from torchvision.datasets import MNIST
from torchvision.transforms import (
    ToTensor,
    Normalize,
    Compose,
)
from torch.utils.data import random_split
from torch.utils.data import DataLoader

SEED = 2023


def get_mnist(destination: str = "./data"):
    transform = Compose(
        [
            ToTensor(),
            Normalize(mean=[0.1307], std=tuple[0.3081]),
        ]
    )

    train = MNIST(
        destination,
        train=True,
        download=True,
        transform=transform,
    )

    test = MNIST(
        destination,
        train=False,
        download=True,
        transform=transform,
    )

    return train, test


def get_dataset(
    num_partitions: int,
    batch_size: int,
    val_ratio: float = 0.1,
):
    train, test = get_mnist()

    images_per_partition = len(train) // num_partitions

    partition_length = [
        images_per_partition
    ] * num_partitions

    train_partitions = random_split(
        train,
        partition_length,
        torch.Generator().manual_seed(SEED),
    )

    train_dataloaders = []
    validation_dataloaders = []

    for train_partition in train_partitions:
        total = len(train_partition)
        validation_length = int(val_ratio * total)
        train_length = total - validation_length

        train_, validation_ = random_split(
            train_partition,
            [train_length, validation_length],
            torch.Generator().manual_seed(SEED),
        )

        train_dataloaders.append(
            DataLoader(
                train_,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
            )
        )

        validation_dataloaders.append(
            DataLoader(
                validation_,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
            )
        )

    test_dataloader = DataLoader(test, batch_size=128)

    return (
        train_dataloaders,
        validation_dataloaders,
        test_dataloader,
    )
