import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def train(
    net: nn.Module,
    trainloader,
    optimizer,
    epochs,
    device: str,
):
    """Train network on training set"""
    criterion: torch.nn.CrossEntropyLoss = (
        torch.nn.CrossEntropyLoss()
    )
    net.train()
    net.to(device)

    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(
                device
            )
            loss = criterion(net(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


@torch.inference_mode()
def test(net: nn.Module, testloader, device: str):
    """Test network on full test set"""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    net.to(device)

    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(
            device
        )
        outputs = net(images)
        loss += criterion(outputs, labels).item()
        _, predicted = torch.max(outputs.data, dim=1)
        correct += (predicted == labels).sum().item()

    accuracy = correct / len(testloader.dataset)
    return loss, accuracy
