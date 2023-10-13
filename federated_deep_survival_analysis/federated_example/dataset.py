from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose
def get_mnist(destination: str='./data'):
    
    transform = Compose([ToTensor(), Normalize(mean=[0.1307], std=tuple[0.3081])])
    
    train = MNIST(destination, train=True, download=True, transform=transform)
    test = MNIST(destination, train=False, download=True, transform=transform)

    return train, test, None

def get_dataset():
    train, test, val = get_mnist()
    return train, test, val