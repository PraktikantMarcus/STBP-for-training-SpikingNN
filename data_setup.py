import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"
    #return "cuda" if torch.cuda.is_available() else "cpu"


def get_test_loader(batch_size=100, data_path="./raw/"):
    test_set = torchvision.datasets.MNIST(
        root=data_path, train=False, download=True, transform=transforms.ToTensor()
    )
    return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

def get_test_dataset(data_path="./raw/"):
    return torchvision.datasets.MNIST(
        root=data_path, train=False, download=True, transform=transforms.ToTensor()
    )

def get_train_loader(batch_size=100, data_path="./raw/"):
    train_set = torchvision.datasets.MNIST(
        root=data_path, train=True, download=True, transform=transforms.ToTensor()
    )
    return DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

def get_train_dataset(data_path="./raw/"):
    return torchvision.datasets.MNIST(
        root=data_path, train=True, download=True, transform=transforms.ToTensor()
    )