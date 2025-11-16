import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_device():
    """
    Detects and returns the best available device.
    
    Returns:
        torch.device: Device object for tensor operations
    """
    try:
        # Check for Apple Silicon (MPS) - only on macOS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            print(f"✓ Using device: MPS (Apple Silicon)")
            return device
    except Exception as e:
        print(f"MPS check failed: {e}")
    
    # Check for NVIDIA GPU (CUDA)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # Explicitly use GPU 0
        print(f"✓ Using device: CUDA")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        return device
    
    # Fallback to CPU
    # print("⚠ Using device: CPU (no GPU detected)")
    return torch.device("cpu")

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