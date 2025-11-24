"""
Reproducibility utilities for PyTorch training
Ensures deterministic behavior across training runs
"""

import torch
import numpy as np
import random
import os

def set_seed(seed=42):
    """
    Set all random seeds for reproducible training.
    
    Args:
        seed (int): Random seed value
    """
    # Python random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # PyTorch deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Environment variable for hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✓ Random seed set to {seed} for reproducibility")

def set_seed_worker(worker_id):
    """
    Set seed for DataLoader workers.
    Use this with DataLoader's worker_init_fn parameter.
    
    Args:
        worker_id (int): Worker ID (automatically provided by DataLoader)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_reproducible_dataloader(dataset, batch_size, shuffle=True, num_workers=0, seed=42):
    """
    Create a DataLoader with reproducible behavior.
    
    Args:
        dataset: PyTorch dataset
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle data
        num_workers (int): Number of workers
        seed (int): Random seed
        
    Returns:
        DataLoader with deterministic behavior
    """
    # Create generator with fixed seed
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=set_seed_worker if num_workers > 0 else None,
        generator=generator
    )

def print_reproducibility_info():
    """Print current reproducibility settings"""
    print("=" * 60)
    print("Reproducibility Settings")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN deterministic: {torch.backends.cudnn.deterministic}")
        print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    print(f"Python hash seed: {os.environ.get('PYTHONHASHSEED', 'Not set')}")
    print("=" * 60)

# Example usage
if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(42)
    
    # Print settings
    print_reproducibility_info()
    
    # Test that random operations are deterministic
    print("\nTesting deterministic behavior:")
    print("Random tensor 1:", torch.rand(3))
    
    # Reset seed
    set_seed(42)
    print("Random tensor 2 (after reset):", torch.rand(3))
    print("These should be identical!")