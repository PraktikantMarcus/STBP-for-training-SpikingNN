"""
Benchmark comparison: Sequential vs Multiprocessing Event_SMLP
"""

import torch
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torchvision
import time
from spiking_model import Event_SMLP

def sequential_inference(model, test_loader, device='cpu'):
    """Traditional sequential inference."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            predicted = outputs.argmax(dim=1)
            total += targets.numel()
            correct += (predicted == targets).sum().item()
    
    return correct, total


def process_batch_worker(args):
    """Worker function for multiprocessing."""
    images, labels, model_state_dict, time_window = args
    
    model = Event_SMLP(track_extrema=False)
    model.load_state_dict(model_state_dict, strict=False)
    model.eval()
    model.to('cpu')
    
    predictions = []
    with torch.no_grad():
        for image in images:
            image = image.unsqueeze(0)
            output = model(image, time_window=time_window)
            pred = output.argmax(dim=1).item()
            predictions.append(pred)
    
    return predictions, labels.tolist()


def multiprocess_inference(model_state, test_loader, num_workers, time_window=20):
    """Multiprocessing inference."""
    worker_args = []
    for images, labels in test_loader:
        worker_args.append((images, labels, model_state, time_window))
    
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(process_batch_worker, worker_args)
    
    all_predictions = []
    all_labels = []
    for predictions, labels in results:
        all_predictions.extend(predictions)
        all_labels.extend(labels)
    
    correct = sum(p == l for p, l in zip(all_predictions, all_labels))
    total = len(all_labels)
    
    return correct, total


def benchmark():
    print("=" * 80)
    print("Event-Driven SMLP: Sequential vs Multiprocessing Benchmark")
    print("=" * 80)
    print()
    
    # Configuration
    n_samples = 1000  # Use subset for faster benchmarking
    batch_size = 10
    time_window = 20
    num_workers = mp.cpu_count()
    
    print(f"Configuration:")
    print(f"  Samples to test: {n_samples}")
    print(f"  Batch size: {batch_size}")
    print(f"  Time window: {time_window}")
    print(f"  Available CPU cores: {num_workers}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    test_dataset = torchvision.datasets.MNIST(
        root='./raw/',
        train=False,
        download=False,
        transform=transforms.ToTensor()
    )
    
    # Use subset for faster testing
    subset = Subset(test_dataset, range(n_samples))
    test_loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load("./checkpoint/ckptspiking_model.t7", 
                           map_location='cpu', 
                           weights_only=True)
    model_state = checkpoint['net']
    
    model = Event_SMLP(track_extrema=False)
    model.load_state_dict(model_state, strict=False)
    model.to('cpu')
    
    print(f"✓ Model loaded (checkpoint acc: {checkpoint.get('acc', 0):.2f}%)")
    print()
    
    # Benchmark 1: Sequential
    print("-" * 80)
    print("Test 1: Sequential Processing (1 core)")
    print("-" * 80)
    start = time.time()
    correct_seq, total_seq = sequential_inference(model, test_loader, device='cpu')
    time_seq = time.time() - start
    acc_seq = 100.0 * correct_seq / total_seq
    
    print(f"Time:       {time_seq:.2f}s")
    print(f"Accuracy:   {acc_seq:.3f}%")
    print(f"Throughput: {n_samples/time_seq:.2f} samples/sec")
    print()
    
    # Benchmark 2: Multiprocessing with different worker counts
    for workers in [2, 4, 8, num_workers]:
        if workers > num_workers:
            continue
            
        print("-" * 80)
        print(f"Test: Multiprocessing ({workers} workers)")
        print("-" * 80)
        
        start = time.time()
        correct_mp, total_mp = multiprocess_inference(
            model_state, test_loader, workers, time_window
        )
        time_mp = time.time() - start
        acc_mp = 100.0 * correct_mp / total_mp
        
        speedup = time_seq / time_mp
        efficiency = speedup / workers * 100
        
        print(f"Time:       {time_mp:.2f}s")
        print(f"Accuracy:   {acc_mp:.3f}%")
        print(f"Throughput: {n_samples/time_mp:.2f} samples/sec")
        print(f"Speedup:    {speedup:.2f}x (vs sequential)")
        print(f"Efficiency: {efficiency:.1f}% (ideal: 100%)")
        print()
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    best_workers = num_workers
    best_time = time_seq / speedup  # From last test
    
    print(f"Sequential (1 core):          {time_seq:.2f}s")
    print(f"Multiprocessing ({best_workers} cores): {best_time:.2f}s")
    print(f"Best speedup:                 {speedup:.2f}x")
    print()
    print(f"Estimated time for 10,000 samples:")
    print(f"  Sequential:      {time_seq * 10:.1f}s ({time_seq * 10 / 60:.1f} min)")
    print(f"  Multiprocessing: {best_time * 10:.1f}s ({best_time * 10 / 60:.1f} min)")
    print("=" * 80)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    benchmark()
