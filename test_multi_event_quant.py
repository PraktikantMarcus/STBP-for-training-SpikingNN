import torch
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torchvision
import time
from models.spiking_model import *
from models.quant_utils import*
from tqdm import tqdm
import numpy as np
import argparse
import pandas as pd
import os


def process_batch(args):
    """
    Process a batch of samples in a single worker.
    More efficient than process_sample for reducing overhead.
    
    Args:
        args: tuple of (images, labels, model_state_dict, time_window, rnd, ovf)
    
    Returns:
        tuple of (predictions, labels)
    """
    images, labels, model_state_dict, time_window, rnd, ovf = args
    
    # Create model in this worker process
    model = Event_SMLP_Quantized(True,
                                 6,
                                 8,
                                 rnd,
                                 ovf,
                                 False)
    model.load_state_dict(model_state_dict, strict=False)
    quantize_model_weights_(model,0,2,"nearest","saturate")
    model.eval()
    model.to('cpu')
    
    predictions = []
    
    # Process each image in the batch
    with torch.no_grad():
        for image in images:
            image = image.unsqueeze(0)  # Add batch dimension
            output = model(image, time_window=time_window)
            pred = output.argmax(dim=1).item()
            predictions.append(pred)
    
    return predictions, labels.tolist()

def run_inference(rnd = "nearest", ovf="saturate"):
    print("=" * 80)
    print(f"Running multicore, event-driven inference with '{rnd}'- and '{ovf}'-mechanics ")
    print("=" * 80)

    # Configuration
    time_window = 20
    num_workers = min(mp.cpu_count(), 32)
    batch_size = 100  # Samples per worker task
    data_path = "./raw/"

    print(f"\nConfiguration:")
    print(f"  CPU cores available: {num_workers}")
    print(f"  Samples per worker task: {batch_size}")
    print(f"  Time window: {time_window}")
    print()

    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load("./checkpoint/ckptspiking_model.t7", 
                           map_location='cpu', 
                           weights_only=True)
    # Get model state dict
    model_state = checkpoint['net']
    print(f"✓ Loaded checkpoint (epoch={checkpoint.get('epoch')}, acc={checkpoint.get('acc'):.2f}%)")
    
     # Load test dataset
    print("Loading test dataset...")
    test_dataset = torchvision.datasets.MNIST(
        root=data_path,
        train=False,
        download=False,
        transform=transforms.ToTensor()
    )

    # Create batches for multiprocessing
    # Group samples into batches for workers
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0  # Handling multiprocessing ourselves
    )

    print(f"✓ Loaded {len(test_dataset)} test samples")
    print(f"✓ Split into {len(test_loader)} batches")
    print()

    # Prepare arguments for each worker
    print(f"Starting inference with {num_workers} workers...")
    worker_args = []
    for images, labels in test_loader:
        worker_args.append((images, labels, model_state, time_window, rnd, ovf))

    # Create process pool and process batches
    start_time = time.time()
    
    with mp.Pool(processes=num_workers) as pool:
        # Process batches in parallel with progress bar
        results = list(tqdm(
            pool.imap(process_batch, worker_args),
            total=len(worker_args),
            desc="Processing batches",
            unit="batch"
        ))

    elapsed_time = time.time() - start_time

    # Collect results
    all_predictions = []
    all_labels = []
    
    for predictions, labels in results:
        all_predictions.extend(predictions)
        all_labels.extend(labels)
    
    # Calculate accuracy
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    correct = (all_predictions == all_labels).sum()
    total = len(all_labels)
    accuracy = 100.0 * correct / total
    
    # Print results
    print()
    print("=" * 70)
    print("Results:")
    print("=" * 70)
    print(f"Total samples:      {total}")
    print(f"Correct:            {correct}")
    print(f"Accuracy:           {accuracy:.3f}%")
    print(f"Total time:         {elapsed_time:.1f}s")
    print(f"Time per sample:    {elapsed_time/total*1000:.1f}ms")
    print(f"Throughput:         {total/elapsed_time:.1f} samples/sec")
    print("=" * 70)
    
    # Compare with checkpoint accuracy
    if 'acc' in checkpoint:
        ckpt_acc = checkpoint['acc']
        print(f"\nCheckpoint accuracy: {ckpt_acc:.2f}%")
        print(f"Inference accuracy:  {accuracy:.3f}%")
        print(f"Difference:          {abs(accuracy - ckpt_acc):.3f}%")

def main():
    parser = argparse.ArgumentParser(description="Run multicore, fully-quantized inference. Using specific rounding and overflow mechanics for membrane quantization")
    parser.add_argument("--rnd", help="Select valid rounding mechanics: floor, ceil, trunc, nearest, stochastic")
    parser.add_argument("--ovf", help="Select valid overflow mechanics: saturate, wrap")

    args = parser.parse_args()
    print("Running, import working")
    mp.set_start_method('spawn', force=True)
    run_inference(args.rnd, args.ovf)

if __name__ == '__main__':
    main()
    