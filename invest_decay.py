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
import random

decay_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#decay_values = [0, 0.1]

def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)


def process_batch(args):
    """
    Process a batch of samples in a single worker.
    More efficient than process_sample for reducing overhead.
    
    Args:
        args: tuple of (images, labels,layers, model_state_dict, time_window, decay)
    
    Returns:
        tuple of (predictions, labels)
    """
    images, labels, layers, model_state_dict, time_window, decay = args
    
    # Create model in this worker process
    model = Event_SMLP_Quantized(layers, decay)
    model.load_state_dict(model_state_dict, strict=False)
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

def run_inference(args, decay):

    # Configuration
    time_window = 20
    num_workers = min(mp.cpu_count(), 48)
    batch_size = 100  # Samples per worker task
    data_path = "./raw/"

    print(f"\nConfiguration:")
    print(f"  CPU cores available: {num_workers}")
    print(f"  Samples per worker task: {batch_size}")
    print(f"  Time window: {time_window}")
    print()

    # Load checkpoint
    print("Loading checkpoint...")

    layer_string = "_".join(str(x) for x in args.layers)
    checkpoint_path = f"./checkpoint/ckpt_{layer_string}.t7"
    ckpt = torch.load(f"{checkpoint_path}", map_location=device)

    new_outdir = args.outdir + layer_string
    os.makedirs(new_outdir, exist_ok=True)

    # Get model state dict
    model_state = ckpt['net']
    print(f"✓ Loaded checkpoint (epoch={ckpt.get('epoch')}, acc={ckpt.get('acc'):.2f}%)")
    
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
        worker_args.append((images, labels, args.layers, model_state, time_window, decay))

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
    if 'acc' in ckpt:
        ckpt_acc = ckpt['acc']
        print(f"\nCheckpoint accuracy: {ckpt_acc:.2f}%")
        print(f"Inference accuracy:  {accuracy:.3f}%")
        print(f"Difference:          {abs(accuracy - ckpt_acc):.3f}%")

    # Return results for collection
    return {
        'decay': decay,
        'acc': accuracy
    }
def main():
    parser = argparse.ArgumentParser(description="Run multicore, fully-quantized inference. Using specific rounding and overflow mechanics for membrane quantization")
    parser.add_argument("--layers", type=int, nargs="+", 
                       default=[784, 400, 10],
                       help="Layer sizes (e.g., --layers 784 400 10)")
    parser.add_argument("--seed", type=int, default=0, help="Enter the global seed for reproducibility")
    parser.add_argument("--outdir", type=str, default="results/event_invest_decay/", help="Output CSV file for results")
    parser.add_argument("--decay", type=float, default= 0.2)

    args = parser.parse_args()
    print("Starting quantization parameter sweep...")

    # Set seed in main process
    set_seed(args.seed)

    layer_string = "_".join(str(x) for x in args.layers)

    mp.set_start_method('spawn', force=True)

    all_results = []

    for decay in enumerate(decay_values):
        print(f"Runningn inference on the model: {layer_string} with a decay {decay[1]} ")
        results = run_inference(args, 1)
        all_results.append(results)
        print()
    
    # Create DataFrame with results
    df = pd.DataFrame(all_results)
    
    # Sort by accuracy descending
    df = df.sort_values('acc', ascending=False)
    
    # Save to CSV
    output_path = args.outdir + layer_string +"/"+layer_string + f"_decay.csv"
    df.to_csv(output_path, index=False)
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}\n")
    

if __name__ == '__main__':
    main()
    