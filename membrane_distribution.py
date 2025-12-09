import torch
import pandas as pd
import os
from models.spiking_model import Event_SMLP_Quantized
import models.data_setup
from tqdm import tqdm
import argparse
import numpy as np
import random


def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def collect_membrane_stats(model, test_loader, device, num_samples=1000):
    """
    Collect ALL membrane potential values across all layers for distribution plotting.
    
    Args:
        model: Event-driven spiking neural network model
        test_loader: DataLoader for test data
        device: Device to run on
        num_samples: Number of test samples to process
        
    Returns:
        dict: Statistics containing all membrane values and summary stats
    """
    model.eval()
    model.track_extrema = True  # Enable extrema tracking
    
    # Store ALL membrane potentials in a list
    all_membrane_potentials = []
    
    global_max = -float("inf")
    global_min = +float("inf")
    sample_count = 0
    
    print(f"Collecting membrane potential statistics...")
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Processing batches"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            B = inputs.size(0)
            
            # Process each sample in the batch independently
            for b in range(B):
                if sample_count >= num_samples:
                    break
                    
                model.reset_state()
                sample = inputs[b:b+1]
                
                # Run forward pass with extrema tracking
                for t in range(20):  # time_window = 20
                    x = (sample > torch.rand_like(sample)).float()
                    model.input_vec = x.view(-1)
                    
                    # Process timestep and collect ALL membrane values
                    mems, step_max, step_min = model.process_time_step_collect_extrema()
                    
                    # Append all membrane potentials from this timestep
                    # mems should be a list/tensor of all membrane values
                    all_membrane_potentials.extend(mems.cpu().numpy().flatten())
                    
                    if step_max > global_max:
                        global_max = step_max
                    if step_min < global_min:
                        global_min = step_min
                
                sample_count += 1
            
            if sample_count >= num_samples:
                break
    
    # Convert to numpy array for efficient storage and analysis
    all_membrane_potentials = np.array(all_membrane_potentials)
    
    stats = {
        "global_min": global_min,
        "global_max": global_max,
        "range": global_max - global_min,
        "num_samples": sample_count,
        "all_values": all_membrane_potentials,  # All membrane potentials
        "mean": float(np.mean(all_membrane_potentials)),
        "std": float(np.std(all_membrane_potentials)),
        "median": float(np.median(all_membrane_potentials)),
        "total_values": len(all_membrane_potentials)
    }
    
    return stats

def main():
    parser = argparse.ArgumentParser(
        description="Collect membrane potential distribution statistics for event-driven SNN"
    )
    parser.add_argument(
        "--layers", 
        type=int, 
        nargs="+", 
        required=True,
        help="Layer sizes (e.g., --layers 784 400 10)"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=10000,
        help="Number of test samples to analyze"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=100,
        help="Batch size for data loading"
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        default="./raw/",
        help="Path to MNIST data"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--outdir", 
        type=str, 
        default="results/1.1 - membrane_distributions/",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Setup device and data
    device = models.data_setup.get_device()
    test_loader = models.data_setup.get_test_loader(
        batch_size=args.batch_size, 
        data_path=args.data_path
    )
    
    # Create model identifier string
    layer_string = "_".join(str(x) for x in args.layers)
    print(f"\n{'='*70}")
    print(f"Analyzing membrane potentials for model: {layer_string}")
    print(f"{'='*70}\n")
    
    # Load model checkpoint
    checkpoint_path = f"./checkpoint/ckpt_{layer_string}.t7"
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print(f"Please train the model first or check the checkpoint path.")
        return
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Create event-driven model with extrema tracking
    model = Event_SMLP_Quantized(
        layer_sizes=tuple(args.layers),
        quant_mem=False,  # No quantization for baseline measurement
        track_extrema=True
    ).to(device)
    
    # Load weights
    model.load_state_dict(ckpt["net"], strict=False)
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"Analyzing {args.num_samples} samples from test set...\n")
    
    # Collect membrane potential statistics
    stats = collect_membrane_stats(
        model, 
        test_loader, 
        device, 
        num_samples=args.num_samples
    )
    
   # In main(), after collecting stats:

    # Print results
    print(f"\n{'='*70}")
    print(f"RESULTS for {layer_string}")
    print(f"{'='*70}")
    print(f"Global minimum membrane potential: {stats['global_min']:.6f}")
    print(f"Global maximum membrane potential: {stats['global_max']:.6f}")
    print(f"Total range: {stats['range']:.6f}")
    print(f"Mean: {stats['mean']:.6f}")
    print(f"Std Dev: {stats['std']:.6f}")
    print(f"Median: {stats['median']:.6f}")
    print(f"Total values collected: {stats['total_values']:,}")
    print(f"Samples analyzed: {stats['num_samples']}")
    print(f"{'='*70}\n")

    # Save summary statistics to CSV
    os.makedirs(args.outdir, exist_ok=True)

    results_df = pd.DataFrame([{
        "model_config": layer_string,
        "layers": str(args.layers),
        "num_layers": len(args.layers) - 1,
        "global_min": stats['global_min'],
        "global_max": stats['global_max'],
        "range": stats['range'],
        "mean": stats['mean'],
        "std": stats['std'],
        "median": stats['median'],
        "total_values": stats['total_values'],
        "num_samples": stats['num_samples'],
        "seed": args.seed
    }])

    csv_path = os.path.join(args.outdir, f"membrane_stats_{layer_string}.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"✓ Saved summary statistics to: {csv_path}\n")

    # Save ALL membrane potential values for plotting
    npy_path = os.path.join(args.outdir, f"membrane_values_{layer_string}.npy")
    np.save(npy_path, stats['all_values'])
    print(f"✓ Saved all membrane values to: {npy_path}\n")

    # Create a histogram plot
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.hist(stats['all_values'], bins=200, alpha=0.7, edgecolor='black')
    plt.xlabel('Membrane Potential', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Membrane Potential Distribution - {layer_string}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f'Mean: {stats["mean"]:.3f}')
    plt.axvline(stats['median'], color='green', linestyle='--', linewidth=2, label=f'Median: {stats["median"]:.3f}')
    plt.legend()

    plot_path = os.path.join(args.outdir, f"membrane_distribution_{layer_string}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved distribution plot to: {plot_path}\n")
    plt.close()

if __name__ == "__main__":
    main()