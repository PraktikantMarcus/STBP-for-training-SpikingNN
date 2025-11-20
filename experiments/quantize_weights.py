import torch
from quant_utils import *
import torchvision
import torchvision.transforms as transforms
import os
from models.data_setup import *
from models.spiking_model import*
import argparse
import pandas as pd

device = data_setup.get_device()
test_loader = data_setup.get_test_loader(batch_size=100, data_path="./raw/")

# Load trained model
snn = SMLP().to(device)
ckpt = torch.load("./checkpoint/ckptspiking_model.t7", map_location=device)
snn.load_state_dict(ckpt["net"])
snn.eval()

# Run sweep
df_results = run_quant_sweep(snn, test_loader, device,max_m=2, max_n=8)
print("Saved quant CSV and plot")

def qmn_grid (min_m: int = 0, min_n: int = 0, max_m: int = None, max_n: int = None):
    """
    Build all Qm.n pairs within given ranges.
    """
    pairs = []
    for m in range(min_m, max_m + 1):
        for n in range(min_n, max_n + 1):
            pairs.append((m, n))
    return pairs

def investigateQuantWeights(args):
    device = data_setup.get_device()
    test_loader = data_setup.get_test_loader(batch_size=100, data_path="./raw/")

    # Load trained model
    snn = SMLP(args.layers).to(device)
    layer_string = "_".join(str(x) for x in args.layers)
    ckpt = torch.load(f"./checkpoint/ckpt_"+layer_string+".t7", map_location=device)
    snn.load_state_dict(ckpt["net"])
    snn.eval()

    # Run sweep
    df_results = run_quant_sweep(snn, test_loader, device,max_m=2, max_n=8)
    print("Saved quant CSV and plot")



def main():
    parser = argparse.ArgumentParser(description="Investigate how weight quantization effects accuracy with different rounding and overflow mechanics")
    parser.add_argument("--layers", type=int, nargs="+", 
                       default=[784, 400, 10],
                       help="Layer sizes (e.g., --layers 784 400 10)")
    parser.add_argument("--rnd", help="Select valid rounding mechanics: floor, ceil, trunc, nearest, stochastic")
    parser.add_argument("--ovf", help="Select valid overflow mechanics: saturate, wrap")
    
    args = parser.parse_args()
    print(f"Starting weight quantization vs. accuracy: {args.layers}")

        print("Starting quantization parameter sweep...")
    print(f"m range: [{args.m_min}, {args.m_max}]")
    print(f"n range: [{args.n_min}, {args.n_max}]")
    print(f"Rounding: {args.rnd}, Overflow: {args.ovf}")
    print()

    mp.set_start_method('spawn', force=True)

    # Collect results for all configurations
    all_results = []
    total_configs = (args.m_max - args.m_min + 1) * (args.n_max - args.n_min + 1)
    current_config = 0

    # Iterate over all m and n combinations
    for m in range(args.m_min, args.m_max + 1):
        for n in range(args.n_min, args.n_max + 1):
            current_config += 1
            print(f"\n{'='*80}")
            print(f"Configuration {current_config}/{total_configs}: Q{m}.{n}")
            print(f"{'='*80}\n")
            
            result = run_inference(m, n, args.rnd, args.ovf)
            all_results.append(result)
            print()
    
    # Create DataFrame with results
    df = pd.DataFrame(all_results)
    
    # Sort by accuracy descending
    df = df.sort_values('accuracy', ascending=False)
    
    # Save to CSV
    output_path = args.output
    df.to_csv(output_path, index=False)
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}\n")
    
    # Print summary
    print("\nTop 10 Configurations by Accuracy:")
    print("="*80)
    print(df[['m', 'n', 'accuracy', 'throughput']].head(10).to_string(index=False))
    print()
    
    print("\nBottom 5 Configurations by Accuracy:")
    print("="*80)
    print(df[['m', 'n', 'accuracy', 'throughput']].tail(5).to_string(index=False))
    print()
    
    # Find best configuration
    best = df.iloc[0]
    print(f"\nBest Configuration:")
    print(f"  Q{int(best['m'])}.{int(best['n'])} - Accuracy: {best['accuracy']:.3f}%")
    print(f"  Throughput: {best['throughput']:.1f} samples/sec")
    print(f"  Time: {best['elapsed_time']:.1f}s")
    print()


if __name__ == '__main__':
    main()