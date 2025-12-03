import torch
import pandas as pd
import os
from models.spiking_model import SMLP, SMLP_MemQuant
import models.data_setup
from models.quant_utils import evaluate_accuracy, quantize_model_weights_
import argparse
import pandas as pd
import time
import random
import numpy as np
from tqdm import tqdm

# device = data_setup.get_device()
# test_loader = data_setup.get_test_loader(batch_size=100, data_path="./raw/")

# # Load trained model
# snn = SMLP().to(device)
# ckpt = torch.load("./checkpoint/ckptspiking_model.t7", map_location=device)
# snn.load_state_dict(ckpt["net"])
# snn.eval()

# # Baseline accuracy
# base_acc = evaluate_accuracy(snn, test_loader, device)
# print(f"Baseline (FP32): {base_acc:.2f}%\n")

# results = []

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

def mem_quant(args):
    """
    Run membrane potential quantization experiments and save results.
    """
    # Prepare arguments for each config
    device = models.data_setup.get_device()
    test_loader = models.data_setup.get_test_loader(batch_size=100, data_path="./raw/")
   
    layer_string = "_".join(str(x) for x in args.layers)
    checkpoint_path = f"./checkpoint/ckpt_{layer_string}.t7"
    ckpt = torch.load(f"{checkpoint_path}", map_location=device)

    new_outdir = args.outdir + layer_string
    os.makedirs(new_outdir, exist_ok=True)

    results = []
    
    for m in range(args.m + 1):
        for n in range(args.n + 1):
            print(f"\nTesting Q{m}.{n} with {args.rnd} rounding and {args.ovf} overflow for quantization of membrane potentials...")

            # Create new model with membrane quantization
            snn_memquant = SMLP_MemQuant(
                quant_mem=True, 
                mem_m=m, 
                mem_n=n,
                mem_rounding=args.rnd,
                mem_overflow=args.ovf,
                layers = args.layers
            ).to(device)
            
            # Load same weights (not quantized)
            snn_memquant.load_state_dict(ckpt["net"])
            if args.fixWQuant:
                models.quant_utils.quantize_model_weights_(snn_memquant, args.wm, args.wn, "nearest", "saturate")
            elif args.dynWQuant:
                models.quant_utils.quantize_model_weights_(snn_memquant, m, n, args.rnd, args.ovf)
                    
            snn_memquant.eval()
            
            acc = models.quant_utils.evaluate_accuracy(snn_memquant, test_loader, device)
            print(f"  Q{m}.{n} membrane: {acc:.2f}%")
            
            results.append({
                "m": m,
                "n": n,
                "rounding": args.rnd,
                "overflow": args.ovf,
                "acc": acc,
            })
    df = pd.DataFrame(results)
    csv_path = os.path.join(new_outdir, f"{args.rnd}_{args.ovf}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")

    return df
    
    
def weight_quant_experiment():
    """
    Run weight quantization experiments and save results.
    """
    # Experiment 2: Weight quantization only (for comparison)
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Weight Quantization Only")
    print("=" * 70)

    for m in [0, 1, 2, 3, 4, 5]:
        for n in range(0, 9):
            for rnd in ["nearest", "stochastic", "floor", "ceil", "trunc"]:
                for ovf in ["saturate", "wrap"]:
                    print(f"\nTesting Q{m}.{n} with {rnd} rounding and {ovf} overflow for weight quantization ...")
            
                    # Use original SMLP with quantized weights
                    snn_wquant = SMLP().to(device)
                    snn_wquant.load_state_dict(ckpt["net"])
                    quantize_model_weights_(snn_wquant, m, n, rnd, ovf)
                    snn_wquant.eval()
                    
                    acc = evaluate_accuracy(snn_wquant, test_loader, device)
                    print(f"  Q{m}.{n} weights: {acc:.2f}%")
                    
                    results.append({
                        "experiment": "weight_only",
                        "weight_m": m,
                        "weight_n": n,
                        "mem_m": None,
                        "mem_n": None,
                        "rounding": rnd,
                        "overflow": ovf,
                        "acc": acc,
                        "total_bits": 1 + m + n
                    })

def both_quant_experiment():
    """
    Run experiments with both weights and membrane potentials quantized.
    """
    # Experiment 3: Both weights AND membrane quantized
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Both Weights and Membrane Quantization")
    print("=" * 70)

    for m in [0, 1, 2, 3, 4, 5]:
        for n in range(0, 9):
            for rnd in ["nearest", "stochastic", "floor", "ceil", "trunc"]:
                for ovf in ["saturate", "wrap"]:
                    print(f"\nTesting Q{m}.{n} with {rnd} rounding and {ovf} overflow for weight AND membrane potential quantization ...")
            
                    # Quantize both
                    snn_both = SMLP_MemQuant(
                        quant_mem=True,
                        mem_m=m,
                        mem_n=n,
                        mem_rounding=rnd,
                        mem_overflow=ovf
                    ).to(device)
                    
                    snn_both.load_state_dict(ckpt["net"])
                    quantize_model_weights_(snn_both, m, n, rnd, ovf)
                    snn_both.eval()
                    
                    acc = evaluate_accuracy(snn_both, test_loader, device)
                    print(f"  Q{m}.{n} both: {acc:.2f}%")
                    
                    results.append({
                        "experiment": "both",
                        "weight_m": m,
                        "weight_n": n,
                        "mem_m": m,
                        "mem_n": n,
                        "rounding": rnd,
                        "overflow": ovf,
                        "acc": acc,
                        "total_bits": 1 + m + n
                    })

def save_results():
    """
    Save results to CSV.
    """
    outdir = "membrane_quant_results"
    os.makedirs(outdir, exist_ok=True)

    df = pd.DataFrame(results)
    csv_path = os.path.join(outdir, "combined_membrane_quantization_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved results to {csv_path}")
    return df

def print_summary(df: pd.DataFrame):
    """
    Print summary of best configurations per experiment.
    """
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY - Best configurations per experiment:")
    print("=" * 70)

    for exp in ["mem_only", "weight_only", "both"]:
        exp_df = df[df["experiment"] == exp].sort_values("acc", ascending=False)
        print(f"\n{exp.upper()}:")
        print(exp_df.head(5).to_string(index=False))

def main():
    parser = argparse.ArgumentParser(description="Investigate how membrane quantization effects accuracy with different rounding and overflow mechanics")
    parser.add_argument("--layers", type=int, nargs="+", 
                       default=[784, 400, 10],
                       help="Layer sizes (e.g., --layers 784 400 10)")
    parser.add_argument("--rnd", help="Select valid rounding mechanics: floor, ceil, trunc, nearest, stochastic")
    parser.add_argument("--ovf", help="Select valid overflow mechanics: saturate, wrap")
    parser.add_argument("--m", default=5)
    parser.add_argument("--n", default=8)
    parser.add_argument("--seed", type=int, default=0, help="Enter the global seed for reproducibility")
    parser.add_argument("--outdir", default="results/fix_membrane_quantization/")
    parser.add_argument("--fixWQuant", action='store_true')
    parser.add_argument("--dynWQuant", action='store_true')
    parser.add_argument("--wm", default=1)
    parser.add_argument("--wn", default=3)


    args = parser.parse_args()
    if args.dynWQuant: 
        args.outdir = "results/dyn_full_quantization/"
        print(f"Dynamic weight quantization activated")
    
    elif args.fixWQuant:
        args.outdir = "results/fix_full_quantization/"
        print(f"Fixed weight quantization activated")
        

    print(f"Starting membrane quantization vs. accuracy: {args.layers}")
    print("Starting quantization parameter sweep...")
    print(f"Rounding: {args.rnd}, Overflow: {args.ovf}")
    print()

    # Set seed in main process
    set_seed(args.seed)

    all_results = []
    all_results = mem_quant(args)
    print(f"Membrane quantization finished")


if __name__ == '__main__':
    main()