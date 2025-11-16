import torch
import pandas as pd
import os
from spiking_model import SMLP, SMLP_MemQuant
import data_setup
from quant_utils import evaluate_accuracy, quantize_model_weights_

device = data_setup.get_device()
test_loader = data_setup.get_test_loader(batch_size=100, data_path="./raw/")

# Load trained model
snn = SMLP().to(device)
ckpt = torch.load("./checkpoint/ckptspiking_model.t7", map_location=device)
snn.load_state_dict(ckpt["net"])
snn.eval()

# Baseline accuracy
base_acc = evaluate_accuracy(snn, test_loader, device)
print(f"Baseline (FP32): {base_acc:.2f}%\n")

results = []

def mem_quant_experiment():
    """
    Run membrane potential quantization experiments and save results.
    """

    # Experiment 1: Membrane quantization only (no weight quantization)
    print("=" * 70)
    print("EXPERIMENT 1: Membrane Quantization Only")
    print("=" * 70)

    for m in [0, 1, 2, 3, 4, 5 ]:
        for n in range(0, 9):
            for rnd in ["nearest", "stochastic", "floor", "ceil", "trunc"]:
                for ovf in ["saturate", "wrap"]:
                    print(f"\nTesting Q{m}.{n} with {rnd} rounding and {ovf} overflow for quantization of membrane potentials...")

                    # Create new model with membrane quantization
                    snn_memquant = SMLP_MemQuant(
                        quant_mem=True, 
                        mem_m=m, 
                        mem_n=n,
                        mem_rounding=rnd,
                        mem_overflow=ovf
                    ).to(device)
                    
                    # Load same weights (not quantized)
                    snn_memquant.load_state_dict(ckpt["net"])
                    snn_memquant.eval()
                    
                    acc = evaluate_accuracy(snn_memquant, test_loader, device)
                    print(f"  Q{m}.{n} membrane: {acc:.2f}%")
                    
                    results.append({
                        "experiment": "mem_only",
                        "weight_m": None,
                        "weight_n": None,
                        "mem_m": m,
                        "mem_n": n,
                        "rounding": rnd,
                        "overflow": ovf,
                        "acc": acc,
                        "total_bits": 1 + m + n
                    })
    
    
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

if __name__ == "__main__":
    # mem_quant_experiment()
    # weight_quant_experiment()
    # both_quant_experiment()
    # df=save_results()
    # print_summary(df)