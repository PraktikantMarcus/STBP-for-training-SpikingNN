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

# Experiment 1: Membrane quantization only (no weight quantization)
print("=" * 70)
print("EXPERIMENT 1: Membrane Quantization Only")
print("=" * 70)

for m in [0, 1, 2]:
    for n in range(0, 9):
        print(f"\nTesting Q{m}.{n} for membrane potentials...")
        
        # Create new model with membrane quantization
        snn_memquant = SMLP_MemQuant(
            quant_mem=True, 
            mem_m=m, 
            mem_n=n,
            mem_rounding="nearest",
            mem_overflow="saturate"
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
            "acc": acc,
            "total_bits": 1 + m + n
        })

# Experiment 2: Weight quantization only (for comparison)
print("\n" + "=" * 70)
print("EXPERIMENT 2: Weight Quantization Only")
print("=" * 70)

for m in [0, 1, 2]:
    for n in range(0, 9):
        print(f"\nTesting Q{m}.{n} for weights...")
        
        # Use original SMLP with quantized weights
        snn_wquant = SMLP().to(device)
        snn_wquant.load_state_dict(ckpt["net"])
        quantize_model_weights_(snn_wquant, m, n, "nearest", "saturate")
        snn_wquant.eval()
        
        acc = evaluate_accuracy(snn_wquant, test_loader, device)
        print(f"  Q{m}.{n} weights: {acc:.2f}%")
        
        results.append({
            "experiment": "weight_only",
            "weight_m": m,
            "weight_n": n,
            "mem_m": None,
            "mem_n": None,
            "acc": acc,
            "total_bits": 1 + m + n
        })

# Experiment 3: Both weights AND membrane quantized
print("\n" + "=" * 70)
print("EXPERIMENT 3: Both Weights and Membrane Quantization")
print("=" * 70)

for m in [0, 1, 2]:
    for n in range(0, 9):
        print(f"\nTesting Q{m}.{n} for BOTH...")
        
        # Quantize both
        snn_both = SMLP_MemQuant(
            quant_mem=True,
            mem_m=m,
            mem_n=n,
            mem_rounding="nearest",
            mem_overflow="saturate"
        ).to(device)
        
        snn_both.load_state_dict(ckpt["net"])
        quantize_model_weights_(snn_both, m, n, "nearest", "saturate")
        snn_both.eval()
        
        acc = evaluate_accuracy(snn_both, test_loader, device)
        print(f"  Q{m}.{n} both: {acc:.2f}%")
        
        results.append({
            "experiment": "both",
            "weight_m": m,
            "weight_n": n,
            "mem_m": m,
            "mem_n": n,
            "acc": acc,
            "total_bits": 1 + m + n
        })

# Save results
outdir = "membrane_quant_results"
os.makedirs(outdir, exist_ok=True)

df = pd.DataFrame(results)
csv_path = os.path.join(outdir, "membrane_quantization_results.csv")
df.to_csv(csv_path, index=False)
print(f"\n✓ Saved results to {csv_path}")

# Print summary
print("\n" + "=" * 70)
print("SUMMARY - Best configurations per experiment:")
print("=" * 70)

for exp in ["mem_only", "weight_only", "both"]:
    exp_df = df[df["experiment"] == exp].sort_values("acc", ascending=False)
    print(f"\n{exp.upper()}:")
    print(exp_df.head(5).to_string(index=False))