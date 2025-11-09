import pandas as pd
import matplotlib.pyplot as plt
import os

# Load results
df = pd.read_csv("membrane_quant_results/membrane_quantization_results.csv")

# Create comparison plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

experiments = ["mem_only", "weight_only", "both"]
titles = ["Membrane Only", "Weights Only", "Both Quantized"]
colors_map = {0: 'blue', 1: 'orange', 2: 'green'}

for idx, (exp, title) in enumerate(zip(experiments, titles)):
    ax = axes[idx]
    exp_df = df[df["experiment"] == exp]
    
    # Plot for each m value
    for m in sorted(exp_df["mem_m" if "mem" in exp_df.columns else "weight_m"].dropna().unique()):
        if exp == "mem_only":
            subset = exp_df[exp_df["mem_m"] == m].sort_values("mem_n")
            ax.plot(subset["mem_n"], subset["acc"], 
                   marker='o', label=f'm={int(m)}',
                   color=colors_map.get(int(m), 'black'),
                   linewidth=2, markersize=6)
        elif exp == "weight_only":
            subset = exp_df[exp_df["weight_m"] == m].sort_values("weight_n")
            ax.plot(subset["weight_n"], subset["acc"],
                   marker='s', label=f'm={int(m)}',
                   color=colors_map.get(int(m), 'black'),
                   linewidth=2, markersize=6)
        else:  # both
            subset = exp_df[exp_df["mem_m"] == m].sort_values("mem_n")
            ax.plot(subset["mem_n"], subset["acc"],
                   marker='^', label=f'm={int(m)}',
                   color=colors_map.get(int(m), 'black'),
                   linewidth=2, markersize=6)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel("n (fractional bits)", fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()

plt.tight_layout()
plt.savefig("membrane_quant_results/comparison.png", dpi=150)
print("✓ Saved comparison plot")
plt.close()