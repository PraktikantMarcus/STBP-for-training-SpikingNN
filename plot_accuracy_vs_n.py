#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-GUI backend for scripts
import matplotlib.pyplot as plt

# Define colors for different m values
colors = {0: 'blue', 1: 'green', 2: 'red', 3: 'orange', 4: 'purple', 
          5: 'brown', 6: 'pink', 7: 'gray', 8: 'olive', 9: 'cyan'}

def main():
    parser = argparse.ArgumentParser(description="Plot accuracy vs n with m=2 for each rounding/overflow from a quantization CSV.")
    parser.add_argument("csv_path", help="Path to the CSV (e.g., qmn_ti_10bits.csv)")
    parser.add_argument("--max_m", help="Max m value to filter (default: 2)", type=int, default=2)
    parser.add_argument("--max_n", help="Max n value to plot (default: 8)", type=int, default=8)
    parser.add_argument("--min_m", help="Min m value to filter (default: 0)", type=int, default=0)
    parser.add_argument("--min_n", help="Min n value to plot (default: 0)", type=int, default=0)
    
    parser.add_argument("--type", help="What kind of graph is being plotted", type=str, default="single")
    # If you want to use single you have to set the m-value as the max_m argument
    args = parser.parse_args()

    # Load
    df = pd.read_csv(args.csv_path)

    # Drop baseline rows (m,n are NaN/None)
    df = df.dropna(subset=["m", "n", "rounding", "overflow", "acc"])

    # Ensure numeric
    df["m"] = df["m"].astype(int)
    df["n"] = df["n"].astype(int)

    if args.type == "single":
        outdir = f"plots_accuracy_Q{args.max_m}.{args.max_n}"
        os.makedirs(outdir, exist_ok=True)

        # Fix m = m
        df_m = df[df["m"] == args.max_m].copy()
        if df_m.empty:
            raise SystemExit(f"No rows with m=={args.m} found in the CSV. Check your input or bit-format.")

        # We expect these 5 rounding methods; but only plot those present
        desired_roundings = ["nearest", "trunc", "stochastic", "floor", "ceil"]
        roundings_present = [r for r in desired_roundings if r in set(df_m["rounding"].unique())]

        # We expect these two overflow modes; but only plot those present
        desired_overflows = ["saturate", "wrap"]
        overflows_present = [o for o in desired_overflows if o in set(df_m["overflow"].unique())]

        if not roundings_present:
            raise SystemExit("No expected rounding methods found among {nearest,trunc,stochastic,floor,ceil} in rows with m==2.")
        if not overflows_present:
            raise SystemExit(f"No expected overflow methods found among {{saturate,wrap}} in rows with m=={args.m}.")

        # Generate one plot per (rounding, overflow)
        for ovf in overflows_present:
            for rnd in roundings_present:
                sub = df_m[(df_m["rounding"] == rnd) & (df_m["overflow"] == ovf)].copy()
                if sub.empty:
                    print(f"[skip] No data for rounding={rnd}, overflow={ovf}")
                    continue

                # If multiple rows per n (e.g., repeats), take the best acc for that n
                sub = sub.groupby("n", as_index=False)["acc"].max().sort_values("n")

                plt.figure(figsize=(6, 4))
                plt.plot(sub["n"], sub["acc"], marker="o")
                plt.title(f"Accuracy vs n (m=2)\nrounding={rnd}, overflow={ovf}")
                plt.xlabel("n")
                plt.ylabel("Accuracy (%)")
                plt.ylim(0, 100)
                plt.grid(True, linestyle="--", alpha=0.4)
                plt.tight_layout()

                fname = f"acc_vs_n_m2_{rnd}_{ovf}.png"
                out_path = os.path.join(outdir, fname)
                plt.savefig(out_path, dpi=150)
                plt.close()
                print(f"Saved: {out_path}")

        print("Done.")

    elif args.type == "multi":
        outdir = f"plots_accuracy_Q{args.max_m}.{args.max_n}_multi"
        os.makedirs(outdir, exist_ok=True)

        for rnd in df["rounding"].unique():
            for ovf in df["overflow"].unique():
                # Create figure INSIDE the loop
                plt.figure(figsize=(8, 6))
                plt.title(f"Accuracy vs n (rounding={rnd}, overflow={ovf})")
                plt.xlabel("n (fractional bits)")
                plt.ylabel("Accuracy (%)")
                plt.ylim(0, 100)
                plt.grid(True, linestyle="--", alpha=0.4)
                
                # Plot one line for each m value
                for m_val in sorted(df["m"].unique()):
                    # Filter data for this specific combination
                    mask = (df["rounding"] == rnd) & (df["overflow"] == ovf) & (df["m"] == m_val)
                    subset = df[mask].sort_values("n")  # Sort by n for proper line plotting
                    
                    if len(subset) > 0:  # Only plot if data exists
                        plt.plot(subset["n"], subset["acc"], 
                                marker="o", label=f"m={m_val}", 
                                color=colors.get(m_val, 'black'))
                        print(f"  Plotted: m={m_val}, {len(subset)} points")
                
                plt.legend()
                plt.tight_layout()
                
                # Save figure AFTER all lines are plotted
                fname = f"acc_vs_n_{rnd}_{ovf}.png"
                out_path = os.path.join(outdir, fname)
                plt.savefig(out_path, dpi=150)
                plt.close()
                print(f"Saved: {out_path}\n")
        print("Done.")
                
if __name__ == "__main__":
    main()
