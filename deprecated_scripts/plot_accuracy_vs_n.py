#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-GUI backend for scripts
import matplotlib.pyplot as plt

# Define colors for different m values
colors = {0: 'blue',
          1: 'green', 
          2: 'red',
          3: 'orange',
          4: 'purple',
          5: 'brown',
          6: 'pink',
          7: 'gray',
          8: 'olive',
          9: 'cyan'}

markers = {0: 'o',      # Circle
           1: 's',      # Square
           2: '^',      # Triangle up
           3: 'D',      # Diamond
           4: 'v',      # Triangle down
           5: 'p',      # Pentagon
           6: '*',      # Star
           7: 'X',      # X (filled)
           8: 'P',      # Plus (filled)
           9: 'h'}      # Hexagon

linestyles = {0: '-',       # Solid
              1: '--',      # Dashed
              2: '-.',      # Dash-dot
              3: ':',       # Dotted
              4: '-',       # Solid (repeat)
              5: '--',      # Dashed (repeat)
              6: '-.',      # Dash-dot (repeat)
              7: ':',       # Dotted (repeat)
              8: '-',       # Solid (repeat)
              9: '--'}      # Dashed (repeat)

def main():
    parser = argparse.ArgumentParser(description="Plot accuracy vs n with m=2 for each rounding/overflow from a quantization CSV.")
    parser.add_argument("csv_path", help="Path to the CSV (e.g., qmn_ti_10bits.csv)")
    parser.add_argument("--max_m", help="Max m value to filter (default: 2)", type=int, default=2)
    parser.add_argument("--max_n", help="Max n value to plot (default: 3)", type=int, default=3)
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
                for m_val in range(args.min_m, args.max_m +1):
                    # Filter data for this specific combination
                    mask = (df["rounding"] == rnd) & (df["overflow"] == ovf) & (df["m"] == m_val)
                    subset = df[mask].sort_values("n")  # Sort by n for proper line plotting
                    
                    if len(subset) > 0:  # Only plot if data exists
                        plt.plot(subset["n"], subset["acc"], 
                            marker=markers.get(m_val, 'o'),          # Different marker per m
                            linestyle=linestyles.get(m_val, '-'),    # Different line style per m
                            label=f"m={m_val}", 
                            color=colors.get(m_val, 'black'),
                            linewidth=2.5,
                            markersize=8,
                            alpha=0.85)
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

    elif args.type == "combi":
        fig, axes = plt.subplots(2, 5, figsize=(15, 5))
        axes = axes.flatten()
        outdir = f"plots_accuracy_Q{args.max_m}.{args.max_n}_combi"
        os.makedirs(outdir, exist_ok=True)
        counter = 0

        for rnd in df["rounding"].unique():
            for ovf in df["overflow"].unique():
                
                ax = axes[counter]
                counter += 1
                print("Debug")
                # Plot one line for each m value
                for m_val in range(args.min_m, args.max_m +1):
                    # Filter data for this specific combination
                    mask = (df["rounding"] == rnd) & (df["overflow"] == ovf) & (df["m"] == m_val)
                    subset = df[mask].sort_values("n")  # Sort by n for proper line plotting
                    
                    if len(subset) > 0:  # Only plot if data exists
                        ax.plot(subset["n"], subset["acc"], 
                            marker=markers.get(m_val, 'o'),          # Different marker per m
                            linestyle=linestyles.get(m_val, '-'),    # Different line style per m
                            label=f"m={m_val}", 
                            color=colors.get(m_val, 'black'),
                            linewidth=2.5,
                            markersize=8,
                            alpha=0.85)
                        print(f"  Plotted: m={m_val}, {len(subset)} points")
                
                title = f"Rnd = {rnd}, Ovf = {ovf}"
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.set_xlabel("n (fractional bits)", fontsize=10)
                ax.set_ylabel("Accuracy (%)", fontsize=10)
                ax.set_ylim(0, 100)
                ax.grid(True, linestyle='--', alpha=0.4)
                ax.legend()  

        plt.tight_layout()
        fname = "acc_vs_n_combined.png"
        out_path = os.path.join(outdir, fname)
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")
        print("Done.")

    elif args.type == "membrane":
        # Load membrane quantization results
        df_mem = df.copy()
        
        # Filter for membrane-only experiment
        if "experiment" in df_mem.columns:
            df_mem = df_mem[df_mem["experiment"] == "mem_only"].copy()
        
        # Drop rows with NaN values
        df_mem = df_mem.dropna(subset=["mem_m", "mem_n", "rounding", "overflow", "acc"])
        
        # Ensure numeric
        df_mem["mem_m"] = df_mem["mem_m"].astype(int)
        df_mem["mem_n"] = df_mem["mem_n"].astype(int)
        
        print(f"Loaded {len(df_mem)} membrane quantization results")
        print(f"Rounding methods: {sorted(df_mem['rounding'].unique())}")
        print(f"Overflow methods: {sorted(df_mem['overflow'].unique())}")
        print(f"m values: {sorted(df_mem['mem_m'].unique())}")
        print(f"n values: {sorted(df_mem['mem_n'].unique())}")
        
        # Create subplot grid
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        outdir = f"plots_membrane_Q{args.max_m}.{args.max_n}_combi"
        os.makedirs(outdir, exist_ok=True)
        
        counter = 0
        for rnd in sorted(df_mem["rounding"].unique()):
            for ovf in sorted(df_mem["overflow"].unique()):
                
                if counter >= len(axes):
                    print(f"Warning: More combinations than subplot positions")
                    break
                
                ax = axes[counter]
                counter += 1
                
                print(f"\nProcessing subplot {counter}: rounding={rnd}, overflow={ovf}")
                
                # Plot one line for each m value
                for m_val in range(args.min_m, args.max_m + 1):
                    mask = (df_mem["rounding"] == rnd) & (df_mem["overflow"] == ovf) & (df_mem["mem_m"] == m_val)
                    subset = df_mem[mask].sort_values("mem_n")
                    
                    if len(subset) > 0:
                        ax.plot(subset["mem_n"], subset["acc"], 
                            marker=markers.get(m_val, 'o'),
                            linestyle=linestyles.get(m_val, '-'),
                            label=f"m={m_val}", 
                            color=colors.get(m_val, 'black'),
                            linewidth=2.0,
                            markersize=6,
                            alpha=0.85)
                        print(f"  Plotted m={m_val}: {len(subset)} points")
                
                # Format subplot
                title = f"Rnd={rnd}, Ovf={ovf}"
                ax.set_title(title, fontsize=10, fontweight='bold')
                ax.set_xlabel("n (fractional bits)", fontsize=9)
                ax.set_ylabel("Accuracy (%)", fontsize=9)
                ax.set_ylim(0, 100)
                ax.set_xticks(range(args.min_n, args.max_n + 1))
                ax.grid(True, linestyle='--', alpha=0.4)
                ax.legend(fontsize=8, loc='best')
        
        # Hide unused subplots
        for i in range(counter, len(axes)):
            axes[i].set_visible(False)
        
        # Add main title
        fig.suptitle('Membrane Potential Quantization - Accuracy vs n', 
                     fontsize=14, fontweight='bold', y=0.98)
        
        # Save figure
        plt.tight_layout()
        fname = "membrane_acc_vs_n_combined.png"
        out_path = os.path.join(outdir, fname)
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n✓ Saved: {out_path}")
        print("Done.")

    else:  # ADD THIS
        print(f"ERROR: Unknown plot type '{args.type}'")
        print(f"Valid types are: single, multi, combi, membrane")
        print(f"Example usage:")
        print(f"  python plot_accuracy_vs_n.py data.csv --type membrane --max_m 2")
if __name__ == "__main__":
    main()
