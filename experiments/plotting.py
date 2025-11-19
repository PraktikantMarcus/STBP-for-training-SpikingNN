#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-GUI backend for scripts
import matplotlib.pyplot as plt

# Define colors for different values
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
# Define markers for different values
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
# Define linestyles for different values
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
# Order of the Graphs in the combinedGraphs is dictated by the following orders
rounding_order = ["floor",
                   "ceil",
                    "trunc",
                    "nearest",
                    "stochastic"]
overflow_order = ["saturate",
                   "wrap"]

def main():
    parser = argparse.ArgumentParser(description="Combined file for general plotting")
    parser.add_argument("type", help="Type of plotting desired, one type is available for each plotting behaviour")

    args = parser.parse_args()
    result = "This result"
    if args.type == "singleW":
        result = singleWeightGraphs()
    
    elif args.type == "multiW":
        result = multiWeightGraph()

    elif args.type == "combiW":
        result = combiWeightGraph(max_m=5, max_n=8)
    
    elif args.type == "combiM":
        result = combiMembraneGraph(max_m=5, max_n=8)
    
    elif args.type == "mem_and_weights":
        result = combiMemWeightsGraph(max_m = 5, max_n=8)
    
    elif args.type == "full_event_quant":
        result = fullEventQuant(max_m = 8, max_n = 8)
    
    else:
        print(f"ERROR: Unknown plot type: '{args.type}'")
        print(f"Valid types are: combiW, combiM")
        print(f"Example usage:")
        print(f"python plotting.py combiW")

    print(result)

def dataLoader(dataPath: str, dataName :str):
    df = pd.read_csv(dataPath)

    if dataName == "weights":
        # Filter for membrane-only experiment
        if "experiment" in df.columns:
            df = df[df["experiment"] == "weight_only"].copy()
        
        # Drop rows with NaN values
        df = df.dropna(subset=["weight_m", "weight_n", "rounding", "overflow", "acc"])

        # Ensure numeric
        df["weight_m"] = df["weight_m"].astype(int)
        df["weight_n"] = df["weight_n"].astype(int)
    
    elif dataName == "membrane":

        # Filter for membrane-only experiment
        if "experiment" in df.columns:
            df = df[df["experiment"] == "mem_only"].copy()

        # Drop rows with NaN values
        df = df.dropna(subset=["mem_m", "mem_n", "rounding", "overflow", "acc"])
        
        # Ensure numeric
        df["mem_m"] = df["mem_m"].astype(int)
        df["mem_n"] = df["mem_n"].astype(int)
    
    elif dataName == "mem_and_weights":

        # Filter for membrane-only experiment
        if "experiment" in df.columns:
            df = df[df["experiment"] == "both"].copy()

        # Drop rows with NaN values
        df = df.dropna(subset=["mem_m", "mem_n", "rounding", "overflow", "acc"])
        
        # Ensure numeric
        df["mem_m"] = df["mem_m"].astype(int)
        df["mem_n"] = df["mem_n"].astype(int)

    elif dataName == "full_event_quant":
        df =df.dropna(subset=["m", "n", "rnd", "ovf", "accuracy"])

        # Ensure numeric
        df["m"] = df["m"].astype(int)
        df["n"] = df["n"].astype(int)

    else:
        print(f"ERROR: dataName in 'dataLoader()-function' invalid")

    return df

def realRangeCheck(df: pd.DataFrame, dataName: str, max_m: int, max_n: int, min_m: int, min_n:int):
    real_max_m = max_m
    real_max_n = max_n
    real_min_m = min_m
    real_min_n = min_n


    if dataName == "weights":
        real_max_m = df["weight_m"].max()
        real_max_n = df["weight_n"].max()
        real_min_m = df["weight_m"].min()
        real_min_n = df["weight_n"].min()

    elif dataName == "membrane":
        real_max_m = df["mem_m"].max()
        real_max_n = df["mem_n"].max()
        real_min_m = df["mem_m"].min()
        real_min_n = df["mem_n"].min()

    elif dataName == "full_event_quant":
        real_max_m = df["m"].max()
        real_max_n = df["n"].max()
        real_min_m = df["m"].min()
        real_min_n = df["n"].min()
    
    else:
        print(f"ERROR: dataName in 'realRangeCheck()-function' invalid")
        return 

    if (real_max_m < max_m):
        print(f"""
            The value 'max_m'={max_m}, is larger then the
            maximum value present in the datasource
            which is {real_max_m}
            """)

    if (real_max_n < max_n):
        print(f"""
            The value 'max_n'={max_n}, is larger then the
            maximum value present in the datasource
            which is {real_max_n}
            """)
        
    if (real_min_m > min_m):
        print(f"""
            The value 'min_m'={min_m}, is smaller then the
            minimum value present in the datasource
            which is {real_min_m}
            """)

    if (real_min_n > min_n):
        print(f"""
            The value 'min_n'={min_n}, is smaller then the
            minimum value present in the datasource
            which is {real_max_n}
            """)

    return (real_max_m, real_max_n, real_min_m, real_min_n)

def singleWeightGraphs():
    return "This function is currently not implemented, please use plot_accuracy_vs_n.py"

def multiWeightGraph():
    return "This function is currently not implemented, please use plot_accuracy_vs_n.py"

def combiWeightGraph(max_m = int(3), min_m = 0, max_n = 5, min_n = 0):
    """
    Plots 10 graphs as subgraphs, so that the accuracy loss attributed
    to the weight quantization can be compared. 
    10 subgraphs are returned because of the combination of the different
    rounding and overflow mechanisms.
    """
    dataPath = "membrane_quant_results/combined_membrane_quantization_results.csv"
    df = dataLoader(dataPath, "weights")

    max_m, max_n, min_m, min_n = realRangeCheck(df,"weights", max_m, max_n, min_m, min_n)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    counter = 0

    # One subgraph per possible combination of rounding and overflow methodology
    for rnd in rounding_order:
        for ovf in overflow_order:   

            if counter >= len(axes):
                print(f"Warning: More combinations than subplot positions")
                break

            ax = axes[counter]
            counter += 1

            # Plot one line for each m value
            for m_val in range(min_m, max_m +1):
                # Filter data for this specific combination
                mask = (
                    (df["rounding"] == rnd) &
                    (df["overflow"] == ovf) &
                    (df["weight_m"] == m_val) &
                    (df["weight_n"]<= max_n) &
                    (df["weight_n"]>= min_n))
                
                subset = df[mask].sort_values("weight_n")  # Sort by n for proper line plotting
            
                if len(subset) > 0:  # Only plot if data exists
                    ax.plot(
                        subset["weight_n"],
                        subset["acc"], 
                        marker = markers.get(m_val, 'o'),        # Different marker per m
                        linestyle=linestyles.get(m_val, '-'),    # Different line style per m
                        color=colors.get(m_val, 'black'),        # Different color per m 
                        label=f"m={m_val}",
                        linewidth=2.0,
                        markersize=6,
                        alpha=0.85)
            
            # Format subplot
            title = f"Rnd = {rnd}, Ovf = {ovf}"
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_xlabel("n (fractional bits)", fontsize=10)
            ax.set_ylabel("Accuracy (%)", fontsize=10)
            ax.set_ylim(0, 100)
            ax.set_xticks(range(min_n, max_n + 1))
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend(fontsize=6, loc='best')

    # Add main title
    fig.suptitle('Weights Quantization - Accuracy vs n', 
                    fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    outdir = f"plots/quantization_weights_Q{max_m}.{max_n}_combi"
    os.makedirs(outdir, exist_ok=True)
    fname = "weight_quantization.png"
    out_path = os.path.join(outdir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return "Saved to: "+ out_path

def combiMembraneGraph(max_m = 5, min_m = 0, max_n = 5, min_n = 0):
    """
    Plots 10 graphs as subgraphs, so that the accuracy loss attributed
    to the membrane potential quantization can be compared. 
    10 subgraphs are returned because of the combination of the different
    rounding and overflow mechanisms.
    """
    dataPath = "membrane_quant_results/combined_membrane_quantization_results.csv"
    df = dataLoader(dataPath, "membrane")

    max_m, max_n, min_m, min_n = realRangeCheck(df,"membrane", max_m, max_n, min_m, min_n)

    # Create subplot grid
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    counter = 0

    # One subgraph per possible combination of rounding and overflow methodology
    for rnd in rounding_order:
        for ovf in overflow_order:
            
            if counter >= len(axes):
                print(f"Warning: More combinations than subplot positions")
                break
            
            ax = axes[counter]
            counter += 1
            
            # Plot one line for each m value
            for m_val in range(min_m, max_m + 1):
                
                # Filter data for this specific combination
                mask = (
                    (df["rounding"] == rnd) &
                    (df["overflow"] == ovf) &
                    (df["mem_m"] == m_val) &
                    (df["mem_n"]<= max_n) &
                    (df["mem_n"]>= min_n))
                
                subset = df[mask].sort_values("mem_n") # Sort by n for proper line plotting
                
                if len(subset) > 0:
                    ax.plot(
                        subset["mem_n"],
                        subset["acc"], 
                        marker = markers.get(m_val, 'o'),        # Different marker per m
                        linestyle=linestyles.get(m_val, '-'),    # Different line style per m
                        color=colors.get(m_val, 'black'),        # Different color per m 
                        label=f"m={m_val}", 
                        linewidth=2.0,
                        markersize=6,
                        alpha=0.85)

            
            # Format subplot
            title = f"Rnd={rnd}, Ovf={ovf}"
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_xlabel("n (fractional bits)", fontsize=10)
            ax.set_ylabel("Accuracy (%)", fontsize=10)
            ax.set_ylim(0, 100)
            ax.set_xticks(range(min_n, max_n + 1))
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend(fontsize=6, loc='best')
    
    
    # Add main title
    fig.suptitle('Membrane Potential Quantization - Accuracy vs n', 
                    fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    outdir = f"plots/quantization_membrane_Q{max_m}.{max_n}_combi"
    os.makedirs(outdir, exist_ok=True)
    fname = "membrane_acc_vs_n_combined.png"
    out_path = os.path.join(outdir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    return "Saved to: "+ out_path

def combiMemWeightsGraph(max_m = 5, min_m = 0, max_n = 5, min_n = 0):
    """
    Plots 10 graphs as subgraphs, so that the accuracy loss attributed
    to the membrane potential AND weight quantization can be compared. 
    10 subgraphs are returned because of the combination of the different
    rounding and overflow mechanisms.
    """
    dataPath = "membrane_quant_results/combined_membrane_quantization_results.csv"
    df = dataLoader(dataPath, "mem_and_weights")

    max_m, max_n, min_m, min_n = realRangeCheck(df,"membrane", max_m, max_n, min_m, min_n)

    # Create subplot grid
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    counter = 0

    # One subgraph per possible combination of rounding and overflow methodology
    for rnd in rounding_order:
        for ovf in overflow_order:
            
            if counter >= len(axes):
                print(f"Warning: More combinations than subplot positions")
                break
            
            ax = axes[counter]
            counter += 1
            
            # Plot one line for each m value
            for m_val in range(min_m, max_m + 1):
                
                # Filter data for this specific combination
                mask = (
                    (df["rounding"] == rnd) &
                    (df["overflow"] == ovf) &
                    (df["mem_m"] == m_val) &
                    (df["mem_n"]<= max_n) &
                    (df["mem_n"]>= min_n))
                
                subset = df[mask].sort_values("mem_n") # Sort by n for proper line plotting
                
                if len(subset) > 0:
                    ax.plot(
                        subset["mem_n"],
                        subset["acc"], 
                        marker = markers.get(m_val, 'o'),        # Different marker per m
                        linestyle=linestyles.get(m_val, '-'),    # Different line style per m
                        color=colors.get(m_val, 'black'),        # Different color per m 
                        label=f"m={m_val}", 
                        linewidth=2.0,
                        markersize=6,
                        alpha=0.85)

            
            # Format subplot
            title = f"Rnd={rnd}, Ovf={ovf}"
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_xlabel("n (fractional bits)", fontsize=10)
            ax.set_ylabel("Accuracy (%)", fontsize=10)
            ax.set_ylim(0, 100)
            ax.set_xticks(range(min_n, max_n + 1))
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend(fontsize=6, loc='best')
    
    
    # Add main title
    fig.suptitle('Membrane Potential AND Weights Quantization - Accuracy vs n', 
                    fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    outdir = f"plots/quantization_mem_n_weights_Q{max_m}.{max_n}_combi"
    os.makedirs(outdir, exist_ok=True)
    fname = "mem_n_weights_acc_vs_n_combined.png"
    out_path = os.path.join(outdir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    return "Saved to: "+ out_path

def fullEventQuant(max_m = 5, min_m = 0, max_n = 5, min_n = 0):
    """
    Plots 10 graphs as subgraphs, so that the accuracy loss attributed
    to the membrane potential, weight quantization AND event driven approach
     can be compared. 10 subgraphs are returned because of the combination of the different
    rounding and overflow mechanisms.
    """

    # Create subplot grid
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    counter = 0

    # One subgraph per possible combination of rounding and overflow methodology
    for ovf in overflow_order:
        for rnd in rounding_order:

            if counter >= len(axes):
                print(f"Warning: More combinations than subplot positions")
                break
            
            ax = axes[counter]
            counter += 1

            dataPath = f"../results/results_{rnd}_{ovf}.csv"
            df = dataLoader(dataPath, "full_event_quant")

            max_m, max_n, min_m, min_n = realRangeCheck(df,"full_event_quant", max_m, max_n, min_m, min_n)

            # Plot one line for each m value
            for m_val in range(min_m, max_m + 1):
               
                # Filter data for this specific combination
                mask = (
                    (df["rnd"] == rnd) &
                    (df["ovf"] == ovf) &
                    (df["m"] == m_val) &
                    (df["n"]<= max_n) &
                    (df["n"]>= min_n))
                
                subset = df[mask].sort_values("n") # Sort by n for proper line plotting

                if len(subset) > 0:
                    ax.plot(
                        subset["n"],
                        subset["accuracy"], 
                        marker = markers.get(m_val, 'o'),        # Different marker per m
                        linestyle=linestyles.get(m_val, '-'),    # Different line style per m
                        color=colors.get(m_val, 'black'),        # Different color per m 
                        label=f"m={m_val}", 
                        linewidth=2.0,
                        markersize=6,
                        alpha=0.85)
            
            # Format subplot
            title = f"Rnd={rnd}, Ovf={ovf}"
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_xlabel("n (fractional bits)", fontsize=10)
            ax.set_ylabel("Accuracy (%)", fontsize=10)
            ax.set_ylim(0, 100)
            ax.set_xticks(range(min_n, max_n + 1))
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend(fontsize=6, loc='best')
    
    # Add main title
    fig.suptitle('Event Driven + Quantization - Accuracy vs n', 
                    fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    outdir = f"../plots/full_event_quant{max_m}.{max_n}"
    os.makedirs(outdir, exist_ok=True)
    fname = "full_event_quant.png"
    out_path = os.path.join(outdir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    return "Saved to: "+ out_path



if __name__ == "__main__":
    main()