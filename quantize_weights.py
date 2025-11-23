import torch
import torch.multiprocessing as mp
from models.quant_utils import *
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import os
from models.data_setup import *
from models.spiking_model import*
import argparse
import pandas as pd
import time
import random
import numpy as np
from tqdm import tqdm

# device = models.data_setup.get_device()
# test_loader = models.data_setup.get_test_loader(batch_size=100, data_path="./raw/")

# # Load trained model
# snn = SMLP().to(device)
# ckpt = torch.load("./checkpoint/ckpt_784_400_10.t7", map_location=device)
# snn.load_state_dict(ckpt["net"])
# snn.eval()

# # Run sweep
# df_results = run_quant_sweep(snn, test_loader, device,max_m=2, max_n=8, outdir="test")
# print("Saved quant CSV and plot")

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


def light_quant_sweep(base_model, test_loader, device, args):
    """
    Run a Qm.n sweep
    Saves a CSV
    """
    layer_string = "_".join(str(x) for x in args.layers)
    new_outdir = args.outdir + layer_string
    outdir = new_outdir
    os.makedirs(outdir, exist_ok=True)

    # Baseline (no quantization)
    base_acc = models.quant_utils.evaluate_accuracy(base_model, test_loader, device)
    rows = [{"m": None, "n": None, "rounding": "none", "overflow": "none",
             "acc": base_acc}]

    grid= qmn_grid(min_m=0, min_n=0, max_m=args.m, max_n=args.n)
    print(f"Sweeping {len(grid)} Qm.n pairs")


    for (m, n) in grid:
        model_q = copy.deepcopy(base_model).to(device).eval()
        # torch.manual_seed(0)  # reproducible stochastic rounding
        models.quant_utils.quantize_model_weights_(model_q, m, n, rounding=args.rnd, overflow=args.ovf, which="weight")
        acc = models.quant_utils.evaluate_accuracy(model_q, test_loader, device)
        rows.append({"m": m, "n": n,
                    "rounding": args.rnd, "overflow": args.ovf,
                    "acc": acc})
        print(f"Finished Q{m}.{n} with {args.rnd}-rounding and {args.rnd}-overflow")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, f"{args.rnd}_{args.ovf}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")

    return df

def main():
    parser = argparse.ArgumentParser(description="Investigate how weight quantization effects accuracy with different rounding and overflow mechanics")
    parser.add_argument("--layers", type=int, nargs="+", 
                       default=[784, 400, 10],
                       help="Layer sizes (e.g., --layers 784 400 10)")
    parser.add_argument("--rnd", help="Select valid rounding mechanics: floor, ceil, trunc, nearest, stochastic")
    parser.add_argument("--ovf", help="Select valid overflow mechanics: saturate, wrap")
    parser.add_argument("--m", default=5)
    parser.add_argument("--n", default=8)
    parser.add_argument("--seed", type=int, default=0, help="Enter the global seed for reproducibility")
    parser.add_argument("--outdir", default="results/weight_quantization/")


    args = parser.parse_args()
    print(f"Starting weight quantization vs. accuracy: {args.layers}")
    print("Starting quantization parameter sweep...")
    print(f"Rounding: {args.rnd}, Overflow: {args.ovf}")
    print()

    # Set seed in main process
    set_seed(args.seed)

    # Prepare arguments for each config
    layer_string = "_".join(str(x) for x in args.layers)
    checkpoint_path = f"./checkpoint/ckpt_{layer_string}.t7"

    device = models.data_setup.get_device()
    test_loader = models.data_setup.get_test_loader(batch_size=100, data_path="./raw/")

    # Load trained model
    snn = SMLP(args.layers).to(device)
    ckpt = torch.load(f"./checkpoint/ckpt_{layer_string}.t7", map_location=device)
    snn.load_state_dict(ckpt["net"])
    snn.eval()

    all_results = []
    all_results = light_quant_sweep(snn, test_loader, device, args)
    print(f"Weight qunatization finished")


if __name__ == '__main__':
    main()