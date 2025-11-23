# import random
# import numpy as np

# def set_seed(seed):
#     """Set all random seeds for reproducibility"""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# def process_batch(args):
#     """Process a batch - now with seeding"""
#     images, labels, model_state_dict, time_window, m, n, rnd, ovf, seed = args
    
#     # CRITICAL: Set seed in each worker process
#     set_seed(seed)
    
#     # Create model AFTER setting seed
#     model = Event_SMLP_Quantized(True, m, n, rnd, ovf, False)
#     model.load_state_dict(model_state_dict, strict=False)
#     quantize_model_weights_(model, 0, 2, "nearest", "saturate")
#     model.eval()
#     model.to('cpu')
    
#     predictions = []
    
#     with torch.no_grad():
#         for image in images:
#             image = image.unsqueeze(0)
#             output = model(image, time_window=time_window)
#             pred = output.argmax(dim=1).item()
#             predictions.append(pred)
    
#     return predictions, labels.tolist()

# def run_inference(m, n, rnd="nearest", ovf="saturate", base_seed=42):
#     print("=" * 80)
#     print(f"Running Q{m}.{n} with '{rnd}' rounding and '{ovf}' overflow (seed={base_seed})")
#     print("=" * 80)

#     # ... configuration code ...

#     # Prepare arguments - ADD SEED
#     print(f"Starting inference with {num_workers} workers...")
#     worker_args = []
#     for batch_idx, (images, labels) in enumerate(test_loader):
#         # Unique seed per batch for deterministic but varied randomness
#         batch_seed = base_seed + batch_idx
#         worker_args.append((images, labels, model_state, time_window, m, n, rnd, ovf, batch_seed))

#     # ... rest of processing ...
    
#     return {
#         'm': m,
#         'n': n,
#         'rnd': rnd,
#         'ovf': ovf,
#         'accuracy': accuracy,
#         'seed': base_seed,  # Record seed used
#         # ... other fields ...
#     }

# def main():
#     parser = argparse.ArgumentParser()
#     # ... existing arguments ...
#     parser.add_argument("--seed", type=int, default=42, 
#                        help="Random seed for reproducibility")
    
#     args = parser.parse_args()
    
#     # Set seed in main process
#     set_seed(args.seed)
    
#     print(f"Random seed: {args.seed}")
    
#     # ... rest of main ...
    
#     for m in range(args.m_min, args.m_max + 1):
#         for n in range(args.n_min, args.n_max + 1):
#             # Use deterministic seed per (m,n) config
#             config_seed = args.seed + m * 100 + n
#             result = run_inference(m, n, args.rnd, args.ovf, config_seed)
#             all_results.append(result)



from concurrent.futures import ProcessPoolExecutor, as_completed
import torch.multiprocessing as mp
import torch
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

def process_single_image(args):
    """Process one image - innermost level"""
    image, label, quantized_state, layers, time_window, seed = args
    
    set_seed(seed)
    
    model = SMLP(layers)
    model.load_state_dict(quantized_state)
    model.eval()
    
    with torch.no_grad():
        image = image.unsqueeze(0)
        output = model(image, time_window=time_window)
        pred = output.argmax(dim=1).item()
    
    return pred, label

def evaluate_one_config(args_tuple):
    """Evaluate one Q(m,n) config - middle level, runs in parallel"""
    args, m, n, config_seed, checkpoint_path = args_tuple
    
    print(f"Starting Q{m}.{n}")
    
    # Load and quantize model ONCE
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model = SMLP(args.layers)
    model.load_state_dict(checkpoint['net'])
    quantize_model_weights_(model, m, n, args.rnd, args.ovf)
    quantized_state = model.state_dict()
    
    # Load test data
    test_dataset = torchvision.datasets.MNIST(
        root="./raw/", train=False, download=False,
        transform=transforms.ToTensor()
    )
    test_dataset = Subset(test_dataset, range(1000))  # Reduce for testing
    
    # Prepare per-image arguments
    image_args = []
    for idx in range(len(test_dataset)):
        image, label = test_dataset[idx]
        image_args.append((image, label, quantized_state, args.layers, 20, config_seed + idx))
    
    # Process images in parallel (inner parallelism)
    predictions = []
    labels = []
    
    with ProcessPoolExecutor(max_workers=32) as executor:  # Inner pool
        futures = [executor.submit(process_single_image, arg) for arg in image_args]
        
        for future in tqdm(as_completed(futures), total=len(futures), 
                          desc=f"Q{m}.{n}", leave=False):
            pred, label = future.result()
            predictions.append(pred)
            labels.append(label)
    
    # Calculate accuracy
    correct = sum(p == l for p, l in zip(predictions, labels))
    accuracy = 100.0 * correct / len(labels)
    
    print(f"Q{m}.{n} done: {accuracy:.2f}%")
    
    return {
        'm': m, 'n': n,
        'rnd': args.rnd, 'ovf': args.ovf,
        'accuracy': accuracy,
        'config_seed': config_seed
    }

def light_quant_sweep(base_model, test_loader, device,
                    roundings="nearest",
                    overflows="saturate",
                    outdir="test_quant_results_ti",
                    min_m: int = 0, min_n: int = 0,
                    max_m: int = 5, max_n: int = 8,
                    which: str = "weight"):
    """
    Run a TI Qm.n sweep under a total bit budget (default: <= 10 bits).
    Saves a CSV and a heatmap of BEST accuracy per (m,n) over rounding/overflow.
    """

    max_total_bits = 1 + max_m + max_n
    new_outdir = outdir + f"_{max_total_bits}bits"
    outdir = new_outdir
    os.makedirs(outdir, exist_ok=True)

    # Baseline (no quantization)
    base_acc = models.quant_utils.evaluate_accuracy(base_model, test_loader, device)
    rows = [{"m": None, "n": None, "rounding": "none", "overflow": "none",
             "acc": base_acc, "total_bits": None, "format": "TI"}]

    grid= qmn_grid(min_m=min_m, min_n=min_n, max_m=max_m, max_n=max_n)
    print(f"Sweeping {len(grid)} TI Qm.n pairs with total bits <= {max_total_bits}...")


    for (m, n) in grid:
        model_q = copy.deepcopy(base_model).to(device).eval()
        torch.manual_seed(0)  # reproducible stochastic rounding
        models.quant_utils.quantize_model_weights_(model_q, m, n, rounding="nearest", overflow="saturate", which=which)
        acc = models.quant_utils.evaluate_accuracy(model_q, test_loader, device)
        rows.append({"m": m, "n": n, "rounding": "nearest", "overflow": "saturate",
                        "acc": acc, "total_bits": 1 + m + n, "format": "TI"})
        print(f"Finished Q{m}.{n} with nearest-rounding and saturate-overflow")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(outdir, f"qmn_ti_{max_total_bits}bits.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")

    # Heatmap of BEST acc per (m,n) across rounding/overflow
    best = (df.dropna(subset=["m","n"])
              .groupby(["m","n"], as_index=False)["acc"].max())

    Ms = sorted(set(best["m"]))
    Ns = sorted(set(best["n"]))
    acc_grid = [[float('nan') for _ in Ns] for _ in Ms]
    for _, r in best.iterrows():
        i = Ms.index(r["m"]); j = Ns.index(r["n"])
        acc_grid[i][j] = r["acc"]

    plt.figure(figsize=(10, 6))
    im = plt.imshow(acc_grid, aspect="auto", origin="lower")
    plt.colorbar(im, label="Accuracy (%)")
    plt.xticks(range(len(Ns)), [f"n={n}" for n in Ns])
    plt.yticks(range(len(Ms)), [f"m={m}" for m in Ms])
    plt.title(f"TI Qm.n — best accuracy per pair (total bits ≤ {max_total_bits})")
    for i, m in enumerate(Ms):
        for j, n in enumerate(Ns):
            val = acc_grid[i][j]
            if not (val != val):  # not NaN
                plt.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)
    heat_path = os.path.join(outdir, f"qmn_ti_{max_total_bits}bits_heatmap_best.png")
    plt.tight_layout()
    plt.savefig(heat_path, dpi=150)
    plt.close()
    print(f"Saved heatmap to {heat_path}")

    # Top configs
    print("Top TI Qm.n configurations (best over rounding/overflow):")
    print(best.sort_values("acc", ascending=False).head(10).to_string(index=False))

    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=[784, 400, 10])
    parser.add_argument("--rnd", required=True)
    parser.add_argument("--ovf", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parallel_configs", type=int, default=1,
                       help="Number of Q(m,n) configs to run in parallel")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Generate grid
    grid = qmn_grid(min_m=0, min_n=0, max_m=5, max_n=8)
    
    # Prepare arguments for each config
    layer_string = "_".join(str(x) for x in args.layers)
    checkpoint_path = f"./checkpoint/ckpt_{layer_string}.t7"
    
    config_args = []
    for m, n in grid:
        config_seed = args.seed + m * 100 + n
        config_args.append((args, m, n, config_seed, checkpoint_path))
    
    # Outer parallelism: Multiple Q(m,n) configs in parallel
    print(f"Processing {len(grid)} configs with {args.parallel_configs} parallel")
    
    device = models.data_setup.get_device()
    test_loader = models.data_setup.get_test_loader(batch_size=100, data_path="./raw/")

    # Load trained model
    snn = SMLP(args.layers).to(device)
    ckpt = torch.load("./checkpoint/ckpt_784_400_10.t7", map_location=device)
    snn.load_state_dict(ckpt["net"])
    snn.eval()

    all_results = []
    
    # with ProcessPoolExecutor(max_workers=args.parallel_configs) as executor:
    #     futures = [executor.submit(evaluate_one_config, arg) for arg in config_args]
        
    #     for future in tqdm(as_completed(futures), total=len(futures),
    #                       desc="Overall progress"):
    #         result = future.result()
    #         all_results.append(result)

    all_results = light_quant_sweep(snn, test_loader, device,max_m=2, max_n=8, outdir="test")
    
    # Save results
    # df = pd.DataFrame(all_results).sort_values('accuracy', ascending=False)
    # output_path = f"./results/weight_quantization_{layer_string}.csv"
    # df.to_csv(output_path, index=False)
    
    print(f"\nBest: Q{int(df.iloc[0]['m'])}.{int(df.iloc[0]['n'])} - {df.iloc[0]['accuracy']:.2f}%")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()