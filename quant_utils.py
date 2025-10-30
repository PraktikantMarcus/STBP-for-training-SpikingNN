import math, copy, os, torch
import matplotlib
matplotlib.use("Agg")  # safe backend for scripts
import matplotlib.pyplot as plt
import pandas as pd


@torch.inference_mode()
def quantize_tensor_fixed(x: torch.Tensor,
                          m: int, n: int,
                          rounding: str = "nearest",
                          overflow: str = "saturate") -> torch.Tensor:
    """
    TI Qm.n fixed-point quantization (signed).
    m: integer magnitude bits (excludes sign), m >= 0
    n: fractional bits, n >= 0
    Integer range: [-2^m, 2^m - 1]
    Real range: that / (2^n)
    """
    if m < 0 or n < 0:
        raise ValueError(f"Invalid Qm.n: m={m}, n={n}. Need m>=0, n>=0 (TI-Format).")

    # OLD (INCORRECT):
    # qmin = -(1 << m)        # -2^m
    # qmax = (1 << m) - 1     # 2^m - 1
   
    # NEW (CORRECT):
    # qmin = -(1 << (m + n))      # -2^(m+n)
    # qmax = (1 << (m + n)) - 1   # 2^(m+n) - 1
    
    # test_qmin = -(1 << m)
    # test_qmax = (1 << m) - (1.0 / (1 << n))
    # print(f"The range for Q{m}.{n} is: [{test_qmin}, {test_qmax}]")

    # MINE 
    qmin = -(1 << m) 
    qmax = (1 << m) - (1.0 / (1 << n))
    
    scale = float(1 << n)   # 2^n

    x_scaled = x * scale

    if rounding == "nearest":
        q = torch.round(x_scaled)
    elif rounding == "floor":
        q = torch.floor(x_scaled)
    elif rounding == "ceil":
        q = torch.ceil(x_scaled)
    elif rounding == "trunc":
        q = torch.trunc(x_scaled)
    elif rounding == "stochastic":
        frac = torch.frac(x_scaled)
        q = torch.floor(x_scaled) + (torch.rand_like(x_scaled) < frac).to(x_scaled.dtype)
    else:
        raise ValueError(f"Unknown rounding: {rounding}")

    if overflow == "saturate":
        q = torch.clamp(q, qmin, qmax)
    elif overflow == "wrap":
        span = qmax - qmin  # +1 for inclusive range (i dont think we need this here)
        q = (q - qmin) % span + qmin
    else:
        raise ValueError(f"Unknown overflow: {overflow}")

    return q / scale


@torch.inference_mode()
def quantize_model_weights_(model: torch.nn.Module,
                            m: int, n: int,
                            rounding: str = "nearest",
                            overflow: str = "saturate",
                            which: str = "weight"):
    """
    In-place quantization (TI Qm.n) for all parameters whose name contains `which`.
    Default: only 'weight' tensors.
    """
    for name, p in model.named_parameters():
        if which in name:
            p.copy_(quantize_tensor_fixed(p, m, n, rounding, overflow))


@torch.inference_mode()
def evaluate_accuracy(model: torch.nn.Module,
                      loader, device: str) -> float:
    model.eval()
    total, correct = 0, 0
    for inputs, targets in loader:
        inputs  = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        pred    = outputs.argmax(dim=1)
        total  += targets.numel()
        correct += (pred == targets).sum().item()
    return 100.0 * correct / total


def qmn_grid (min_m: int = 0, min_n: int = 0, max_m: int = None, max_n: int = None):
    """
    Build all Qm.n pairs within given ranges.
    """
    pairs = []
    for m in range(min_m, max_m + 1):
        for n in range(min_n, max_n + 1):
            pairs.append((m, n))
    return pairs
    


def run_quant_sweep(base_model, test_loader, device,
                    roundings=("nearest", "ceil", "floor", "trunc", "stochastic"),
                    overflows=("saturate", "wrap"),
                    outdir="quant_results_ti",
                    min_m: int = 0, min_n: int = 0,
                    max_m: int = 9, max_n: int = 9,
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
    base_acc = evaluate_accuracy(base_model, test_loader, device)
    rows = [{"m": None, "n": None, "rounding": "none", "overflow": "none",
             "acc": base_acc, "total_bits": None, "format": "TI"}]

    grid= qmn_grid(min_m=min_m, min_n=min_n, max_m=max_m, max_n=max_n)
    print(f"Sweeping {len(grid)} TI Qm.n pairs with total bits <= {max_total_bits}...")


    for (m, n) in grid:
        for rnd in roundings:
            for ovf in overflows:
                model_q = copy.deepcopy(base_model).to(device).eval()
                torch.manual_seed(0)  # reproducible stochastic rounding
                quantize_model_weights_(model_q, m, n, rounding=rnd, overflow=ovf, which=which)
                acc = evaluate_accuracy(model_q, test_loader, device)
                rows.append({"m": m, "n": n, "rounding": rnd, "overflow": ovf,
                             "acc": acc, "total_bits": 1 + m + n, "format": "TI"})

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
