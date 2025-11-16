# Comments    :    
#   Event-driven SNN. Collect global pre-spike max/min across all samples,
#   layers and timesteps, updating extrema AFTER EACH accumulation event
#   (after each input spike added to hidden, and after each hidden spike
#   added to output), but BEFORE any layer fires at that stage.

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# --- Device & hyperparameters ---
device = torch.device("cpu")
thresh = 0.5
decay = 0.2
cfg_fc = [784, 400, 10]
time_window = 20

DATA_FILE = Path("txts/test_data.txt")
LABEL_FILE = Path("txts/test_labels.txt")
WEIGHT_FILE = Path("txts/model_weights.txt")
OUT_TXT = Path("txts/pot_range_event.txt")

# --- Spike activation ---
class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return x.gt(thresh).float()

act_fun = ActFun.apply

# --- SNN MLP model ---
class SNN_MLP(nn.Module):
    """
    Two-layer event-driven SNN.
    `process_time_step_collect_extrema` updates extrema AFTER EACH accumulation
    (hidden from inputs, output from hidden spikes) but BEFORE firing at that stage.
    """
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(cfg_fc[0], cfg_fc[1], bias=False)
        self.fc2 = nn.Linear(cfg_fc[1], cfg_fc[2], bias=False)
        self.register_buffer("h1_mem", torch.zeros(cfg_fc[1]))
        self.register_buffer("h2_mem", torch.zeros(cfg_fc[2]))
        self.register_buffer("input_vec", torch.zeros(cfg_fc[0]))

    def reset_state(self) -> None:
        self.h1_mem = torch.zeros(cfg_fc[1], device=device)
        self.h2_mem = torch.zeros(cfg_fc[2], device=device)

    def process_time_step_collect_extrema(self) -> Tuple[torch.Tensor, float, float]:
        """
        Use the current `self.input_vec` (shape [784], values {0,1}) to process one step.

        Returns:
            h2_spiked: (OUT,) output spike vector at end of step.
            step_max: float, max membrane among {hidden, output} pre-fire snapshots in this step.
            step_min: float, min membrane among {hidden, output} pre-fire snapshots in this step.
        """
        step_max = -float("inf")
        step_min = +float("inf")

        # 1) Hidden accumulation from input spikes — update extrema BEFORE hidden fires
        for idx in torch.nonzero(self.input_vec, as_tuple=False).flatten():
            self.h1_mem += self.fc1.weight[:, idx]
            h1_max = self.h1_mem.max().item()
            h1_min = self.h1_mem.min().item()
            if h1_max > step_max:
                step_max = h1_max
            if h1_min < step_min:
                step_min = h1_min

        # Hidden threshold & reset
        h1_spiked = act_fun(self.h1_mem)
        self.h1_mem[h1_spiked.bool()] = 0.0

        # 2) Output accumulation from hidden spikes — update extrema BEFORE output fires
        for h in torch.nonzero(h1_spiked, as_tuple=False).flatten():
            self.h2_mem += self.fc2.weight[:, h]
            h2_max = self.h2_mem.max().item()
            h2_min = self.h2_mem.min().item()
            if h2_max > step_max:
                step_max = h2_max
            if h2_min < step_min:
                step_min = h2_min

        # Output threshold & reset
        h2_spiked = act_fun(self.h2_mem)
        self.h2_mem[h2_spiked.bool()] = 0.0

        # Decay after both layers processed
        self.h1_mem *= decay
        self.h2_mem *= decay

        return h2_spiked, step_max, step_min

# --- Load weights ---
def load_weights_flat(model: SNN_MLP, weight_file: Path = WEIGHT_FILE) -> None:
    """Load flat txt weights into fc1/fc2 (expects IN*H1 + H1*OUT floats)."""
    tokens = weight_file.read_text().split()
    raw = np.array(tokens, dtype=np.float32)
    IN, H1, OUT = cfg_fc
    assert raw.size == IN * H1 + H1 * OUT, "Weight count mismatch"
    fc1_flat = raw[: IN * H1].reshape(H1, IN)
    fc2_flat = raw[IN * H1 :].reshape(OUT, H1)
    model.fc1.weight.data.copy_(torch.from_numpy(fc1_flat).to(device))
    model.fc2.weight.data.copy_(torch.from_numpy(fc2_flat).to(device))

# --- Load local spike trains & labels ---
def load_test_data(data_file: Path = DATA_FILE, label_file: Path = LABEL_FILE, T: int = time_window) -> Tuple[np.ndarray, np.ndarray]:
    """Load saved spike trains (flattened) and labels; return shapes (N,784,T) and (N,)."""
    spikes = np.loadtxt(data_file, dtype=np.float32)
    labels = np.loadtxt(label_file, dtype=np.int64)
    N = labels.shape[0]
    assert spikes.size == N * 784 * T, "Mismatch in spike data size"
    return spikes.reshape(N, 784, T), labels

# --- Main ---
def main() -> None:
    snn = SNN_MLP().to(device)
    load_weights_flat(snn, WEIGHT_FILE)

    spikes_np, labels_np = load_test_data(DATA_FILE, LABEL_FILE, time_window)
    N = labels_np.shape[0]

    # Global extrema across all samples/timesteps/layers (pre-fire, event-wise)
    global_max = -float("inf")
    global_min = +float("inf")

    correct = 0
    for i in tqdm(range(N), desc="Testing", unit="sample"):
        spike_mat = torch.from_numpy(spikes_np[i]).to(device)  # (784, T)
        label = int(labels_np[i])

        snn.reset_state()
        counts = torch.zeros(cfg_fc[2], device=device)

        for t in range(time_window):
            snn.input_vec = spike_mat[:, t]
            h2_spk, step_max, step_min = snn.process_time_step_collect_extrema()

            if h2_spk.any():
                counts += h2_spk

            if step_max > global_max:
                global_max = step_max
            if step_min < global_min:
                global_min = step_min

        pred = int(counts.argmax().item())
        if pred == label:
            correct += 1

    acc = 100.0 * correct / N
    print(f"\nTest Accuracy over {N} samples: {acc:.2f}% ({correct}/{N})")
    print(f"\nPre-spike global maximum (event-wise, hidden+output): {global_max:.6f}")
    print(f"Pre-spike global minimum (event-wise, hidden+output): {global_min:.6f}")

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TXT.open("w", encoding="utf-8") as f:
        f.write(f"{global_min:.8f}\n{global_max:.8f}\n")
    print(f"[INFO] Range saved to {OUT_TXT}")

if __name__ == "__main__":
    main()
