# check_membrane_range.py
import torch
from spiking_model import SMLP_Debug
import data_setup

device = data_setup.get_device()
test_loader = data_setup.get_test_loader(batch_size=100, data_path="./raw/")

snn = SMLP_Debug().to(device)
ckpt = torch.load("./checkpoint/ckptspiking_model.t7", map_location=device)
snn.load_state_dict(ckpt["net"])
snn.eval()

print("Checking membrane potential ranges...")
with torch.no_grad():
    for i, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        _ = snn(inputs)
        if i >= 10:  # Check first 10 batches
            break

print("\nMembrane Potential Statistics:")
print(f"Layer 1 - Min: {snn.mem_stats['h1_min']:.6f}, Max: {snn.mem_stats['h1_max']:.6f}")
print(f"Layer 2 - Min: {snn.mem_stats['h2_min']:.6f}, Max: {snn.mem_stats['h2_max']:.6f}")
print(f"\nFor reference:")
print(f"Q0.0 range: [{-1.0:.4f}, {0.0:.4f}]")
print(f"Q0.1 range: [{-1.0:.4f}, {0.5:.4f}]")
print(f"Q0.2 range: [{-1.0:.4f}, {0.75:.4f}]")
print(f"Q0.3 range: [{-1.0:.4f}, {0.875:.4f}]")
print(f"Q1.3 range: [{-2.0:.4f}, {1.875:.4f}]")