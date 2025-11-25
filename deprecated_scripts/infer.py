# minimal loader + test inference
import torch
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
from spiking_model import *
from event_driven_smlp import *
import matplotlib.pyplot as plt
import os
import data_setup


device = data_setup.get_device()
test_loader = data_setup.get_test_loader(batch_size=100, data_path="./raw/")

# Recreate the model architecture
snn = Event_SMLP(track_extrema=False).to(device)  



# Load weights from the saved state
ckpt = torch.load("./checkpoint/ckptspiking_model.t7", map_location=device, weights_only=True)  
snn.load_state_dict(ckpt["net"], strict=False)  # strict=False: ignore buffers not in checkpoint
snn.eval()  # Set the model to evaluation mode

correct = 0
total = 0
max_images = 10000  # Limit to 100 images for faster testing, can only be a multiple of 100

with torch.inference_mode():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device, dtype=torch.long)
        outputs = snn(inputs)
        predicted = outputs.argmax(dim=1)
        total += targets.numel()
        correct += (predicted == targets).sum().item()
        
        print(f"Images processed: {total}")
        # Stop after processing max_images
        if total >= max_images:
            break

print(f"Loaded epoch={ckpt.get('epoch')}, best_acc={ckpt.get('acc'):.2f}")
print('Test Accuracy of the model on the %d test images: %.3f' % (total, 100 * correct / total))



# print("Model parameters:")
# for name, _ in snn.named_parameters():
#     print(" ", name)

# print("Plotting weight distributions for all layers...")
# os.makedirs("weight_graph", exist_ok=True)

# for name, param in snn.named_parameters():
#     if "weight" not in name:
#         continue

#     w = param.detach().cpu().numpy().flatten()
#     w_min, w_max = w.min(), w.max()

#     plt.figure()
#     plt.hist(w, bins=50, color="skyblue", edgecolor="black")
#     plt.title(f"Weight distribution for {name}")
#     plt.xlabel("Weight value")
#     plt.ylabel("Frequency")

#     # Add vertical lines for min and max
#     plt.axvline(w_min, color="red", linestyle="--", linewidth=1.5, label=f"Min: {w_min:.4f}")
#     plt.axvline(w_max, color="green", linestyle="--", linewidth=1.5, label=f"Max: {w_max:.4f}")

#     # Optional: text annotations above the lines
#     y_min, y_max = plt.ylim()
#     plt.text(w_min, y_max * 0.9, f"{w_min:.4f}", color="red", ha="right", va="bottom", fontsize=8, rotation=90)
#     plt.text(w_max, y_max * 0.9, f"{w_max:.4f}", color="green", ha="left", va="bottom", fontsize=8, rotation=90)

#     plt.legend(loc="upper right")
#     plt.tight_layout()

#     out_path = f"weight_graph/{name.replace('.', '_')}_weights.png"
#     plt.savefig(out_path, dpi=150)
#     plt.close()

#     print(f"Saved: {out_path} (min={w_min:.4f}, max={w_max:.4f})")