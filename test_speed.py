import torch
import time
from models.spiking_model import Event_SMLP_Quantized

print("Testing model speed...")

model = Event_SMLP_Quantized(layer_sizes=(784, 400, 10), decay=0.2)
model.train()

# Time one batch
dummy_input = torch.randn(100, 1, 28, 28)
dummy_target = torch.zeros(100, 10).scatter_(1, torch.randint(0, 10, (100, 1)), 1)

start = time.time()
output = model(dummy_input)
loss = ((output - dummy_target) ** 2).mean()
loss.backward()
elapsed = time.time() - start

print(f"Time for 1 batch (100 samples): {elapsed:.2f}s")
print(f"Expected time for 10 batches: {elapsed * 10 / 60:.1f} minutes")
print(f"Expected time for 600 batches (1 epoch): {elapsed * 600 / 60:.1f} minutes")
print(f"Expected time for 100 epochs: {elapsed * 600 * 100 / 3600:.1f} hours")