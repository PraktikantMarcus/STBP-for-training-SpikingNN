import torch
from spiking_model import SMLP
import data_setup
from quant_utils import quantize_tensor_fixed, quantize_model_weights_, evaluate_accuracy, run_quant_sweep

device = data_setup.get_device()
test_loader = data_setup.get_test_loader(batch_size=100, data_path="./raw/")

# Load trained model
snn = SMLP().to(device)
ckpt = torch.load("./checkpoint/ckptspiking_model.t7", map_location=device)
snn.load_state_dict(ckpt["net"])
snn.eval()

# Run sweep
df_results = run_quant_sweep(snn, test_loader, device,max_m=4, max_n=8)
print("Saved quant CSV and plot")