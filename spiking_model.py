import torch
import torch.nn as nn
import torch.nn.functional as F

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "mps" if torch.backends.mps.is_available() else "cpu" #Apple Silicon support
thresh = 0.5 # neuronal threshold
lens = 0.5 # hyper-parameters of approximate function
decay = 0.2 # decay constants
num_classes = 10
batch_size  = 100
learning_rate = 1e-3
num_epochs = 100 # max epoch
# define approximate firing function
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - thresh) < lens
        return grad_input * temp.float()

act_fun = ActFun.apply
# membrane potential update
def mem_update(ops, x, mem, spike):
    mem = mem * decay * (1. - spike) + ops(x)
    spike = act_fun(mem) # act_fun : approximation firing function
    return mem, spike

# cnn_layer(in_planes, out_planes, stride, padding, kernel_size)
cfg_cnn = [(1, 32, 1, 1, 3),
           (32, 32, 1, 1, 3),]
# kernel size
cfg_kernel = [28, 14, 7]
# fc layer
cfg_fc = [128, 10]

# Dacay learning_rate
def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer

class SMLP(nn.Module):
    """
    784-400-10 spiking MLP (fully connected).
    Uses  existing act_fun (surrogate spike) and mem_update (LIF update).
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 400)   # 784 -> 400
        self.fc2 = nn.Linear(400, 10)      # 400 -> 10

    def forward(self, input, time_window=20):
        # input: [B, 1, 28, 28] with values in [0,1]
        B = input.size(0)

        # Membrane and spike states
        h1_mem = torch.zeros(B, 400, device=device)
        h1_spike = torch.zeros(B, 400, device=device)
        h1_sumspike = torch.zeros(B, 400, device=device)

        h2_mem = torch.zeros(B, 10, device=device)
        h2_spike = torch.zeros(B, 10, device=device)
        h2_sumspike = torch.zeros(B, 10, device=device)

        for _ in range(time_window):
            # Poisson/Bernoulli spike encoding from static image
            x = (input > torch.rand_like(input)).float()   # [B,1,28,28]
            x = x.view(B, -1)                              # flatten to [B,784]

            # LIF updates through fully connected layers
            h1_mem, h1_spike = mem_update(self.fc1, x, h1_mem, h1_spike)
            h1_sumspike += h1_spike

            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem, h2_spike)
            h2_sumspike += h2_spike

        outputs = h2_sumspike / time_window   # rate-coded outputs
        return outputs

class SMLP_MemQuant(nn.Module):
    """
    SMLP with configurable membrane potential quantization.
    """
    def __init__(self, 
                 quant_mem: bool = False,
                 mem_m: int = 2, 
                 mem_n: int = 4,
                 mem_rounding: str = "nearest",
                 mem_overflow: str = "saturate"):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 400)
        self.fc2 = nn.Linear(400, 10)
        
        # Membrane quantization config
        self.quant_mem = quant_mem
        self.mem_m = mem_m
        self.mem_n = mem_n
        self.mem_rounding = mem_rounding
        self.mem_overflow = mem_overflow

    def forward(self, input, time_window=20):
        B = input.size(0)

        # Membrane and spike states
        h1_mem = torch.zeros(B, 400, device=device)
        h1_spike = torch.zeros(B, 400, device=device)
        h1_sumspike = torch.zeros(B, 400, device=device)

        h2_mem = torch.zeros(B, 10, device=device)
        h2_spike = torch.zeros(B, 10, device=device)
        h2_sumspike = torch.zeros(B, 10, device=device)

        for _ in range(time_window):
            x = (input > torch.rand_like(input)).float()
            x = x.view(B, -1)

            # Layer 1 update
            h1_mem, h1_spike = mem_update(self.fc1, x, h1_mem, h1_spike)
            
            # Quantize membrane potential if enabled
            if self.quant_mem:
                from quant_utils import quantize_membrane_potential
                h1_mem = quantize_membrane_potential(
                    h1_mem, self.mem_m, self.mem_n, 
                    self.mem_rounding, self.mem_overflow
                )
            
            h1_sumspike += h1_spike

            # Layer 2 update
            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem, h2_spike)
            
            # Quantize membrane potential if enabled
            if self.quant_mem:
                from quant_utils import quantize_membrane_potential
                h2_mem = quantize_membrane_potential(
                    h2_mem, self.mem_m, self.mem_n,
                    self.mem_rounding, self.mem_overflow
                )
            
            h2_sumspike += h2_spike

        outputs = h2_sumspike / time_window
        return outputs


class SCNN(nn.Module):
    def __init__(self):
        super(SCNN, self).__init__()
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[0]
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
        in_planes, out_planes, stride, padding, kernel_size = cfg_cnn[1]
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)

        self.fc1 = nn.Linear(cfg_kernel[-1] * cfg_kernel[-1] * cfg_cnn[-1][1], cfg_fc[0])
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1])

    def forward(self, input, time_window = 20):
        c1_mem = c1_spike = torch.zeros(batch_size, cfg_cnn[0][1], cfg_kernel[0], cfg_kernel[0], device=device)
        c2_mem = c2_spike = torch.zeros(batch_size, cfg_cnn[1][1], cfg_kernel[1], cfg_kernel[1], device=device)

        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=device)

        for step in range(time_window): # simulation time steps
            x = input > torch.rand(input.size(), device=device) # prob. firing

            c1_mem, c1_spike = mem_update(self.conv1, x.float(), c1_mem, c1_spike)

            x = F.avg_pool2d(c1_spike, 2)

            c2_mem, c2_spike = mem_update(self.conv2,x, c2_mem,c2_spike)

            x = F.avg_pool2d(c2_spike, 2)
            x = x.view(batch_size, -1)

            h1_mem, h1_spike = mem_update(self.fc1, x, h1_mem, h1_spike)
            h1_sumspike += h1_spike
            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem,h2_spike)
            h2_sumspike += h2_spike

        outputs = h2_sumspike / time_window
        return outputs


