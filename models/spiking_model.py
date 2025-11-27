import torch
import torch.nn as nn
import torch.nn.functional as F
from models.quant_utils import quantize_membrane_potential
import models.data_setup

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = models.data_setup.get_device()
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

# membrane potential update but quantized
def mem_update_quantized(ops, x, mem, spike, 
                         mem_m, mem_n, 
                         rounding="nearest", 
                         overflow="saturate"):
    """
    Membrane update with quantization BEFORE spike generation.
    This ensures spikes are based on quantized membrane values.
    """
    from models.spiking_model import act_fun
    
    # Compute new membrane
    mem = mem * decay * (1. - spike) + ops(x)
    
    # QUANTIZE BEFORE SPIKE DECISION!
    mem = quantize_membrane_potential(mem, mem_m, mem_n, rounding, overflow)
    
    # Now generate spike from quantized membrane
    spike = act_fun(mem)
    
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
    
    def __init__(self, layers=(784, 400, 10)):
        super().__init__()
        # Create layers dynamically
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))


    def forward(self, input, time_window=20):
        # input: [B, 1, 28, 28] with values in [0,1]
        B = input.size(0)

        # Membrane and spike states
        # h1_mem = torch.zeros(B, 400, device=device)
        # h1_spike = torch.zeros(B, 400, device=device)
        # h1_sumspike = torch.zeros(B, 400, device=device)

        # h2_mem = torch.zeros(B, 10, device=device)
        # h2_spike = torch.zeros(B, 10, device=device)
        # h2_sumspike = torch.zeros(B, 10, device=device)

        num_layers = len(self.layers)
        
        # Initialize membrane/spike states for each layer
        mem = [torch.zeros(B, layer.out_features, device=device) 
               for layer in self.layers]
        spike = [torch.zeros(B, layer.out_features, device=device) 
                 for layer in self.layers]
        sumspike = [torch.zeros(B, layer.out_features, device=device) 
                    for layer in self.layers]

        for _ in range(time_window):
            # Poisson/Bernoulli spike encoding from static image
            x = (input > torch.rand_like(input)).float()   # [B,1,28,28]
            x = x.view(B, -1)                              # flatten to [B,784]

            # # LIF updates through fully connected layers
            # h1_mem, h1_spike = mem_update(self.fc1, x, h1_mem, h1_spike)
            # h1_sumspike += h1_spike

            # h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem, h2_spike)
            # h2_sumspike += h2_spike

            # Process through all layers
            for i, layer in enumerate(self.layers):
                mem[i], spike[i] = mem_update(layer, x, mem[i], spike[i])
                sumspike[i] += spike[i]
                x = spike[i]  # Output of this layer is input to next
        
        outputs = sumspike[-1] / time_window  # Last layer output

        # outputs = h2_sumspike / time_window   # rate-coded outputs
        return outputs

class SMLP_MemQuant(nn.Module,):
    """SMLP with membrane quantization"""
    
    def __init__(self, 
                 quant_mem: bool = False,
                 mem_m: int = 2, 
                 mem_n: int = 4,
                 mem_rounding: str = "nearest",
                 mem_overflow: str = "saturate",
                 layers=(784, 400, 10) ):
        super().__init__()
        self.layer_sizes = layers
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
        self.quant_mem = quant_mem
        self.mem_m = mem_m
        self.mem_n = mem_n
        self.mem_rounding = mem_rounding
        self.mem_overflow = mem_overflow

    def forward(self, input, time_window=20):
        B = input.size(0)
        
        # Initialize membrane potentials, spikes, and sum spikes for each layer
        mem = []
        spike = []
        sumspike = []
        
        for i in range(1, len(self.layer_sizes)):  # Skip input layer
            layer_size = self.layer_sizes[i]
            mem.append(torch.zeros(B, layer_size, device=input.device))
            spike.append(torch.zeros(B, layer_size, device=input.device))
            sumspike.append(torch.zeros(B, layer_size, device=input.device))
        
        for _ in range(time_window):
            x = (input > torch.rand_like(input)).float()
            x = x.view(B, -1)
            
            # Process each layer
            for layer_idx, layer in enumerate(self.layers):
                if self.quant_mem:
                    mem[layer_idx], spike[layer_idx] = mem_update_quantized(
                        layer, x, mem[layer_idx], spike[layer_idx],
                        self.mem_m, self.mem_n,
                        self.mem_rounding, self.mem_overflow
                    )
                else:
                    mem[layer_idx], spike[layer_idx] = mem_update(
                        layer, x, mem[layer_idx], spike[layer_idx]
                    )
                
                sumspike[layer_idx] += spike[layer_idx]
                x = spike[layer_idx]  # Output of this layer becomes input to next
        
        # Return the output from the last layer
        outputs = sumspike[-1] / time_window
        return outputs

class Event_SMLP(nn.Module):

    def __init__(self, 
                 quant_mem: bool = False,
                 mem_m: int = 2, 
                 mem_n: int = 4,
                 mem_rounding: str = "nearest",
                 mem_overflow: str = "saturate",
                 track_extrema=False, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 400)
        self.fc2 = nn.Linear(400, 10)
        self.track_extrema = track_extrema
        
        self.quant_mem = quant_mem
        self.mem_m = mem_m
        self.mem_n = mem_n
        self.mem_rounding = mem_rounding
        self.mem_overflow = mem_overflow
        self.register_buffer("h1_mem",torch.zeros(400))
        self.register_buffer("h2_mem",torch.zeros(10))
        self.register_buffer("input_vec",torch.zeros(784))

    def reset_state(self):
        self.h1_mem = torch.zeros(400, device=device)  
        self.h2_mem = torch.zeros(10, device=device)

    def forward(self, input, time_window=20):
        B = input.size(0)
        outputs = torch.zeros(B, 10, device=input.device)
        
        for b in range(B):
            self.reset_state()
            sample = input[b]
            h2_sumspike = torch.zeros(10, device=input.device)
            
            for t in range(time_window):
                x = (sample > torch.rand_like(sample)).float()
                self.input_vec = x.view(-1)
                
                if self.track_extrema:
                    h2_spike, _, _ = self.process_time_step_collect_extrema()
                else:
                    h2_spike = self.process_time_step_fast()  
                
                h2_sumspike += h2_spike
            
            outputs[b] = h2_sumspike / time_window
        
        return outputs

    def process_time_step_fast(self):
        """Fast version without extrema tracking."""
        # Hidden layer
        for idx in torch.nonzero(self.input_vec, as_tuple=False).flatten():
            self.h1_mem += self.fc1.weight[:, idx]
        
        h1_spiked = act_fun(self.h1_mem)
        self.h1_mem[h1_spiked.bool()] = 0.0
        
        # Output layer
        for h in torch.nonzero(h1_spiked, as_tuple=False).flatten():
            self.h2_mem += self.fc2.weight[:, h]
        
        h2_spiked = act_fun(self.h2_mem)
        self.h2_mem[h2_spiked.bool()] = 0.0
        
        # Decay
        self.h1_mem *= decay
        self.h2_mem *= decay
        
        return h2_spiked

    def process_time_step_collect_extrema(self) -> tuple[torch.Tensor, float, float]:
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

class Event_SMLP_Quantized(nn.Module):
    """
    Event-driven SMLP with membrane quantization support.
    
    Args:
        quant_mem: Enable membrane quantization
        mem_m: Mantissa bits for quantization
        mem_n: Exponent bits for quantization
        mem_rounding: Rounding mode ('nearest', 'floor', 'ceil')
        mem_overflow: Overflow handling ('saturate', 'wrap')
        track_extrema: Track min/max membrane potentials (for analysis)
    """
    
    def __init__(self, 
                layers = (784, 400, 10),
                 quant_mem: bool = False,
                 mem_m: int = 2, 
                 mem_n: int = 4,
                 mem_rounding: str = "nearest",
                 mem_overflow: str = "saturate",
                 track_extrema: bool = False):
        super().__init__()
        self.layers = layers
        self.num_layers = len(layers) - 1  # Number of linear layers
        
        # Create dynamic linear layers
        self.fcs = nn.ModuleList([
            nn.Linear(layers[i], layers[i+1]) 
            for i in range(self.num_layers)
        ])
        
        # Quantization parameters
        self.quant_mem = quant_mem
        self.mem_m = mem_m
        self.mem_n = mem_n
        self.mem_rounding = mem_rounding
        self.mem_overflow = mem_overflow
        
        # Analysis flag
        self.track_extrema = track_extrema
        
        # State buffers (single sample) - create for each layer
        self.register_buffer("input_vec", torch.zeros(layers[0]))
        for i in range(self.num_layers):
            self.register_buffer(f"h{i+1}_mem", torch.zeros(layers[i+1]))
        
        self.mems = nn.ParameterList([
            nn.Parameter(torch.zeros(layers[i+1]), requires_grad=False)
            for i in range(self.num_layers)])

    def reset_state(self):
        """Reset membrane potentials for new sample."""
        device = self.input_vec.device
        for i in range(self.num_layers):
            mem = getattr(self, f"h{i+1}_mem")
            setattr(self, f"h{i+1}_mem", torch.zeros_like(mem, device=device))

    def forward(self, input, time_window=20):
        """
        Forward pass with event-driven processing.
        
        Args:
            input: Input images [B, 1, 28, 28]
            time_window: Number of timesteps to simulate
            
        Returns:
            outputs: Rate-coded outputs [B, 10]
        """
        B = input.size(0)
        output_size = self.layers[-1]
        outputs = torch.zeros(B, 10, device=input.device)
        
        # Process each sample independently
        for b in range(B):
            self.reset_state()
            sample = input[b]
            h_final_sumspike = torch.zeros(output_size, device=input.device)
            
            for t in range(time_window):
                # Poisson/Bernoulli spike encoding
                x = (sample > torch.rand_like(sample)).float()
                self.input_vec = x.view(-1)
                
                # Process timestep (with or without extrema tracking)
                if self.track_extrema:
                    h_final_spike, _, _ = self.process_time_step_collect_extrema()
                else:
                    h_final_spike = self.process_time_step_fast()
                
                h_final_sumspike += h_final_spike
            
            outputs[b] = h_final_sumspike / time_window
        
        return outputs

    def _quantize_if_enabled(self, membrane):
        """
        Apply quantization to membrane potential if enabled.
        
        Args:
            membrane: Membrane potential tensor
            
        Returns:
            Quantized membrane potential (or original if quantization disabled)
        """
        if self.quant_mem:
            return quantize_membrane_potential(
                membrane, 
                self.mem_m, 
                self.mem_n,
                self.mem_rounding,
                self.mem_overflow
            )
        else:
            return membrane

    def process_time_step_fast(self):
        """
        Fast event-driven processing with optional quantization.
        No extrema tracking.
        
       Returns:
            h_final_spiked: Output spike vector [output_size]
        """
        from models.spiking_model import act_fun, decay
        
        # ===== DECAY =====
        for i in range(self.num_layers):
            self.mems[i] *= decay
        
        # Process through all layers
        layer_input = self.input_vec
        
        for layer_idx in range(self.num_layers):
            mem = getattr(self, f"h{layer_idx+1}_mem")
            fc = self.fcs[layer_idx]
            
            # 1. Event-driven accumulation (only spiking inputs)
            for idx in torch.nonzero(layer_input, as_tuple=False).flatten():
                mem += fc.weight[:, idx]
            
                # 2. QUANTIZE membrane potential (if enabled)
                mem = self._quantize_if_enabled(mem)
                self.mems[layer_idx] = mem
                # setattr(self, f"h{layer_idx+1}_mem", mem)
            
            # 3. Generate spikes based on (quantized) membrane
            h_spiked = act_fun(mem)
            
            # 4. Reset spiking neurons
            mem[h_spiked.bool()] = 0.0
            self.mems[layer_idx] = mem
            # setattr(self, f"h{layer_idx+1}_mem", mem)
            
            # Output of this layer becomes input to next
            layer_input = h_spiked
        
        return layer_input  # Final layer spikes

    def process_time_step_collect_extrema(self):
        """
        Event-driven processing with extrema tracking and optional quantization.
        
        Returns:
            h2_spiked: Output spike vector [10]
            step_max: Maximum membrane potential (pre-quantization)
            step_min: Minimum membrane potential (pre-quantization)
        """
        from spiking_model import act_fun, decay
        
        step_max = -float("inf")
        step_min = +float("inf")
        
        # Process through all layers
        layer_input = self.input_vec
        
        # ===== DECAY =====
        for i in range(self.num_layers):
            self.mems[i] *= decay

        for layer_idx in range(self.num_layers):
            mem = getattr(self, f"h{layer_idx+1}_mem")
            fc = self.fcs[layer_idx]
            
            # 1. Event-driven accumulation
            for idx in torch.nonzero(layer_input, as_tuple=False).flatten():
                mem += fc.weight[:, idx]
                
                # Track extrema BEFORE quantization
                h_max = mem.max().item()
                h_min = mem.min().item()
                if h_max > step_max:
                    step_max = h_max
                if h_min < step_min:
                    step_min = h_min
            
                # 2. QUANTIZE membrane potential (if enabled)
                mem = self._quantize_if_enabled(mem)
                self.mems[layer_idx] = mem
                #setattr(self, f"h{layer_idx+1}_mem", mem)
            
            # 3. Generate spikes based on (quantized) membrane
            h_spiked = act_fun(mem)
            
            # 4. Reset spiking neurons
            mem[h_spiked.bool()] = 0.0
            self.mems[layer_idx] = mem
            # setattr(self, f"h{layer_idx+1}_mem", mem)
            
            # Output of this layer becomes input to next
            layer_input = h_spiked
        
        return layer_input, step_max, step_min



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

