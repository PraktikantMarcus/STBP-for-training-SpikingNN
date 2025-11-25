class SMLP_MemQuant(nn.Module):
    """SMLP with membrane quantization - FIXED VERSION."""
    def __init__(self, 
                 quant_mem: bool = False,
                 mem_m: int = 2, 
                 mem_n: int = 4,
                 mem_rounding: str = "nearest",
                 mem_overflow: str = "saturate",
                 layers=(784, 400, 10)):
        super().__init__()
        self.layer_sizes = layers  # Store layer sizes
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