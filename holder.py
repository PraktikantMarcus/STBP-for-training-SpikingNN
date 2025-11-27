class Event_SMLP_Quantized(nn.Module):
    """
    Event-driven SMLP with membrane quantization support.
    
    Args:
        layers: List of layer sizes (e.g., [784, 400, 10])
        quant_mem: Enable membrane quantization
        mem_m: Mantissa bits for quantization
        mem_n: Exponent bits for quantization
        mem_rounding: Rounding mode ('nearest', 'floor', 'ceil')
        mem_overflow: Overflow handling ('saturate', 'wrap')
        track_extrema: Track min/max membrane potentials (for analysis)
    """
    
    def __init__(self, 
                 layers: list,
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
            input: Input images [B, 1, 28, 28] or [B, input_size]
            time_window: Number of timesteps to simulate
            
        Returns:
            outputs: Rate-coded outputs [B, output_size]
        """
        B = input.size(0)
        output_size = self.layers[-1]
        outputs = torch.zeros(B, output_size, device=input.device)
        
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
        """Apply quantization to membrane potential if enabled."""
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
            mem = getattr(self, f"h{i+1}_mem")
            setattr(self, f"h{i+1}_mem", mem * decay)
        
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
            setattr(self, f"h{layer_idx+1}_mem", mem)
            
            # 3. Generate spikes based on (quantized) membrane
            h_spiked = act_fun(mem)
            
            # 4. Reset spiking neurons
            mem[h_spiked.bool()] = 0.0
            setattr(self, f"h{layer_idx+1}_mem", mem)
            
            # Output of this layer becomes input to next
            layer_input = h_spiked
        
        return layer_input  # Final layer spikes

    def process_time_step_collect_extrema(self):
        """
        Event-driven processing with extrema tracking and optional quantization.
        
        Returns:
            h_final_spiked: Output spike vector
            step_max: Maximum membrane potential (pre-quantization)
            step_min: Minimum membrane potential (pre-quantization)
        """
        from models.spiking_model import act_fun, decay
        
        step_max = -float("inf")
        step_min = +float("inf")
        
        # Process through all layers
        layer_input = self.input_vec
        
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
            setattr(self, f"h{layer_idx+1}_mem", mem)
            
            # 3. Generate spikes based on (quantized) membrane
            h_spiked = act_fun(mem)
            
            # 4. Reset spiking neurons
            mem[h_spiked.bool()] = 0.0
            setattr(self, f"h{layer_idx+1}_mem", mem)
            
            # Output of this layer becomes input to next
            layer_input = h_spiked
        
        # ===== DECAY =====
        for i in range(self.num_layers):
            mem = getattr(self, f"h{i+1}_mem")
            setattr(self, f"h{i+1}_mem", mem * decay)
        
        return layer_input, step_max, step_min