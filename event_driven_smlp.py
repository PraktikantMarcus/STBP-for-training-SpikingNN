"""
Event-Driven SMLP - Direct comparison to original SMLP

This implementation keeps the same structure as your SMLP:
- Inherits from nn.Module
- Uses same fc1 and fc2 Linear layers
- Can load your trained checkpoint directly
- But processes spikes in chronological order instead of fixed timesteps

Key differences from time-stepped SMLP:
1. No fixed time_window loop
2. Tracks event times explicitly
3. Exponential decay with variable Δt
4. Only computes when spikes occur
"""

import torch
import torch.nn as nn
import math
from typing import List, Tuple
from dataclasses import dataclass
import data_setup

# Use same global parameters as original
device = data_setup.get_device()
thresh = 0.5  # neuronal threshold
decay = 0.2   # decay constant from original

# Calculate tau from your decay constant
# decay = e^(-1/tau) when Δt=1
# Therefore: tau = -1/ln(decay)
tau = -1.0 / math.log(decay)  # ≈ 0.621


@dataclass
class SpikeEvent:
    """Simple spike event structure"""
    time: float       # When the spike occurs (ms)
    neuron_idx: int   # Which neuron fired
    layer: int        # 0=input, 1=hidden, 2=output
    
    def __lt__(self, other):
        return self.time < other.time


class EventDrivenSMLP(nn.Module):
    """
    Event-driven version of SMLP with architecture:
    784 -> 400 -> 10
    
    Can load weights directly from trained SMLP checkpoint!
    """
    
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(28*28, 400)   # 784 -> 400
        self.fc2 = nn.Linear(400, 10)      # 400 -> 10
        
        # Event-driven specific parameters
        self.tau = tau
        self.thresh = thresh
        self.refractory_period = 0.0  # Set to 0 to match original behavior
        
    def forward(self, input, time_window=0.01):
        """
        Forward pass that looks similar to original but uses event-driven processing.
        
        Args:
            input: [B, 1, 28, 28] batch of images
            time_window: Simulation duration (treated as milliseconds)
            
        Returns:
            outputs: [B, 10] rate-coded output (like original)
        """
        B = input.size(0)
        
        # Process each sample in batch
        all_outputs = []
        
        for b in range(B):
            sample = input[b]  # [1, 28, 28]
            output = self._process_single_sample(sample, time_window)
            all_outputs.append(output)
        
        return torch.stack(all_outputs)  # [B, 10]
    
    def _process_single_sample(self, image, time_window):
        """
        Process a single image using event-driven dynamics.
        This is where the magic happens!
        """
        # Initialize membrane potentials and spike counts
        h1_mem = torch.zeros(400, device=device)
        h1_spike_counts = torch.zeros(400, device=device)
        h1_last_spike_time = torch.full((400,), -float('inf'), device=device)
        h1_last_update_time = torch.zeros(400, device=device)
        
        h2_mem = torch.zeros(10, device=device)
        h2_spike_counts = torch.zeros(10, device=device)
        h2_last_spike_time = torch.full((10,), -float('inf'), device=device)
        h2_last_update_time = torch.zeros(10, device=device)
        
        # Generate input spike events using Poisson encoding
        input_events = self._generate_input_spikes(image, time_window)
        
        # Sort events by time (priority queue in simple form)
        input_events.sort(key=lambda e: e.time)
        
        # Process events chronologically
        for event in input_events:
            if event.layer == 0:  # Input spike
                # Propagate to hidden layer (layer 1)
                new_events = self._process_input_spike(
                    event, h1_mem, h1_spike_counts, 
                    h1_last_spike_time, h1_last_update_time
                )
                
                # Add new hidden layer spikes to process
                input_events.extend(new_events)
                input_events.sort(key=lambda e: e.time)
            
            elif event.layer == 1:  # Hidden layer spike
                # Propagate to output layer (layer 2)
                self._process_hidden_spike(
                    event, h2_mem, h2_spike_counts,
                    h2_last_spike_time, h2_last_update_time
                )
        
        # Return rate-coded output (same as original)
        return h2_spike_counts / time_window
    
    def _generate_input_spikes(self, image, duration):
        """
        Generate input spike events from image using Poisson encoding.
        
        This replaces the time_window loop with:
            x = (input > torch.rand_like(input)).float()
        
        Instead of generating spikes at each timestep, we generate them
        with continuous times.
        """
        events = []
        image_flat = image.view(-1)  # [784]
        
        # For each input pixel, generate spikes based on intensity
        for pixel_idx in range(len(image_flat)):
            intensity = image_flat[pixel_idx].item()
            
            if intensity < 0.01:  # Skip very low intensities
                continue
            
            # Generate spike times for this pixel
            # Higher intensity = more spikes
            n_spikes = int(duration * intensity * 2) 
            
            for _ in range(n_spikes):
                # Uniform random time within duration
                spike_time = torch.rand(1).item() * duration
                events.append(SpikeEvent(spike_time, pixel_idx, layer=0))
        
        return events
    
    def _apply_decay(self, mem, current_time, last_update_time):
        """
        Apply exponential decay: V(t) = V(t0) * e^(-(t-t0)/tau)
        
        This replaces: mem = mem * decay
        But handles variable time intervals!
        """
        delta_t = current_time - last_update_time
        
        if delta_t > 0:
            decay_factor = torch.exp(-delta_t / self.tau)
            mem = mem * decay_factor
        
        return mem
    
    def _process_input_spike(self, event, h1_mem, h1_spike_counts, 
                            h1_last_spike_time, h1_last_update_time):
        """
        Process input spike and propagate to hidden layer.
        
        This replaces:
            h1_mem, h1_spike = mem_update(self.fc1, x, h1_mem, h1_spike)
        """
        new_events = []
        current_time = event.time
        input_idx = event.neuron_idx
        
        # Get weights from input neuron to all hidden neurons
        # fc1.weight is [400, 784], so we want column input_idx
        weights = self.fc1.weight[:, input_idx]  # [400]
        bias_contribution = self.fc1.bias / 784  # Distribute bias across inputs
        
        # Update each hidden neuron
        for h_idx in range(400):
            # Apply decay since last update
            h1_mem[h_idx] = self._apply_decay(
                h1_mem[h_idx], current_time, h1_last_update_time[h_idx]
            )
            h1_last_update_time[h_idx] = current_time
            
            # Check refractory period
            if current_time - h1_last_spike_time[h_idx] < self.refractory_period:
                continue
            
            # Add weighted input (this replaces: mem = mem + ops(x))
            h1_mem[h_idx] = h1_mem[h_idx] + weights[h_idx] + bias_contribution[h_idx]
            
            # Check for spike (this replaces: spike = act_fun(mem))
            if h1_mem[h_idx] >= self.thresh:
                # Spike!
                h1_mem[h_idx] = 0.0  # Reset
                h1_spike_counts[h_idx] += 1
                h1_last_spike_time[h_idx] = current_time
                
                # Create event for this spike (minimal delay)
                new_events.append(SpikeEvent(
                    current_time + 0.1,  # Small propagation delay
                    h_idx,
                    layer=1
                ))
        
        return new_events
    
    def _process_hidden_spike(self, event, h2_mem, h2_spike_counts,
                             h2_last_spike_time, h2_last_update_time):
        """
        Process hidden layer spike and propagate to output.
        
        This replaces:
            h2_mem, h2_spike = mem_update(self.fc2, h1_spike, h2_mem, h2_spike)
        """
        current_time = event.time
        hidden_idx = event.neuron_idx
        
        # Get weights from hidden neuron to all output neurons
        weights = self.fc2.weight[:, hidden_idx]  # [10]
        bias_contribution = self.fc2.bias / 400
        
        # Update each output neuron
        for out_idx in range(10):
            # Apply decay since last update
            h2_mem[out_idx] = self._apply_decay(
                h2_mem[out_idx], current_time, h2_last_update_time[out_idx]
            )
            h2_last_update_time[out_idx] = current_time
            
            # Check refractory period
            if current_time - h2_last_spike_time[out_idx] < self.refractory_period:
                continue
            
            # Add weighted input
            h2_mem[out_idx] = h2_mem[out_idx] + weights[out_idx] + bias_contribution[out_idx]
            
            # Check for spike
            if h2_mem[out_idx] >= self.thresh:
                # Spike!
                h2_mem[out_idx] = 0.0
                h2_spike_counts[out_idx] += 1
                h2_last_spike_time[out_idx] = current_time
