"""
Event-Driven Spiking Neural Network - Educational Example

Key differences from time-stepped approach:
1. No fixed time_window loop
2. Events processed in chronological order (priority queue)
3. Exponential decay calculated on-demand with variable Δt
4. Computation only happens when spikes occur
5. Axonal delays between layers
"""

import torch
import heapq
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class SpikeEvent:
    """Represents a spike event occurring at a specific time"""
    time: float           # When the spike occurs
    neuron_id: int        # Which neuron fired
    layer_id: int         # Which layer (0=input, 1=hidden, 2=output)
    
    def __lt__(self, other):
        """For priority queue ordering (earliest time first)"""
        return self.time < other.time


class EventDrivenNeuron:
    """
    A single LIF (Leaky Integrate-and-Fire) neuron with event-driven dynamics
    """
    def __init__(self, neuron_id: int, layer_id: int, 
                 threshold: float = 0.5, tau: float = 5.0):
        """
        Args:
            neuron_id: Unique identifier for this neuron
            layer_id: Which layer this neuron belongs to
            threshold: Firing threshold
            tau: Membrane time constant (controls decay rate)
        """
        self.neuron_id = neuron_id
        self.layer_id = layer_id
        self.threshold = threshold
        self.tau = tau
        
        # State variables
        self.membrane = 0.0       # Current membrane potential
        self.last_update_time = 0.0  # When was membrane last computed
        self.last_spike_time = -float('inf')  # When did this neuron last fire
        self.refractory_period = 1.0  # Time after spike when neuron can't fire
        
    def apply_exponential_decay(self, current_time: float) -> float:
        """
        Apply exponential membrane decay from last_update_time to current_time.
        
        Formula: V(t) = V(t₀) × e^(-(t - t₀)/τ)
        
        This is the continuous-time version of your discrete: mem = mem * decay
        """
        delta_t = current_time - self.last_update_time
        
        if delta_t > 0:
            decay_factor = torch.exp(torch.tensor(-delta_t / self.tau))
            self.membrane = self.membrane * decay_factor.item()
        
        self.last_update_time = current_time
        return self.membrane
    
    def receive_spike(self, current_time: float, weight: float) -> bool:
        """
        Process an incoming spike with given synaptic weight.
        
        Returns:
            True if this causes the neuron to fire, False otherwise
        """
        # First, apply decay up to current time
        self.apply_exponential_decay(current_time)
        
        # Check if we're in refractory period
        if current_time - self.last_spike_time < self.refractory_period:
            return False
        
        # Add synaptic input (weighted spike)
        self.membrane += weight
        
        # Check for threshold crossing
        if self.membrane >= self.threshold:
            self.membrane = 0.0  # Reset after spike
            self.last_spike_time = current_time
            return True
        
        return False


class EventDrivenLayer:
    """A layer of event-driven neurons with weighted connections"""
    
    def __init__(self, n_inputs: int, n_neurons: int, layer_id: int,
                 axonal_delay: float = 1.0):
        """
        Args:
            n_inputs: Number of input connections per neuron
            n_neurons: Number of neurons in this layer
            layer_id: Layer identifier
            axonal_delay: Transmission delay to next layer (ms)
        """
        self.layer_id = layer_id
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.axonal_delay = axonal_delay
        
        # Create neurons
        self.neurons = [
            EventDrivenNeuron(i, layer_id) 
            for i in range(n_neurons)
        ]
        
        # Weight matrix: [n_neurons, n_inputs]
        # Random initialization (in practice, load trained weights)
        self.weights = torch.randn(n_neurons, n_inputs) * 0.1
    
    def process_spike(self, current_time: float, 
                     source_neuron_id: int) -> List[SpikeEvent]:
        """
        Process incoming spike from source_neuron_id.
        
        Returns:
            List of new spike events if any neurons fire
        """
        new_events = []
        
        # Each neuron receives weighted input from the source
        for target_idx, neuron in enumerate(self.neurons):
            weight = self.weights[target_idx, source_neuron_id].item()
            
            # Check if this weighted spike causes target to fire
            fired = neuron.receive_spike(current_time, weight)
            
            if fired:
                # Create new spike event with axonal delay
                spike_time = current_time + self.axonal_delay
                new_events.append(
                    SpikeEvent(spike_time, target_idx, self.layer_id)
                )
        
        return new_events


class EventDrivenSNN:
    """
    Simple 2-layer event-driven SNN: Input → Hidden → Output
    
    Compare this to your SMLP which does:
        for timestep in range(time_window):
            # Process all neurons at every step
    
    This only computes when spikes occur!
    """
    
    def __init__(self, n_inputs: int = 784, n_hidden: int = 400, 
                 n_outputs: int = 10):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        
        # Create layers
        self.hidden_layer = EventDrivenLayer(n_inputs, n_hidden, layer_id=1)
        self.output_layer = EventDrivenLayer(n_hidden, n_outputs, layer_id=2)
        
        # Event queue (priority queue sorted by time)
        self.event_queue: List[SpikeEvent] = []
        
        # Statistics tracking
        self.events_processed = 0
        self.spikes_generated = 0
    
    def add_input_spikes(self, spike_times: List[Tuple[float, int]]):
        """
        Add input spike events to the queue.
        
        Args:
            spike_times: List of (time, input_neuron_id) tuples
        """
        for time, neuron_id in spike_times:
            event = SpikeEvent(time, neuron_id, layer_id=0)
            heapq.heappush(self.event_queue, event)
    
    def run_simulation(self, max_time: float = 100.0) -> torch.Tensor:
        """
        Run event-driven simulation until max_time or queue empty.
        
        KEY DIFFERENCE: No time_window loop! Just process events as they occur.
        
        Returns:
            Output layer spike counts
        """
        output_spike_counts = torch.zeros(self.n_outputs)
        
        while self.event_queue:
            # Get earliest event
            event = heapq.heappop(self.event_queue)
            self.events_processed += 1
            
            # Stop if we've exceeded simulation time
            if event.time > max_time:
                break
            
            # Process spike based on which layer it came from
            new_events = []
            
            if event.layer_id == 0:  # Input spike
                new_events = self.hidden_layer.process_spike(
                    event.time, event.neuron_id
                )
            
            elif event.layer_id == 1:  # Hidden layer spike
                new_events = self.output_layer.process_spike(
                    event.time, event.neuron_id
                )
            
            elif event.layer_id == 2:  # Output spike
                output_spike_counts[event.neuron_id] += 1
            
            # Add any newly generated spikes to queue
            for new_event in new_events:
                heapq.heappush(self.event_queue, new_event)
                self.spikes_generated += 1
        
        return output_spike_counts
    
    def forward(self, input_image: torch.Tensor, 
                simulation_time: float = 100.0) -> torch.Tensor:
        """
        Process input image and return output spike counts.
        
        Args:
            input_image: [784] flattened MNIST image
            simulation_time: How long to simulate (ms)
        """
        # Reset queue and neurons
        self.event_queue = []
        self.events_processed = 0
        self.spikes_generated = 0
        
        # Reset all neurons
        for neuron in self.hidden_layer.neurons:
            neuron.membrane = 0.0
            neuron.last_update_time = 0.0
        for neuron in self.output_layer.neurons:
            neuron.membrane = 0.0
            neuron.last_update_time = 0.0
        
        # Generate input spikes using Poisson process
        # (similar to your: x = (input > torch.rand_like(input)).float())
        input_spikes = self._poisson_encode(input_image, simulation_time)
        
        # Add input spikes to event queue
        self.add_input_spikes(input_spikes)
        
        # Run simulation
        output_counts = self.run_simulation(simulation_time)
        
        return output_counts
    
    def _poisson_encode(self, image: torch.Tensor, 
                        duration: float) -> List[Tuple[float, int]]:
        """
        Convert pixel intensities to spike times using Poisson process.
        
        Higher intensity → more spikes → earlier average spike time
        """
        spikes = []
        
        for neuron_id, intensity in enumerate(image):
            # Skip if no intensity
            if intensity == 0:
                continue
            
            # Firing rate proportional to intensity
            rate = intensity.item() * 100  # Max ~100 Hz
            
            # Generate spike times for this neuron
            current_time = 0.0
            while current_time < duration:
                # Inter-spike interval from exponential distribution
                if rate > 0:
                    isi = torch.distributions.Exponential(rate).sample().item()
                    current_time += isi
                    
                    if current_time < duration:
                        spikes.append((current_time, neuron_id))
                else:
                    break
        
        return spikes


# ============================================================================
# COMPARISON: Event-Driven vs Time-Stepped
# ============================================================================

def compare_approaches():
    """
    Demonstrate the efficiency difference between approaches.
    """
    print("=" * 70)
    print("COMPARISON: Event-Driven vs Time-Stepped")
    print("=" * 70)
    
    # Simulate a sparse input (few spikes)
    sparse_input = torch.zeros(784)
    sparse_input[0] = 0.8
    sparse_input[100] = 0.6
    sparse_input[200] = 0.4
    
    # Event-driven
    print("\n📊 Event-Driven Approach:")
    event_snn = EventDrivenSNN()
    output = event_snn.forward(sparse_input, simulation_time=100.0)
    print(f"  Events processed: {event_snn.events_processed}")
    print(f"  Spikes generated: {event_snn.spikes_generated}")
    print(f"  Output: {output}")
    
    # Time-stepped (your current approach)
    print("\n📊 Time-Stepped Approach (your SMLP):")
    print(f"  With time_window=20:")
    print(f"    Updates computed: {20 * (784 + 400 + 10)} = {20 * 1194}")
    print(f"    (Every neuron, every timestep, regardless of spikes)")
    
    print("\n✨ Key Insight:")
    print(f"  Event-driven only computed when spikes occurred!")
    print(f"  Time-stepped computed even when nothing happened!")
    print(f"  Efficiency gain: ~{20 * 1194 / max(event_snn.events_processed, 1):.1f}x")


if __name__ == "__main__":
    compare_approaches()
    
    print("\n" + "=" * 70)
    print("KEY CONCEPTS DEMONSTRATED:")
    print("=" * 70)
    print("1. ✅ Priority queue for event ordering")
    print("2. ✅ Exponential decay with variable Δt")
    print("3. ✅ Only compute on spike events")
    print("4. ✅ Axonal delays between layers")
    print("5. ✅ Continuous time (not discrete steps)")