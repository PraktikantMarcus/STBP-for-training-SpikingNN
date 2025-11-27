"""
Test script: Compare Event_SMLP with and without quantization

Tests different quantization settings and measures impact on:
1. Accuracy
2. Membrane potential distributions
3. Spike patterns
"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torchvision
import numpy as np
import matplotlib.pyplot as plt

# Import both versions
from spiking_model import *



def test_quantization_impact(n_samples=100):
    """
    Compare quantized vs non-quantized Event_SMLP.
    
    Args:
        n_samples: Number of samples to test
    """
    print("=" * 80)
    print("Event_SMLP Quantization Comparison")
    print("=" * 80)
    print()
    
    # Load dataset
    print("Loading dataset...")
    test_dataset = torchvision.datasets.MNIST(
        root='./raw/',
        train=False,
        download=False,
        transform=transforms.ToTensor()
    )
    subset = Subset(test_dataset, range(n_samples))
    test_loader = DataLoader(subset, batch_size=10, shuffle=False)
    
    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load("./checkpoint/ckptspiking_model.t7", 
                           map_location='cpu', 
                           weights_only=True)
    model_state = checkpoint['net']
    print(f"✓ Checkpoint loaded (epoch={checkpoint.get('epoch')}, acc={checkpoint.get('acc'):.2f}%)")
    print()
    
    # Test configurations
    configs = [
        {"name": "Full Precision", "quant_mem": False},
        {"name": "Q(2,4)", "quant_mem": True, "mem_m": 2, "mem_n": 4},
        {"name": "Q(3,5)", "quant_mem": True, "mem_m": 3, "mem_n": 5},
        {"name": "Q(4,6)", "quant_mem": True, "mem_m": 4, "mem_n": 6},
        {"name": "Q(1,3)", "quant_mem": True, "mem_m": 1, "mem_n": 3},  # Very aggressive
    ]
    
    results = []
    
    for config in configs:
        print("-" * 80)
        print(f"Testing: {config['name']}")
        print("-" * 80)
        
        # Create model
        model = Event_SMLP_Quantized(**{k: v for k, v in config.items() if k != 'name'})
        model.load_state_dict(model_state, strict=False)
        model.eval()
        model.to('cpu')
        
        # Run inference
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs, time_window=20)
                predicted = outputs.argmax(dim=1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = 100.0 * correct / total
        
        print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")
        
        results.append({
            "config": config['name'],
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        })
        print()
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"{'Configuration':<20} {'Accuracy':<12} {'Degradation':<15}")
    print("-" * 80)
    
    baseline_acc = results[0]['accuracy']
    for r in results:
        degradation = baseline_acc - r['accuracy']
        print(f"{r['config']:<20} {r['accuracy']:>6.2f}%     {degradation:>+6.2f}%")
    print("=" * 80)


def analyze_membrane_distributions(n_samples=10):
    """
    Analyze how quantization affects membrane potential distributions.
    
    Args:
        n_samples: Number of samples to analyze
    """
    print("\n" + "=" * 80)
    print("Membrane Potential Distribution Analysis")
    print("=" * 80)
    print()
    
    # Load data and checkpoint
    test_dataset = torchvision.datasets.MNIST(
        root='./raw/',
        train=False,
        download=False,
        transform=transforms.ToTensor()
    )
    subset = Subset(test_dataset, range(n_samples))
    test_loader = DataLoader(subset, batch_size=1, shuffle=False)
    
    checkpoint = torch.load("./checkpoint/ckptspiking_model.t7", 
                           map_location='cpu', 
                           weights_only=True)
    model_state = checkpoint['net']
    
    # Compare full precision vs quantized
    models = {
        "Full Precision": Event_SMLP_Quantized(quant_mem=False),
        "Quantized Q(2,4)": Event_SMLP_Quantized(quant_mem=True, mem_m=2, mem_n=4)
    }
    
    membrane_values = {name: [] for name in models.keys()}
    
    for name, model in models.items():
        model.load_state_dict(model_state, strict=False)
        model.eval()
        model.to('cpu')
        
        print(f"Collecting membrane values for {name}...")
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                model.reset_state()
                sample = inputs[0]
                
                for t in range(20):  # time_window
                    x = (sample > torch.rand_like(sample)).float()
                    model.input_vec = x.view(-1)
                    
                    # Capture membrane values before spike
                    membrane_values[name].append(model.h1_mem.clone().cpu().numpy())
                    
                    # Process timestep
                    model.process_time_step_fast()
    
    # Plot distributions
    print("Plotting distributions...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    for idx, (name, values) in enumerate(membrane_values.items()):
        all_values = np.concatenate(values)
        
        axes[idx].hist(all_values, bins=100, alpha=0.7, edgecolor='black')
        axes[idx].set_title(f"{name}\n(n={len(all_values)} values)")
        axes[idx].set_xlabel("Membrane Potential")
        axes[idx].set_ylabel("Frequency")
        axes[idx].axvline(0.5, color='red', linestyle='--', label='Threshold')
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig("membrane_distribution_comparison.png", dpi=150)
    print("✓ Saved: membrane_distribution_comparison.png")
    print()


def test_single_sample_detail():
    """
    Detailed analysis of a single sample with/without quantization.
    Shows exact membrane values and spike patterns.
    """
    print("=" * 80)
    print("Single Sample Detailed Analysis")
    print("=" * 80)
    print()
    
    # Load one sample
    test_dataset = torchvision.datasets.MNIST(
        root='./raw/',
        train=False,
        download=False,
        transform=transforms.ToTensor()
    )
    sample_img, sample_label = test_dataset[0]
    
    # Load checkpoint
    checkpoint = torch.load("./checkpoint/ckptspiking_model.t7", 
                           map_location='cpu', 
                           weights_only=True)
    model_state = checkpoint['net']
    
    print(f"Sample label: {sample_label}")
    print()
    
    # Test both versions
    for name, quant_enabled in [("Full Precision", False), ("Quantized Q(2,4)", True)]:
        print(f"--- {name} ---")
        
        model = Event_SMLP_Quantized(quant_mem=quant_enabled, mem_m=2, mem_n=4)
        model.load_state_dict(model_state, strict=False)
        model.eval()
        
        output = model(sample_img.unsqueeze(0), time_window=20)
        prediction = output.argmax(dim=1).item()
        
        print(f"Prediction: {prediction}")
        print(f"Output rates: {output[0].cpu().numpy()}")
        print(f"Predicted correctly: {prediction == sample_label}")
        print()


if __name__ == '__main__':
    import sys
    
    print("Event_SMLP Quantization Testing")
    print()
    
    # Run tests
    test_quantization_impact(n_samples=1000)
    
    # Detailed analysis
    if len(sys.argv) > 1 and sys.argv[1] == '--detailed':
        analyze_membrane_distributions(n_samples=50)
        test_single_sample_detail()
