"""
Simple Event-Driven SMLP Accuracy Test
Tests on a small subset because Python implementation is slow
"""

import torch
import time
from event_driven_smlp import EventDrivenSMLP
from spiking_model import *
import data_setup
@torch.inference_mode()
def test_event_driven_accuracy(n_samples=10):
    """
    Test event-driven SMLP on a small number of samples.
    
    Args:
        n_samples: Number of test samples (keep small! e.g., 10-20)
    """
    print("=" * 70)
    print(f"Event-Driven SMLP Accuracy Test ({n_samples} samples)")
    print("=" * 70)
    print("\n⚠️  WARNING: Event-driven is SLOW in Python!")
    print(f"   Testing only {n_samples} samples (will take ~{n_samples * 2}-{n_samples * 10} seconds)")
    print("=" * 70)
    
    device = data_setup.get_device()
    
    # Load checkpoint
    try:
        ckpt = torch.load("./checkpoint/ckptspiking_model.t7", map_location=device)
        print(f"\n✅ Loaded checkpoint (original accuracy: {ckpt['acc']:.2f}%)")
    except FileNotFoundError:
        print("\n❌ Checkpoint not found!")
        return
    
    # Create and load event-driven model
    print("\n📊 Creating event-driven model...")
    # event_model = EventDrivenSMLP().to(device)
    event_model = Event_SMLP().to(device)
    # event_model.load_state_dict(ckpt['net'])
    event_model.load_state_dict(ckpt['net'], strict=False)
    event_model.eval()
    print("✅ Loaded weights into event-driven model")
    
    # Test data
    test_loader = data_setup.get_test_loader(batch_size=1, data_path="./raw/")
    
    # Run tests
    print(f"\n🔬 Testing on {n_samples} samples...")
    correct = 0
    total = 0
    total_time = 0
    
    for i, (images, labels) in enumerate(test_loader):
        if i >= n_samples:
            break
        
        images = images.to(device)
        true_label = labels[0].item()
        
        # Time this inference
        start = time.time()
        output = event_model(images, time_window=20)
        elapsed = time.time() - start
        total_time += elapsed
        
        predicted = output.argmax(dim=1).item()
        is_correct = (predicted == true_label)
        correct += int(is_correct)
        total += 1
        
        # Print progress
        status = "✓" if is_correct else "✗"
        print(f"  Sample {i+1}/{n_samples}: {status} (true={true_label}, pred={predicted}, {elapsed:.2f}s)")
    
    # Results
    accuracy = 100.0 * correct / total
    avg_time = total_time / total
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Event-Driven Accuracy: {accuracy:.1f}% ({correct}/{total})")
    print(f"Average time per image: {avg_time:.2f}s")
    print(f"Total time: {total_time:.1f}s")
    print(f"\nOriginal SMLP Accuracy: {ckpt['acc']:.2f}%")
    
    # Comparison note
    if abs(accuracy - ckpt['acc']) < 10:
        print("\n✅ Accuracy is reasonably close to original!")
    else:
        print("\n⚠️  Accuracy differs significantly - may need tuning")

@torch.inference_mode()
def compare_with_original(n_samples=5):
    """
    Side-by-side comparison of original vs event-driven.
    
    Args:
        n_samples: Number of samples to compare (keep very small!)
    """
    print("\n" + "=" * 70)
    print(f"COMPARISON: Original vs Event-Driven ({n_samples} samples)")
    print("=" * 70)
    
    device = data_setup.get_device()
    
    # Load models
    try:
        ckpt = torch.load("./checkpoint/ckptspiking_model.t7", map_location=device)
    except FileNotFoundError:
        print("❌ Checkpoint not found!")
        return
    
    original_model = SMLP().to(device)
    original_model.load_state_dict(ckpt['net'])
    original_model.eval()
    
    # event_model = EventDrivenSMLP().to(device)
    event_model = Event_SMLP().to(device)
    # event_model.load_state_dict(ckpt['net'])
    event_model.load_state_dict(ckpt['net'], strict= False)
    event_model.eval()
    
    
    print("\n✅ Both models loaded\n")
    # Test
    test_loader = data_setup.get_test_loader(batch_size=1, data_path="./raw/")
    
    print(f"{'Sample':<8} {'True':<6} {'Original':<10} {'Event-Driven':<12} {'Match':<6} {'Orig_Time':<10} {'Event_Time':<10}")
    print("-" * 70)
    
    for i, (images, labels) in enumerate(test_loader):
        if i >= n_samples:
            break
        
        images = images.to(device)
        true_label = labels[0].item()
        
        # Original
        with torch.no_grad():
            start = time.time()
            output_orig = original_model(images, time_window=20)
            time_orig = time.time() - start
        pred_orig = output_orig.argmax(dim=1).item()
        
        # Event-driven
        start = time.time()
        output_event = event_model(images, time_window=20)
        time_event = time.time() - start
        pred_event = output_event.argmax(dim=1).item()
        
        # Compare
        match = "✓" if pred_orig == pred_event else "✗"
        
        print(f"{i+1:<8} {true_label:<6} {pred_orig:<10} {pred_event:<12} {match:<6} {time_orig:<10.2f} {time_event:<10.2f}")
    

@torch.inference_mode()
def quick_test():
    """Just test on 1 sample """
    print("=" * 70)
    print("QUICK TEST: Single Sample")
    print("=" * 70)
    
    device = data_setup.get_device()
    
    try:
        ckpt = torch.load("./checkpoint/ckptspiking_model.t7", map_location=device)
        # event_model = EventDrivenSMLP().to(device)
        event_model = Event_SMLP().to(device)
        # event_model.load_state_dict(ckpt['net'])
        event_model.load_state_dict(ckpt['net'], strict=False)
        event_model.eval()
        
        test_loader = data_setup.get_test_loader(batch_size=1, data_path="./raw/")
        images, labels = next(iter(test_loader))
        
        print(f"\nTrue label: {labels[0].item()}")
        print("Running inference...")
        
        start = time.time()
        output = event_model(images.to(device), time_window=20) # reasonable time_window for EventDrivenSMLP=0.7
        elapsed = time.time() - start
        
        predicted = output.argmax(dim=1).item()
        
        print(f"Predicted: {predicted}")
        print(f"Correct: {predicted == labels[0].item()}")
        print(f"Time: {elapsed:.2f}s")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 70)
    print("EVENT-DRIVEN SMLP TEST")
    print("=" * 70)
    print("\nChoose a test:")
    print("  1. Quick test (1 sample)")
    print("  2. Small accuracy test (10 samples)")
    print("  3. Comparison with original (5 samples)")
    print("  4. Custom number of samples")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nEnter choice (1-4, default=1): ").strip() or "1"
    
    if choice == "1":
        quick_test()
    elif choice == "2":
        test_event_driven_accuracy(n_samples=10)
    elif choice == "3":
        compare_with_original(n_samples=5)
    elif choice == "4":
        n = int(input("How many samples to test? "))
        test_event_driven_accuracy(n_samples=n)
    else:
        print("Invalid choice, running quick test...")
        quick_test()
    