#!/usr/bin/env python
"""
Plot weight distributions from checkpoint files with min/max highlighting
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from pathlib import Path

def load_checkpoint(checkpoint_path):
    """Load checkpoint and return model state dict"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    return checkpoint['net'], checkpoint.get('acc', 'N/A'), checkpoint.get('epoch', 'N/A')

def get_layer_weights(state_dict, layer_name):
    """Extract weights for a specific layer"""
    return state_dict[layer_name].cpu().numpy().flatten()

def plot_single_checkpoint(checkpoint_path, output_dir='plots'):
    """Plot weight distributions for all layers in a checkpoint"""
    
    state_dict, acc, epoch = load_checkpoint(checkpoint_path)
    checkpoint_name = Path(checkpoint_path).stem
    
    # Find all weight layers
    weight_keys = [k for k in state_dict.keys() if 'weight' in k]
    
    n_layers = len(weight_keys)
    fig, axes = plt.subplots(1, n_layers, figsize=(6*n_layers, 5))
    
    if n_layers == 1:
        axes = [axes]
    
    fig.suptitle(f'Weight Distributions: {checkpoint_name}\nEpoch: {epoch}, Accuracy: {acc:.2f}%' 
                 if isinstance(acc, float) else f'Weight Distributions: {checkpoint_name}',
                 fontsize=14, fontweight='bold')
    
    for idx, weight_key in enumerate(weight_keys):
        weights = get_layer_weights(state_dict, weight_key)
        
        # Calculate statistics
        w_min = weights.min()
        w_max = weights.max()
        w_mean = weights.mean()
        w_std = weights.std()
        
        # Plot histogram
        ax = axes[idx]
        n, bins, patches = ax.hist(weights, bins=50, alpha=0.7, 
                                    color='steelblue', edgecolor='black')
        
        # Highlight min and max
        ax.axvline(w_min, color='red', linestyle='--', linewidth=2, 
                  label=f'Min: {w_min:.4f}')
        ax.axvline(w_max, color='green', linestyle='--', linewidth=2, 
                  label=f'Max: {w_max:.4f}')
        ax.axvline(w_mean, color='orange', linestyle='-', linewidth=2, 
                  label=f'Mean: {w_mean:.4f}')
        
        # Add shading for extreme values
        ax.axvspan(w_min, w_min + (w_max - w_min)*0.05, 
                  alpha=0.2, color='red', label='Min region')
        ax.axvspan(w_max - (w_max - w_min)*0.05, w_max, 
                  alpha=0.2, color='green', label='Max region')
        
        # Labels and styling
        layer_name = weight_key.replace('.weight', '')
        ax.set_xlabel('Weight Value', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'{layer_name}\nShape: {state_dict[weight_key].shape}', 
                    fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add text box with statistics
        stats_text = f'μ={w_mean:.4f}\nσ={w_std:.4f}\nRange=[{w_min:.4f}, {w_max:.4f}]'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'weights_{checkpoint_name}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_checkpoint_comparison(checkpoint_paths, output_dir='plots'):
    """Compare weight distributions across multiple checkpoints"""
    
    n_checkpoints = len(checkpoint_paths)
    
    # Load all checkpoints
    checkpoints_data = []
    for cp_path in checkpoint_paths:
        state_dict, acc, epoch = load_checkpoint(cp_path)
        weight_keys = [k for k in state_dict.keys() if 'weight' in k]
        checkpoints_data.append({
            'path': cp_path,
            'name': Path(cp_path).stem,
            'state_dict': state_dict,
            'acc': acc,
            'epoch': epoch,
            'weight_keys': weight_keys
        })
    
    # Get first checkpoint's layers as reference
    reference_keys = checkpoints_data[0]['weight_keys']
    n_layers = len(reference_keys)
    
    # Create subplot grid: layers x checkpoints
    fig, axes = plt.subplots(n_layers, n_checkpoints, 
                            figsize=(5*n_checkpoints, 4*n_layers))
    
    if n_layers == 1 and n_checkpoints == 1:
        axes = np.array([[axes]])
    elif n_layers == 1:
        axes = axes.reshape(1, -1)
    elif n_checkpoints == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('Weight Distribution Comparison Across Checkpoints', 
                fontsize=16, fontweight='bold')
    
    for layer_idx, weight_key in enumerate(reference_keys):
        for cp_idx, cp_data in enumerate(checkpoints_data):
            ax = axes[layer_idx, cp_idx]
            
            if weight_key not in cp_data['state_dict']:
                ax.text(0.5, 0.5, 'Layer not found', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            weights = get_layer_weights(cp_data['state_dict'], weight_key)
            
            # Calculate statistics
            w_min = weights.min()
            w_max = weights.max()
            w_mean = weights.mean()
            
            # Plot histogram
            ax.hist(weights, bins=40, alpha=0.7, color='steelblue', edgecolor='black')
            
            # Highlight min and max
            ax.axvline(w_min, color='red', linestyle='--', linewidth=1.5, 
                      label=f'Min: {w_min:.3f}')
            ax.axvline(w_max, color='green', linestyle='--', linewidth=1.5, 
                      label=f'Max: {w_max:.3f}')
            
            # Title and labels
            if layer_idx == 0:
                title = f'{cp_data["name"]}\nEpoch: {cp_data["epoch"]}, Acc: {cp_data["acc"]:.2f}%'
                ax.set_title(title, fontsize=10, fontweight='bold')
            
            if cp_idx == 0:
                layer_name = weight_key.replace('.weight', '')
                ax.set_ylabel(f'{layer_name}\nCount', fontsize=10)
            
            if layer_idx == n_layers - 1:
                ax.set_xlabel('Weight Value', fontsize=10)
            
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save comparison plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'weights_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison: {output_path}")
    plt.close()

def plot_weight_evolution(checkpoint_paths, layer_name='fc1.weight', output_dir='plots'):
    """Plot how min/max/mean weights evolve across checkpoints"""
    
    epochs = []
    accuracies = []
    mins = []
    maxs = []
    means = []
    stds = []
    
    for cp_path in sorted(checkpoint_paths, 
                         key=lambda x: load_checkpoint(x)[2] if load_checkpoint(x)[2] != 'N/A' else 0):
        state_dict, acc, epoch = load_checkpoint(cp_path)
        
        if layer_name in state_dict:
            weights = get_layer_weights(state_dict, layer_name)
            
            epochs.append(epoch if epoch != 'N/A' else 0)
            accuracies.append(acc if acc != 'N/A' else 0)
            mins.append(weights.min())
            maxs.append(weights.max())
            means.append(weights.mean())
            stds.append(weights.std())
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Weight statistics over epochs
    ax1.plot(epochs, mins, 'r-o', label='Min', linewidth=2)
    ax1.plot(epochs, maxs, 'g-o', label='Max', linewidth=2)
    ax1.plot(epochs, means, 'b-o', label='Mean', linewidth=2)
    ax1.fill_between(epochs, 
                     np.array(means) - np.array(stds),
                     np.array(means) + np.array(stds),
                     alpha=0.3, label='±1 std')
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Weight Value', fontsize=12)
    ax1.set_title(f'Weight Statistics Evolution: {layer_name}', 
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy over epochs
    ax2.plot(epochs, accuracies, 'purple', marker='o', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Test Accuracy Evolution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save evolution plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'weights_evolution_{layer_name.replace(".", "_")}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved evolution: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Plot weight distributions from checkpoints')
    parser.add_argument('--checkpoint', type=str, help='Single checkpoint to analyze')
    parser.add_argument('--checkpoints', type=str, nargs='+', 
                       help='Multiple checkpoints to compare')
    parser.add_argument('--checkpoint_dir', type=str, 
                       help='Directory containing checkpoints (analyzes all .t7 files)')
    parser.add_argument('--output_dir', type=str, default='weight_plots',
                       help='Output directory for plots')
    parser.add_argument('--evolution', action='store_true',
                       help='Plot weight evolution over epochs')
    parser.add_argument('--layer', type=str, default='fc1.weight',
                       help='Layer name for evolution plot')
    
    args = parser.parse_args()
    
    # Collect checkpoint paths
    checkpoint_paths = []
    
    if args.checkpoint:
        checkpoint_paths = [args.checkpoint]
    elif args.checkpoints:
        checkpoint_paths = args.checkpoints
    elif args.checkpoint_dir:
        checkpoint_paths = list(Path(args.checkpoint_dir).glob('*.t7'))
        checkpoint_paths = [str(p) for p in checkpoint_paths]
    else:
        print("Error: Must provide --checkpoint, --checkpoints, or --checkpoint_dir")
        return
    
    if not checkpoint_paths:
        print("No checkpoints found!")
        return
    
    print(f"Found {len(checkpoint_paths)} checkpoint(s)")
    
    # Plot individual checkpoints
    if len(checkpoint_paths) == 1 or not args.evolution:
        for cp_path in checkpoint_paths:
            print(f"Plotting: {cp_path}")
            plot_single_checkpoint(cp_path, args.output_dir)
    
    # Plot comparison if multiple checkpoints
    if len(checkpoint_paths) > 1:
        print("\nCreating comparison plot...")
        plot_checkpoint_comparison(checkpoint_paths, args.output_dir)
    
    # Plot evolution if requested
    if args.evolution and len(checkpoint_paths) > 1:
        print(f"\nPlotting weight evolution for layer: {args.layer}")
        plot_weight_evolution(checkpoint_paths, args.layer, args.output_dir)
    
    print(f"\nAll plots saved to: {args.output_dir}/")

if __name__ == '__main__':
    main()
