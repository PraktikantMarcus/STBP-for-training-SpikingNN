#!/usr/bin/env python3
"""
Export trained SNN weights to Verilog-compatible format for hardware accelerator.

This script:
1. Loads a trained PyTorch checkpoint
2. Quantizes weights to Q1.3 fixed-point format
3. Exports to hex/binary files for Verilog $readmemh/$readmemb

Usage:
    python export_weights.py --checkpoint ./checkpoint/ckpt_784_400_10.t7 --output ./weights/
"""

import torch
import numpy as np
import argparse
import os
from pathlib import Path


def real_to_q13(value: float) -> int:
    """Convert real number to Q1.3 fixed-point (5-bit signed)."""
    # Scale by 2^3 = 8
    scaled = round(value * 8.0)
    
    # Clamp to Q1.3 range [-16, 15]
    clamped = max(-16, min(15, scaled))
    
    # Convert to 5-bit unsigned representation for Verilog
    if clamped < 0:
        return (32 + clamped) & 0x1F  # Two's complement
    else:
        return clamped & 0x1F


def q13_to_real(q_val: int) -> float:
    """Convert Q1.3 fixed-point back to real (for verification)."""
    # Convert to Python int to avoid numpy uint8 overflow issues
    q_val = int(q_val)
    
    # Sign extend from 5 bits
    if q_val & 0x10:  # Negative (bit 4 is set)
        signed_val = q_val - 32  # Two's complement
    else:
        signed_val = q_val
    return signed_val / 8.0


def quantize_tensor_q13(tensor: torch.Tensor) -> np.ndarray:
    """Quantize a tensor to Q1.3 format."""
    flat = tensor.detach().cpu().numpy().flatten()
    q_values = np.array([real_to_q13(v) for v in flat], dtype=np.uint8)
    return q_values


def export_layer_weights(weights: torch.Tensor, 
                         output_dir: Path, 
                         layer_name: str,
                         format: str = 'hex') -> dict:
    """
    Export layer weights to hardware-compatible format.
    
    Returns statistics about the quantization.
    """
    # Get dimensions
    out_features, in_features = weights.shape
    
    # Quantize
    q_weights = quantize_tensor_q13(weights)
    
    # Reshape back for organized output
    q_weights_2d = q_weights.reshape(out_features, in_features)
    
    # Statistics
    original_flat = weights.detach().cpu().numpy().flatten()
    q_flat_real = np.array([q13_to_real(v) for v in q_weights])
    
    stats = {
        'layer': layer_name,
        'shape': (out_features, in_features),
        'total_weights': out_features * in_features,
        'original_min': float(original_flat.min()),
        'original_max': float(original_flat.max()),
        'original_mean': float(original_flat.mean()),
        'quantized_min': float(q_flat_real.min()),
        'quantized_max': float(q_flat_real.max()),
        'quantized_mean': float(q_flat_real.mean()),
        'quant_error_rms': float(np.sqrt(np.mean((original_flat - q_flat_real)**2))),
    }
    
    # Count clipped values
    clipped_high = np.sum(original_flat > 1.875)
    clipped_low = np.sum(original_flat < -2.0)
    stats['clipped_high'] = int(clipped_high)
    stats['clipped_low'] = int(clipped_low)
    stats['clipped_percent'] = 100.0 * (clipped_high + clipped_low) / len(original_flat)
    
    # Export files
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if format in ['hex', 'both']:
        # Hex format for $readmemh
        hex_file = output_dir / f"{layer_name}_weights.hex"
        with open(hex_file, 'w') as f:
            f.write(f"// {layer_name} weights: {out_features}x{in_features}\n")
            f.write(f"// Format: Q1.3 (5-bit signed)\n")
            f.write(f"// Address = row * {in_features} + col\n")
            for row in range(out_features):
                for col in range(in_features):
                    f.write(f"{q_weights_2d[row, col]:02X}\n")
        stats['hex_file'] = str(hex_file)
    
    if format in ['bin', 'both']:
        # Binary format for $readmemb
        bin_file = output_dir / f"{layer_name}_weights.bin"
        with open(bin_file, 'w') as f:
            f.write(f"// {layer_name} weights: {out_features}x{in_features}\n")
            f.write(f"// Format: Q1.3 (5-bit signed)\n")
            for row in range(out_features):
                for col in range(in_features):
                    f.write(f"{q_weights_2d[row, col]:05b}\n")
        stats['bin_file'] = str(bin_file)
    
    # Also export as raw bytes for direct memory initialization
    raw_file = output_dir / f"{layer_name}_weights.raw"
    q_weights.astype(np.uint8).tofile(raw_file)
    stats['raw_file'] = str(raw_file)
    
    return stats


def export_verilog_include(stats_list: list, output_dir: Path):
    """Generate Verilog include file with weight initialization."""
    
    include_file = output_dir / "weight_init.vh"
    with open(include_file, 'w') as f:
        f.write("// Auto-generated weight initialization\n")
        f.write("// Include this in your testbench or synthesis\n\n")
        
        for stats in stats_list:
            layer = stats['layer']
            shape = stats['shape']
            
            f.write(f"// {layer}: {shape[0]}x{shape[1]} weights\n")
            f.write(f"// RMS quantization error: {stats['quant_error_rms']:.6f}\n")
            f.write(f"initial begin\n")
            f.write(f"    $readmemh(\"{layer}_weights.hex\", {layer}_weights);\n")
            f.write(f"end\n\n")
    
    print(f"Generated Verilog include: {include_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Export SNN weights to hardware format"
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to PyTorch checkpoint (.t7 file)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./weights",
        help="Output directory for weight files"
    )
    parser.add_argument(
        "--format", "-f",
        type=str,
        choices=['hex', 'bin', 'both'],
        default='both',
        help="Output format"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[784, 400, 10],
        help="Network layer sizes"
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output)
    
    print("=" * 60)
    print("SNN Weight Export Tool")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output: {output_dir}")
    print(f"Format: Q1.3 (5-bit signed)")
    print(f"Network: {' -> '.join(map(str, args.layers))}")
    print("=" * 60)
    
    # Load checkpoint
    print("\nLoading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['net']
    
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Accuracy: {checkpoint.get('acc', 'N/A'):.2f}%")
    
    # Process each layer
    stats_list = []
    
    for name, param in state_dict.items():
        if 'weight' in name:
            print(f"\nProcessing {name}...")
            print(f"  Shape: {list(param.shape)}")
            
            # Determine layer name
            if 'layers.0' in name or 'fc1' in name:
                layer_name = 'layer1'
            elif 'layers.1' in name or 'fc2' in name:
                layer_name = 'layer2'
            else:
                layer_name = name.replace('.', '_')
            
            # Export
            stats = export_layer_weights(param, output_dir, layer_name, args.format)
            stats_list.append(stats)
            
            # Print stats
            print(f"  Original range: [{stats['original_min']:.4f}, {stats['original_max']:.4f}]")
            print(f"  Quantized range: [{stats['quantized_min']:.4f}, {stats['quantized_max']:.4f}]")
            print(f"  RMS error: {stats['quant_error_rms']:.6f}")
            print(f"  Clipped: {stats['clipped_percent']:.2f}%")
    
    # Generate Verilog include
    export_verilog_include(stats_list, output_dir)
    
    # Summary
    print("\n" + "=" * 60)
    print("Export Summary")
    print("=" * 60)
    
    total_weights = sum(s['total_weights'] for s in stats_list)
    total_bytes = total_weights  # 5-bit packed would be less
    
    print(f"Total weights: {total_weights:,}")
    print(f"Storage (unpacked): {total_bytes:,} bytes ({total_bytes/1024:.1f} KB)")
    print(f"Storage (packed 5-bit): {(total_weights * 5 + 7) // 8:,} bytes")
    
    print(f"\nOutput files:")
    for stats in stats_list:
        print(f"  {stats['layer']}:")
        if 'hex_file' in stats:
            print(f"    - {stats['hex_file']}")
        if 'bin_file' in stats:
            print(f"    - {stats['bin_file']}")
        print(f"    - {stats['raw_file']}")
    
    print("\n✓ Export complete!")


if __name__ == "__main__":
    main()