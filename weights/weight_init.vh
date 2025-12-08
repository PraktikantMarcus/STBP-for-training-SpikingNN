// Auto-generated weight initialization
// Include this in your testbench or synthesis

// layer1: 400x784 weights
// RMS quantization error: 0.033699
initial begin
    $readmemh("layer1_weights.hex", layer1_weights);
end

// layer2: 10x400 weights
// RMS quantization error: 0.035555
initial begin
    $readmemh("layer2_weights.hex", layer2_weights);
end

