import struct
import random

def compile_simulated_model(filename="model.bin"):
    """
    Compiles a simulated deep neural network (e.g., embedding/classification layers)
    into a custom binary format with INT8 quantization.
    
    Binary Layout:
    - [Header]
      - Magic bytes: 'MMAP' (4 bytes)
      - Number of layers: Int (4 bytes)
      - Input dimensions: Int (4 bytes)
    - [For each layer]
      - Layer ID: Int (4 bytes)
      - Input Dimension (InDim): Int (4 bytes)
      - Output Dimension (OutDim): Int (4 bytes)
      - Quantization Scale (float): Float (4 bytes)
      - Weights payload: InDim * OutDim bytes (INT8: range -128 to 127)
      - Biases payload: OutDim floats (Float32: OutDim * 4 bytes)
    """
    # 4 Layers with typical AI model dimensions
    layer_configs = [
        {"id": 1, "in_dim": 256, "out_dim": 512},
        {"id": 2, "in_dim": 512, "out_dim": 512},
        {"id": 3, "in_dim": 512, "out_dim": 256},
        {"id": 4, "in_dim": 256, "out_dim": 64}
    ]
    
    print(f"Compiling simulated AI model into '{filename}'...")
    
    with open(filename, "wb") as f:
        # Write magic bytes and header info
        f.write(b"MMAP")
        f.write(struct.pack("<II", len(layer_configs), 256)) # 4 layers, input size 256
        
        for cfg in layer_configs:
            l_id = cfg["id"]
            in_dim = cfg["in_dim"]
            out_dim = cfg["out_dim"]
            
            # Generate simulated weights (-1.0 to 1.0)
            weights = [random.uniform(-1.0, 1.0) for _ in range(in_dim * out_dim)]
            biases = [random.uniform(-0.1, 0.1) for _ in range(out_dim)]
            
            # INT8 Quantization: Find maximum absolute value for scaling
            max_val = max(abs(w) for w in weights) if weights else 1.0
            scale = max_val / 127.0
            
            # Quantize weights: float -> int8
            quantized_weights = []
            for w in weights:
                q_w = int(round(w / scale))
                # Clip just in case
                q_w = max(-128, min(127, q_w))
                quantized_weights.append(q_w)
                
            # Pack layer configuration metadata
            f.write(struct.pack("<IIIf", l_id, in_dim, out_dim, scale))
            
            # Write quantized weights as signed 8-bit bytes
            f.write(struct.pack(f"<{in_dim * out_dim}b", *quantized_weights))
            
            # Write float32 biases
            f.write(struct.pack(f"<{out_dim}f", *biases))
            
    print(f"Compilation completed successfully. Generated binary file: {filename}")

if __name__ == "__main__":
    compile_simulated_model()
