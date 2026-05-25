import mmap
import struct
import math

class TinyMmapEngine:
    def __init__(self, model_path):
        self.model_path = model_path
        self.file = open(model_path, "rb")
        # Memory-map the model file directly in virtual memory (Read-Only)
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.parse_header()
        
    def parse_header(self):
        # Read magic bytes and metadata from the mmap buffer
        magic = self.mm[0:4]
        if magic != b"MMAP":
            raise ValueError("Invalid model format! Magic bytes do not match 'MMAP'.")
        
        self.num_layers, self.input_dim = struct.unpack("<II", self.mm[4:12])
        # Layer indices in file are tracked dynamically during execution
        
    def run_inference(self, input_vector):
        """
        Runs neural network inference layer-by-layer.
        It accesses weights directly from the virtual memory-mapped space (mmap)
        without keeping the weights of other layers in RAM.
        """
        if len(input_vector) != self.input_dim:
            raise ValueError(f"Input dimensions mismatch! Expected {self.input_dim}, got {len(input_vector)}.")
            
        current_activation = input_vector
        offset = 12 # Header offset in bytes
        
        for layer_idx in range(self.num_layers):
            # 1. Read layer metadata (16 bytes: ID, InDim, OutDim, Scale)
            l_id, in_dim, out_dim, scale = struct.unpack("<IIIf", self.mm[offset:offset+16])
            offset += 16
            
            # 2. Extract pointers (offsets) to weights and biases inside the memory-mapped file
            weights_start_offset = offset
            weights_size = in_dim * out_dim
            offset += weights_size # Weights are stored as 1-byte INT8
            
            biases_start_offset = offset
            biases_size = out_dim * 4
            offset += biases_size # Biases are stored as 4-byte Float32
            
            # 3. Stream and calculate current layer output
            # We construct outputs mathematically in place, streaming elements from the mapped file on the fly
            next_activation = [0.0] * out_dim
            
            for j in range(out_dim):
                # Calculate dot product: input_vector . layer_weights_for_node_j
                # Directly slice weight byte array from mmap on demand (no massive RAM footprint)
                dot_prod = 0.0
                
                # Fetch biases on demand
                bias_val = struct.unpack("<f", self.mm[biases_start_offset + j*4 : biases_start_offset + (j+1)*4])[0]
                
                for i in range(in_dim):
                    # Fetch single weight byte on-demand (Memory page read)
                    weight_byte = self.mm[weights_start_offset + (i * out_dim + j)]
                    # Convert signed 8-bit byte to signed integer
                    weight_int8 = weight_byte if weight_byte < 128 else weight_byte - 256
                    
                    # Dequantize: weight_float = weight_int8 * scale
                    weight_float = weight_int8 * scale
                    dot_prod += current_activation[i] * weight_float
                    
                # Add bias and apply Activation Function (ReLU for hidden layers, Linear for output layer)
                output_val = dot_prod + bias_val
                if layer_idx < self.num_layers - 1:
                    # ReLU Activation: max(0, x)
                    next_activation[j] = max(0.0, output_val)
                else:
                    # Linear Activation for output layer
                    next_activation[j] = output_val
                    
            # Move activations forward, discarding previous layers
            current_activation = next_activation
            
        return current_activation

    def close(self):
        self.mm.close()
        self.file.close()

if __name__ == "__main__":
    print("TinyMmapEngine defined successfully.")
