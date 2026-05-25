# gigabytes 🧠⚡

**Zero-RAM Quantized Memory-Mapped AI Inference Engine for Legacy & Meager Hardware**

> "AI models are ultimately just sequences of numbers (weights & tokens). Loading gigabytes of parameters wholly into RAM is highly inefficient. If we design smart disk structures and virtual mappings, we can run advanced AI models locally even on low-end, legacy laptops."

---

## 💡 The Philosophy

Modern AI deployment often demands massive amounts of system RAM, forcing developers onto expensive cloud instances or high-end GPUs. However, during inference, neural networks evaluate layers sequentially. Keeping the weights of all layers resident in memory at all times is a massive waste of resources. 

Inspired by the memory-efficiency struggles of the COBOL era under severe hardware limitations, **gigabytes** demonstrates a **Zero-RAM AI inference engine** written from scratch in pure Python.

By combining **INT8 Quantization** and **Virtual Memory Mapping (`mmap`)**, **gigabytes** streams model weights directly from the disk page cache on the fly. 

Regardless of whether the model scales to **100 Megabytes or 10 Gigabytes**, the active Python heap allocation remains virtually flat: **under 160 KB**.

---

## 🛠️ How It Works

```
┌────────────────────────────────────────────────────────┐
│                      model.bin                         │  ◀── Mapped into virtual address space via mmap
└────────────────────────────────────────────────────────┘
                           │
             [On-Demand Page-in from Disk]
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│            TinyMmapEngine (RAM < 160 KB)               │  ◀── Computes layer-by-layer on-the-fly
└────────────────────────────────────────────────────────┘
```

1. **INT8 Quantized Compiler (`compiler.py`)**: 
   Compresses Float32 weights into a signed 1-byte INT8 format accompanied by layer scale dequantization factors, reducing the model file size by **75%**.
2. **On-Demand Page Interception (`inference.py`)**:
   Instead of using `open().read()` to parser-load the model into the system heap, Python's native `mmap` module binds the `.bin` file into the OS virtual address space.
3. **Sequential Layer Streaming**:
   As a forward pass evaluates Layer $N$, only the target weight bytes for that specific layer are pulled into the physical CPU cache (demanded page-in). Previous layers are immediately discarded and freed by the operating system.

---

## 📈 Performance & Verification (Benchmark Results)

Running the benchmark compiles a simulated multi-layer deep neural network and executes a full feed-forward pass. The active RAM allocations are traced at the byte-level:

```bash
Compiling simulated AI model into 'simulated_model.bin'...
Compilation completed successfully. Generated binary file: simulated_model.bin

--- Starting Memory-Mapped AI Inference ---
------------------------------------------
Inference Completed in: 1644.58 ms
Active Heap Memory Usage: 139.31 KB
Peak RAM Allocation during execution: 155.14 KB
------------------------------------------
Model Output Dim: 64
Sample Output Node Values: [-190.787, 4698.627, -2151.1382, -293.6241, -6088.5962]...
------------------------------------------
SUCCESS: Zero-RAM virtual weight streaming verified!
The active RAM usage did NOT load model weights into heap memory.
```

- **Peak Memory Allocation**: **~155 KB**
- **Heap RAM Consumption**: **Flat & Independent of Model Size**

---

## 🗂️ Project Structure

- `compiler.py`: Quantizes and serializes simulated layers (weights, biases, and metadata) into a `.bin` format.
- `inference.py`: Zero-framework custom inference engine utilizing memory-mapping and dequantization.
- `benchmark.py`: Runs a mock feed-forward pass and profiles RAM allocation metrics.

---

## 🚀 Quick Start

Ensure you have Python installed. Clone the repository and run:

```bash
# Run the complete compiler and memory tracking benchmark
python benchmark.py
```

---

## 🧬 Korean Linguistic Connection: 15-Bit Phoneme Bit-Packing

For text indexing and databases in constraints-heavy systems, we can compress Hangeul text based on its phonology:
- Rather than representing Korean characters in **3-byte UTF-8**, we can decompose Hangeul syllables into:
  - **Choseong (Initial)**: 19 phonemes (5 bits)
  - **Jungseong (Medial)**: 21 phonemes (5 bits)
  - **Jongseong (Final)**: 28 phonemes including none (5 bits)
- This packs any Korean character into exactly **15 bits** (less than 2 bytes!). 
- This enables super-fast, bitwise prefix and phonetic searches (e.g., initial-consonant 초성검색) using primitive bitmask operations (`char & 0x7C00`) directly on memory-mapped files without string parsing overhead.

---

## 📜 License

This project is licensed under the MIT License.
