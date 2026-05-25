import os
import sys
import gc
import time
import psutil
import torch

def get_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024) # MB

# Load qwen_tts
from qwen_tts import Qwen3TTSModel

model_dir = ""  # Set this to your local model path

def benchmark_baseline():
    print("\n--- 1. Baseline Model Loading (Float32) ---")
    sys.stdout.flush()
    gc.collect()
    
    start_ram = get_ram_usage()
    print(f"Start RAM: {start_ram:.2f} MB")
    sys.stdout.flush()
    
    start_time = time.time()
    try:
        model = Qwen3TTSModel.from_pretrained(
            model_dir,
            device_map="cpu",
            dtype=torch.float32,
            low_cpu_mem_usage=False
        )
        end_time = time.time()
        end_ram = get_ram_usage()
        
        print(f"Loading Time: {end_time - start_time:.2f} seconds")
        print(f"End RAM: {end_ram:.2f} MB")
        ram_overhead = end_ram - start_ram
        print(f"RAM Overhead: {ram_overhead:.2f} MB")
        sys.stdout.flush()
        
        del model
        gc.collect()
        time.sleep(2)
        print(f"RAM after cleanup: {get_ram_usage():.2f} MB")
        sys.stdout.flush()
        return ram_overhead, end_time - start_time
    except Exception as e:
        print(f"Baseline Load Failed: {e}")
        sys.stdout.flush()
        return None

def benchmark_optimized():
    print("\n--- 2. Optimized Model Loading (BFloat16 + low_cpu_mem_usage) ---")
    sys.stdout.flush()
    gc.collect()
    
    # 1. Limit CPU thread count to prevent thrashing
    torch.set_num_threads(4)
    print("CPU threads limited to 4")
    sys.stdout.flush()
    
    start_ram = get_ram_usage()
    print(f"Start RAM: {start_ram:.2f} MB")
    sys.stdout.flush()
    
    start_time = time.time()
    try:
        # Load in BFloat16 with memory-mapped lazy loading
        model = Qwen3TTSModel.from_pretrained(
            model_dir,
            device_map="cpu",
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        end_time = time.time()
        end_ram = get_ram_usage()
        
        print(f"Loading Time: {end_time - start_time:.2f} seconds")
        print(f"End RAM: {end_ram:.2f} MB")
        ram_overhead = end_ram - start_ram
        print(f"RAM Overhead: {ram_overhead:.2f} MB")
        sys.stdout.flush()
        
        # Test a brief synthesis to ensure BFloat16 CPU execution is mathematically fully functional
        print("Testing voice synthesis in BFloat16 CPU mode...")
        sys.stdout.flush()
        ref_audio = "woo1.wav"
        ref_text = "언제까지 하라는 거야. 진짜. 귀찮게 왜 자꾸 귀찮게해.내가 지금 아까 얘기하고 있는데도 내말을 끊어버리고, 그리고 아빠할 말 할려고 아주 눈에 불을 끼고 기다리는 모습이 아주 한데 때려 주고 싶었어. 아바 꿀밤을 맞을거야."
        wavs, sr = model.generate_voice_clone(
            text="안녕하세요.",
            language="Korean",
            ref_audio=ref_audio,
            ref_text=ref_text
        )
        print(f"Voice synthesized successfully in BFloat16 CPU mode! Waveform samples: {len(wavs[0])}")
        sys.stdout.flush()
        
        del model
        gc.collect()
        time.sleep(2)
        print(f"RAM after cleanup: {get_ram_usage():.2f} MB")
        sys.stdout.flush()
        return ram_overhead, end_time - start_time
    except Exception as e:
        print(f"Optimized Load Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return None

if __name__ == "__main__":
    baseline_metrics = benchmark_baseline()
    optimized_metrics = benchmark_optimized()
    
    if baseline_metrics and optimized_metrics:
        ram_diff = baseline_metrics[0] - optimized_metrics[0]
        percent_reduction = (ram_diff / baseline_metrics[0]) * 100
        print("\n==========================================")
        print("  Qwen3-TTS CPU Performance Optimization  ")
        print("==========================================")
        print(f"Baseline RAM Overhead: {baseline_metrics[0]:.2f} MB")
        print(f"Optimized RAM Overhead: {optimized_metrics[0]:.2f} MB")
        print(f"RAM Footprint Reduced by: {ram_diff:.2f} MB ({percent_reduction:.1f}% reduction!)")
        print(f"Load Time (Baseline): {baseline_metrics[1]:.2f} s")
        print(f"Load Time (Optimized): {optimized_metrics[1]:.2f} s")
        print("==========================================")
        sys.stdout.flush()
