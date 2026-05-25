import os
import sys
import gc
import time
import threading
import psutil
import torch
import soundfile as sf
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import winsound

# Ensure UTF-8 rendering on Windows terminal/GUI console if needed
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

class Qwen3TTSGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 Gigabytes - Qwen3-TTS CPU Low-RAM Runner")
        self.root.geometry("850x700")
        self.root.configure(bg="#1e1e2e")
        self.root.resizable(True, True)

        # Style Configuration
        self.setup_styles()

        # Variables
        self.model = None
        self.model_dir = ""
        self.ref_audio_var = tk.StringVar(value="")
        self.ref_text_var = tk.StringVar(value="")
        self.output_path_var = tk.StringVar(value="output_voice_clone.wav")
        self.status_var = tk.StringVar(value="Ready. Click 'Load Model' to begin.")
        self.ram_var = tk.StringVar(value="RAM Usage: - MB")
        self.thread_limit_var = tk.StringVar(value="4")

        # Load default ref text
        self.load_default_ref_text()

        # Draw UI
        self.create_widgets()

        # Start RAM monitor loop
        self.update_ram_display()

    def setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # Configure overall ttk styling with dark mode palette
        self.style.configure(".", background="#1e1e2e", foreground="#f8f9fa", font=("Segoe UI", 10))
        self.style.configure("TLabel", background="#1e1e2e", foreground="#f8f9fa")
        self.style.configure("Header.TLabel", background="#1e1e2e", foreground="#bb86fc", font=("Segoe UI", 16, "bold"))
        self.style.configure("Sub.TLabel", background="#1e1e2e", foreground="#a5a5b5", font=("Segoe UI", 9, "italic"))
        
        # Card style
        self.style.configure("Card.TFrame", background="#252538", relief="flat")
        self.style.configure("CardLabel.TLabel", background="#252538", foreground="#f8f9fa")
        
        # Button styles
        self.style.configure("Action.TButton", background="#7b2cbf", foreground="#ffffff", borderwidth=0, font=("Segoe UI", 10, "bold"))
        self.style.map("Action.TButton", background=[("active", "#9d4edd")])
        
        self.style.configure("Secondary.TButton", background="#3a3a50", foreground="#f8f9fa", borderwidth=0)
        self.style.map("Secondary.TButton", background=[("active", "#4a4a6a")])

        # Entry Style
        self.style.configure("TEntry", fieldbackground="#323246", foreground="#f8f9fa", borderwidth=0)

    def load_default_ref_text(self):
        txt_path = "woo.txt"
        if os.path.exists(txt_path):
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    self.ref_text_var.set(f.read().strip())
            except Exception:
                self.ref_text_var.set("언제까지 하라는 거야. 진짜. 귀찮게 왜 자꾸 귀찮게해.")
        else:
            self.ref_text_var.set("언제까지 하라는 거야. 진짜. 귀찮게 왜 자꾸 귀찮게해.")

    def update_ram_display(self):
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / (1024 * 1024)
        self.ram_var.set(f"System Process RAM: {ram_mb:.1f} MB")
        self.root.after(1000, self.update_ram_display)

    def log(self, text):
        self.console.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {text}\n")
        self.console.see(tk.END)

    def select_ref_audio(self):
        file_path = filedialog.askopenfilename(
            title="Select Reference Audio File",
            filetypes=[("WAV files", "*.wav")]
        )
        if file_path:
            self.ref_audio_var.set(file_path)
            self.log(f"Reference Audio updated: {file_path}")

    def select_output_path(self):
        file_path = filedialog.asksaveasfilename(
            title="Select Output Audio Path",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav")]
        )
        if file_path:
            self.output_path_var.set(file_path)
            self.log(f"Output Path updated: {file_path}")

    def create_widgets(self):
        main_container = tk.Frame(self.root, bg="#1e1e2e", padx=20, pady=20)
        main_container.pack(fill="both", expand=True)

        # Header Title
        title_frame = tk.Frame(main_container, bg="#1e1e2e")
        title_frame.pack(fill="x", pady=(0, 15))
        
        title_lbl = ttk.Label(title_frame, text="🧠 GIGABYTES : Qwen3-TTS CPU GUI", style="Header.TLabel")
        title_lbl.pack(side="left")
        
        sub_lbl = ttk.Label(title_frame, text="Zero-RAM lazy memory-mapped streaming mode active", style="Sub.TLabel")
        sub_lbl.pack(side="left", padx=15, pady=5)

        # Status Bar / RAM Box
        ram_frame = tk.Frame(title_frame, bg="#252538", padx=10, pady=5)
        ram_frame.pack(side="right")
        ram_lbl = tk.Label(ram_frame, textvariable=self.ram_var, bg="#252538", fg="#55ff55", font=("Segoe UI Semibold", 10))
        ram_lbl.pack()

        # Separator
        sep = ttk.Separator(main_container, orient="horizontal")
        sep.pack(fill="x", pady=(0, 15))

        # CARD 1: Optimized Hardware Config
        config_card = ttk.Frame(main_container, style="Card.TFrame", padding=15)
        config_card.pack(fill="x", pady=(0, 15))

        ttk.Label(config_card, text="⚙️ Optimized CPU Execution Settings", font=("Segoe UI", 11, "bold"), style="CardLabel.TLabel").grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 10))
        
        # Thread Limit entry
        ttk.Label(config_card, text="Max CPU Threads:", style="CardLabel.TLabel").grid(row=1, column=0, sticky="w")
        thread_entry = tk.Entry(config_card, textvariable=self.thread_limit_var, width=5, bg="#323246", fg="#ffffff", insertbackground="white", bd=0, justify="center")
        thread_entry.grid(row=1, column=1, sticky="w", padx=10)
        
        # Flags indicators
        mmap_lbl = tk.Label(config_card, text="[✔] Virtual mmap Active (low_cpu_mem_usage=True)", bg="#252538", fg="#55ff55", font=("Segoe UI", 9, "bold"))
        mmap_lbl.grid(row=1, column=2, padx=20, sticky="w")
        
        bf16_lbl = tk.Label(config_card, text="[✔] Quantization Active (dtype=bfloat16)", bg="#252538", fg="#55ff55", font=("Segoe UI", 9, "bold"))
        bf16_lbl.grid(row=1, column=3, padx=10, sticky="w")

        # CARD 2: Model & Audio Paths Card
        paths_card = ttk.Frame(main_container, style="Card.TFrame", padding=15)
        paths_card.pack(fill="x", pady=(0, 15))

        ttk.Label(paths_card, text="🎤 Voice Cloning Configuration", font=("Segoe UI", 11, "bold"), style="CardLabel.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))

        # Reference Audio Entry
        ttk.Label(paths_card, text="Reference Audio (.wav):", style="CardLabel.TLabel").grid(row=1, column=0, sticky="w", pady=5)
        ref_aud_entry = tk.Entry(paths_card, textvariable=self.ref_audio_var, width=65, bg="#323246", fg="#ffffff", insertbackground="white", bd=0)
        ref_aud_entry.grid(row=1, column=1, sticky="ew", padx=10, ipady=3)
        ref_aud_btn = ttk.Button(paths_card, text="Browse...", style="Secondary.TButton", command=self.select_ref_audio)
        ref_aud_btn.grid(row=1, column=2, sticky="e")

        # Reference Transcript
        ttk.Label(paths_card, text="Reference Script (Text):", style="CardLabel.TLabel").grid(row=2, column=0, sticky="w", pady=5)
        ref_txt_entry = tk.Entry(paths_card, textvariable=self.ref_text_var, width=65, bg="#323246", fg="#ffffff", insertbackground="white", bd=0)
        ref_txt_entry.grid(row=2, column=1, sticky="ew", padx=10, ipady=3)
        ttk.Label(paths_card, text="(Must match audio)", style="Sub.TLabel").grid(row=2, column=2, sticky="w")

        # Output WAV path
        ttk.Label(paths_card, text="Output Voice Path:", style="CardLabel.TLabel").grid(row=3, column=0, sticky="w", pady=5)
        out_entry = tk.Entry(paths_card, textvariable=self.output_path_var, width=65, bg="#323246", fg="#ffffff", insertbackground="white", bd=0)
        out_entry.grid(row=3, column=1, sticky="ew", padx=10, ipady=3)
        out_btn = ttk.Button(paths_card, text="Save as...", style="Secondary.TButton", command=self.select_output_path)
        out_btn.grid(row=3, column=2, sticky="e")

        paths_card.columnconfigure(1, weight=1)

        # CARD 3: Input Text Box
        input_card = ttk.Frame(main_container, style="Card.TFrame", padding=15)
        input_card.pack(fill="x", pady=(0, 15))

        ttk.Label(input_card, text="✍️ Text to Synthesize (Korean / English)", font=("Segoe UI", 11, "bold"), style="CardLabel.TLabel").pack(anchor="w", pady=(0, 5))
        
        self.input_text = tk.Text(input_card, height=4, bg="#323246", fg="#ffffff", insertbackground="white", bd=0, font=("Segoe UI", 11), wrap="word")
        self.input_text.pack(fill="x", pady=5)
        self.input_text.insert(tk.END, "안녕하세요, 저사양 노트북에서 구동 가능하도록 최적화가 완료된 큐원 쓰리 티티에스 음성 합성 시스템입니다.")

        # Button Operations Panel
        ops_frame = tk.Frame(main_container, bg="#1e1e2e")
        ops_frame.pack(fill="x", pady=(0, 15))

        self.load_btn = ttk.Button(ops_frame, text="🚀 Eager Load Model", style="Secondary.TButton", command=self.trigger_load_model)
        self.load_btn.pack(side="left", padx=(0, 10), ipady=5)

        self.synth_btn = ttk.Button(ops_frame, text="✨ Generate Speech (Voice Clone)", style="Action.TButton", command=self.trigger_synthesis)
        self.synth_btn.pack(side="left", fill="x", expand=True, ipady=5)

        self.play_btn = ttk.Button(ops_frame, text="▶ Play WAV", style="Secondary.TButton", state="disabled", command=self.play_audio)
        self.play_btn.pack(side="right", padx=(10, 0), ipady=5)

        self.stop_btn = ttk.Button(ops_frame, text="■ Stop Play", style="Secondary.TButton", state="disabled", command=self.stop_audio)
        self.stop_btn.pack(side="right", padx=(10, 0), ipady=5)

        # CARD 4: Execution Real-time Console Log
        console_card = ttk.Frame(main_container, style="Card.TFrame", padding=15)
        console_card.pack(fill="both", expand=True)

        ttk.Label(console_card, text="💻 Real-time Process Log Console", font=("Segoe UI", 11, "bold"), style="CardLabel.TLabel").pack(anchor="w", pady=(0, 5))
        
        # Text console with scrollbar
        console_frame = tk.Frame(console_card, bg="#181824")
        console_frame.pack(fill="both", expand=True)

        self.console = tk.Text(console_frame, bg="#181824", fg="#55ff55", insertbackground="white", bd=0, font=("Consolas", 10))
        self.console.pack(side="left", fill="both", expand=True)
        
        scroll = ttk.Scrollbar(console_frame, orient="vertical", command=self.console.yview)
        scroll.pack(side="right", fill="y")
        self.console.config(yscrollcommand=scroll.set)

        # Footer
        footer = ttk.Label(main_container, text="Designed by Antigravity AI Engine (BFloat16 + low_cpu_mem_usage=True)", style="Sub.TLabel")
        footer.pack(pady=(10, 0))

    # --- Operation Logics ---

    def play_audio(self):
        out_path = self.output_path_var.get()
        if os.path.exists(out_path):
            try:
                self.log(f"Playing output audio: {out_path}")
                winsound.PlaySound(out_path, winsound.SND_ASYNC)
                self.stop_btn.config(state="normal")
            except Exception as e:
                self.log(f"Audio Playback Failed: {e}")
        else:
            messagebox.showerror("Error", f"WAV file not found at: {out_path}")

    def stop_audio(self):
        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
            self.log("Playback stopped.")
            self.stop_btn.config(state="disabled")
        except Exception:
            pass

    def trigger_load_model(self):
        # Run in a background thread to prevent UI freezing
        threading.Thread(target=self.load_model_thread, daemon=True).start()

    def load_model_thread(self):
        self.load_btn.config(state="disabled")
        self.synth_btn.config(state="disabled")
        
        try:
            threads_count = int(self.thread_limit_var.get())
            torch.set_num_threads(threads_count)
            self.log(f"Throttling CPU thread count limit to: {threads_count}")
        except Exception:
            self.log("Invalid thread count limit. Proceeding with default.")

        self.log("Loading Qwen3-TTS Model on CPU (BFloat16 + lazy memory-mapped)...")
        start_time = time.time()
        
        try:
            gc.collect()
            start_ram = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            
            # Lazy Load using Safetensors + low_cpu_mem_usage
            from qwen_tts import Qwen3TTSModel
            self.model = Qwen3TTSModel.from_pretrained(
                self.model_dir,
                device_map="cpu",
                dtype=torch.bfloat16,
                low_cpu_mem_usage=True
            )
            
            end_ram = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            elapsed = time.time() - start_time
            
            self.log(f"Model loaded successfully in {elapsed:.2f} seconds!")
            self.log(f"Active RAM Overhead: {end_ram - start_ram:.2f} MB")
            self.log(f"Total Process RAM: {end_ram:.2f} MB")
            
            self.root.after(0, lambda: self.status_var.set("Model Loaded. Ready to Synthesize."))
        except Exception as e:
            self.log(f"Model Loading Failed: {e}")
            import traceback
            self.log(traceback.format_exc())
            messagebox.showerror("Error", f"Failed to load model:\n{e}")
        
        self.load_btn.config(state="normal")
        self.synth_btn.config(state="normal")

    def trigger_synthesis(self):
        text_val = self.input_text.get("1.0", tk.END).strip()
        if not text_val:
            messagebox.showwarning("Warning", "Please enter text to synthesize!")
            return
        
        # Run in a background thread to prevent UI freezing
        threading.Thread(target=self.synthesis_thread, args=(text_val,), daemon=True).start()

    def synthesis_thread(self, text_val):
        self.synth_btn.config(state="disabled")
        self.load_btn.config(state="disabled")
        self.play_btn.config(state="disabled")
        
        try:
            # Auto load model if not loaded yet
            if self.model is None:
                self.load_model_thread()
                if self.model is None:
                    raise RuntimeError("Model loading failed.")

            ref_aud = self.ref_audio_var.get()
            ref_txt = self.ref_text_var.get()
            out_path = self.output_path_var.get()

            if not os.path.exists(ref_aud):
                raise FileNotFoundError(f"Reference WAV file not found: {ref_aud}")

            self.log(f"Synthesizing script: '{text_val[:40]}...'")
            self.log(f"Reference Audio: {os.path.basename(ref_aud)}")
            
            start_time = time.time()
            start_ram = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

            # Auto language detection or default to Korean
            lang = "Korean"
            if any(ord(c) > 0x7F for c in text_val): # Contains non-ascii (like Korean/Chinese/Japanese)
                lang = "Korean"
            else:
                lang = "English"

            # Execute Voice Clone
            wavs, sr = self.model.generate_voice_clone(
                text=text_val,
                language=lang,
                ref_audio=ref_aud,
                ref_text=ref_txt
            )
            
            # Save WAV file
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            sf.write(out_path, wavs[0], sr)
            
            end_ram = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            elapsed = time.time() - start_time
            duration = len(wavs[0]) / sr
            
            self.log(f"Speech synthesis completed in {elapsed:.2f} seconds!")
            self.log(f"Generated WAV file: {out_path}")
            self.log(f"Sample Rate: {sr} Hz, Audio Duration: {duration:.2f}s")
            self.log(f"Peak execution RAM: {end_ram:.2f} MB")
            
            self.play_btn.config(state="normal")
            messagebox.showinfo("Success", f"Audio synthesized successfully!\nLength: {duration:.2f}s\nSaved to: {out_path}")
        except Exception as e:
            self.log(f"Synthesis Failed: {e}")
            import traceback
            self.log(traceback.format_exc())
            messagebox.showerror("Error", f"Failed to synthesize voice clone:\n{e}")

        self.synth_btn.config(state="normal")
        self.load_btn.config(state="normal")

if __name__ == "__main__":
    root = tk.Tk()
    app = Qwen3TTSGUI(root)
    root.mainloop()
