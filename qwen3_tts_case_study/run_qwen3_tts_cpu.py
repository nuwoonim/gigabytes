import json
import os
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model_dir = "C:\\Users\\gytw2\\Desktop\\ttsk\\Qwen3-TTS-12Hz-1.7B-Base-BNB-4bit"

# Limit CPU thread count to prevent context-switching thread thrashing
torch.set_num_threads(4)

print("Loading model on CPU in BFloat16 with lazy memory-mapping (low RAM mode)...")
model = Qwen3TTSModel.from_pretrained(
    model_dir,
    device_map="cpu",
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True
)

print(f"Model loaded. Device: {model.device}")
print(f"Model type: {model.model.tts_model_type}")

ref_audio = os.path.join(os.path.dirname(__file__), "..", "..", "ttsk", "woo1.wav")
if not os.path.exists(ref_audio):
    ref_audio = "C:\\Users\\gytw2\\Desktop\\ttsk\\woo1.wav"

ref_text = "언제까지 하라는 거야. 진짜. 귀찮게 왜 자꾸 귀찮게해.내가 지금 아까 얘기하고 있는데도 내말을 끊어버리고, 그리고 아빠할 말 할려고 아주 눈에 불을 끼고 기다리는 모습이 아주 한데 때려 주고 싶었어. 아바 꿀밤을 맞을거야."

output_path = os.path.join(os.path.dirname(__file__), "output_voice_clone.wav")
print(f"Synthesizing speech using local voice clone...")
wavs, sr = model.generate_voice_clone(
    text="안녕하세요, 저사양 노트북에서 구동 가능하도록 최적화가 완료된 큐원 쓰리 티티에스 음성 합성 시스템입니다.",
    language="Korean",
    ref_audio=ref_audio,
    ref_text=ref_text,
)
sf.write(output_path, wavs[0], sr)
print(f"Output saved to {output_path}")
print(f"Sample rate: {sr}, Audio length: {len(wavs[0]) / sr:.2f}s")
print("SUCCESS: Model runs correctly on CPU!")
