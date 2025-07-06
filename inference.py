import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from model import PHQ9WithAttnPool
from safetensors.torch import load_file

ckpts = [
    d for d in os.listdir("results")
    if os.path.isdir(os.path.join("results", d)) and d.startswith("checkpoint-")
]

steps = [int(d.split("-")[1]) for d in ckpts]

best_step = max(steps)
best_ckpt = os.path.join("results", f"checkpoint-{best_step}")
print("로드할 체크포인트:", best_ckpt)

model = PHQ9WithAttnPool(
    backbone="YeRyeongLee/mental-bert-base-uncased-finetuned-0505",
    task="regression",
    hidden_layer_list=[1024,512,256,128,64]
)
tokenizer = AutoTokenizer.from_pretrained(best_ckpt)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
safe_path = os.path.join(best_ckpt, "model.safetensors")
state_dict = load_file(safe_path, device=device)
model.load_state_dict(state_dict)

print(model)

input_str = "Today I woke up feeling completely refreshed and energized, enjoyed a healthy breakfast, took a peaceful morning walk in a bright park, chatted with a dear friend over coffee, returned home to finish my work efficiently, then spent the afternoon reading an interesting novel, prepared a nutritious dinner, practiced some gentle stretching, listened to uplifting music, and now I look forward to a restful night’s rest with a calm mind and a light heart, feeling gratitude and optimism, about tomorrow."
input_tokens = tokenizer(input_str, return_tensors="pt")
input_ids = input_tokens["input_ids"].to(device)
attention_mask = input_tokens["attention_mask"].to(device)

print(model.inference(input_ids, attention_mask))