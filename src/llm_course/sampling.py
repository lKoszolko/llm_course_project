import csv
import time
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # ten sam model co w lab01a
PROMPT = "Podaj nazwę i jednozdaniowy opis aplikacji pomagającej studentowi planować naukę."

CONFIGS = {
    "A": dict(do_sample=False),                                   # greedy
    "B": dict(do_sample=True, temperature=0.3, top_p=1.0),
    "C": dict(do_sample=True, temperature=0.8, top_p=0.9),
    "D": dict(do_sample=True, temperature=1.2, top_p=0.9),
}
REPEATS = 5

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).eval()

chat = tokenizer.apply_chat_template(
    [{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True
)
inputs = tokenizer(chat, return_tensors="pt")

rows = []
for configuration, params in CONFIGS.items():
    for run in range(1, REPEATS + 1):
        torch.manual_seed(1000 + run)  # inny seed w każdym powtórzeniu
        start = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=80,
                pad_token_id=tokenizer.eos_token_id,
                **params,
            )
        latency_s = round(time.perf_counter() - start, 3)
        new_ids = output[0][inputs["input_ids"].shape[1]:]
        rows.append({
            "configuration": configuration,
            "run": run,
            "text": tokenizer.decode(new_ids, skip_special_tokens=True).strip(),
            "output_tokens": len(new_ids),
            "latency_s": latency_s,
        })
        print(configuration, run, rows[-1]["output_tokens"], latency_s)


def normalized(text: str) -> str:
    return " ".join(text.lower().split())


counts = Counter(normalized(row["text"]) for row in rows)
print("unikalne odpowiedzi (ogółem):", len(counts))

for configuration in CONFIGS:
    group = [r for r in rows if r["configuration"] == configuration]
    unique = len({normalized(r["text"]) for r in group})
    avg_tokens = sum(r["output_tokens"] for r in group) / len(group)
    avg_time = sum(r["latency_s"] for r in group) / len(group)
    print(f"{configuration}: unikalne={unique}/5, śr. tokeny={avg_tokens:.1f}, śr. czas={avg_time:.2f}s")

with open("lab01b_sampling.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)