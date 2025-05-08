from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import random


model_name = "Qwen/Qwen3-0.6B"
# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="cuda"
)
with open("data/clusters_top10_sorted.json") as f:
    clusters_data = json.load(f)

with open("data/input/news-100.json") as f:
    texts_data = json.load(f)

with open("prompt/base_prompt.txt", encoding="utf-8") as f:
    base_prompt = f.read()

random.seed(42)
# prepare the model input
texts = []
for cluster_id, ids in clusters_data.items():
    random.shuffle(ids)
    ids = ids[:5]
    input_texts = [text["text"][:1000] for text in texts_data if text["docId"] in ids]
    prompt = base_prompt.format(Document="\n\n".join(input_texts))
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Switches between thinking and non-thinking modes. Default is True.
    )
    texts.append(text)

model_inputs = tokenizer(
    texts, return_tensors="pt", truncation=True, padding=True, max_length=16_000
).to(model.device)

# conduct text completion
generated_ids = model.generate(**model_inputs, max_new_tokens=100)

for idx, gen_id in enumerate(generated_ids):
    output_ids = generated_ids[idx][len(model_inputs.input_ids[idx]) :].tolist()

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(
        output_ids[:index], skip_special_tokens=True
    ).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    # print("thinking content:", thinking_content)
    print("content:", content)
