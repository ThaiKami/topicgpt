from openai import OpenAI
import json
import os
import re
from tqdm import tqdm

pattern = re.compile(r"<explaination>:\s*(.*?)\s*<answer>\s*([^\s]+)", re.DOTALL)
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

with open("label_data/pairwise_prompt.txt") as f:
    prompt_template = f.read()
with open("data/input/report-voice-20250101.json") as f:
    texts = [item["text"] for item in json.load(f)[100:6000]]


def llm_pair_label(text1: str, text2: str) -> int:
    prompt = prompt_template.format(
        text1=text1,
        text2=text2,
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    ans = resp.choices[0].message.content
    match = pattern.search(ans)
    if match:
        explanation = match.group(1).strip()
        answer = match.group(2).strip()
    else:
        print("No match found")
    return (explanation, 1 if answer == "yes" else 0)


for index in tqdm(range(0, len(texts), 2)):
    text1 = texts[index]
    text2 = texts[index + 1] if index + 1 < len(texts) else None
    if text2:
        temp = {}
        explanation, answer = llm_pair_label(text1, text2)
        temp["text1"] = text1
        temp["text2"] = text2
        temp["explanation"] = explanation
        temp["label"] = answer
        with open("data/label/llm_pair_label.jsonl", "a") as f:
            f.write(json.dumps(temp, ensure_ascii=False) + "\n")
    else:
        print("No pair found for the last text.")
        continue
