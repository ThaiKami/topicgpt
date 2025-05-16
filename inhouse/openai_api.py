import os
from openai import OpenAI
import json
import random
from tqdm import tqdm

# Load OpenAI API key and paths
api_key = os.getenv("OPENAI_API_KEY")
# Initialize OpenAI client
client = OpenAI(api_key=api_key)
with open("data/clusters_top10_sorted.json") as f:
    clusters_data = json.load(f)

with open("prompt/merge_prompt.txt", encoding="utf-8") as f:
    base_prompt = f.read()

random.seed(42)
# prepare the model input
texts = []
for cluster_id in clusters_data:
    cluster_texts = clusters_data[cluster_id]["texts"]
    # numTexts = clusters_data["cluster_id"]["numTexts"]
    random.shuffle(cluster_texts)
    cluster_texts = cluster_texts[:5]
    input_texts = [text[:500] for text in cluster_texts]
    prompt = base_prompt.format(Document="\n\n".join(input_texts))
    # messages = [{"role": "user", "content": prompt}]
    texts.append(prompt)
topics = []
for i in tqdm(texts):
    # print(i)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": i}],
        max_tokens=1000,
        temperature=0.0,
        top_p=1.0,
    )
    topics.append(completion.choices[0].message.content)
final_dict = []
for topic, cluster_id in zip(topics, clusters_data):
    temp_dict = {}
    temp_dict["topic"] = topic
    temp_dict["numTexts"] = clusters_data[cluster_id]["numTexts"]
    temp_dict["cluster_id"] = cluster_id
    final_dict.append(temp_dict)
    # clusters_data[cluster_id]["topic"] = topic

with open("data/final_output.json", "w", encoding="utf-8") as f:
    json.dump(final_dict, f, ensure_ascii=False, indent=4)
