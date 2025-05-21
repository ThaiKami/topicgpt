import os
from openai import OpenAI
import json
import random
from tqdm import tqdm

# Load OpenAI API key and paths
api_key = os.getenv("OPENAI_API_KEY")
# Initialize OpenAI client
client = OpenAI(api_key=api_key)
with open("data/output/hbdscan_news.json") as f:
    clusters_data = json.load(f)[:10] # take top 10 clusters 

with open("prompt/merge_prompt.txt", encoding="utf-8") as f:
    base_prompt = f.read()

random.seed(42)
# prepare the model input
texts = []
for cluster in clusters_data:
    cluster_texts = cluster["texts"]
    cluster_texts = random.sample(cluster_texts, 5)
    input_texts = [text[:512] for text in cluster_texts]
    prompt = base_prompt.format(Document="\n\n".join(input_texts))
    # messages = [{"role": "user", "content": prompt}]
    texts.append(prompt)
topics = []
for i in tqdm(texts):
    # print(i)
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": i}],
        max_tokens=500,
        temperature=1.0,
        top_p=1.0,
    )
    topics.append(completion.choices[0].message.content)
final_dict = []
for topic, cluster in zip(topics, clusters_data):
    temp = {}
    temp["topic"] = topic
    temp["id"] = cluster["id"]
    temp["numTexts"] = len(cluster["texts"])
    temp["texts"] = cluster["texts"]
    final_dict.append(temp)

with open("data/final_output_news.json", "w", encoding="utf-8") as f:
    json.dump(final_dict, f, ensure_ascii=False, indent=4)
