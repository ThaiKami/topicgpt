from openai import OpenAI
import numpy as np
from tqdm import tqdm
import json
from collections import defaultdict
import hdbscan
import umap
import random
import os
from sklearn.cluster import DBSCAN


# --------------------Create Embeddings--------------------
def get_embeddings(
    data: list[dict],
    openai_api_key: str = "VnSocial",
    openai_api_base: str = "http://10.159.19.101:30043/v1",
):
    client_embedding = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    embeddings = []
    for doc in tqdm(data, desc="Getting embeddings"):
        response = client_embedding.embeddings.create(
            input=doc["text"],
            model="vnsocial_embedding",  # OpenAI recommended
        )
        embeddings_float16 = np.array(response.data[0].embedding, dtype=np.float32)
        embeddings.append(embeddings_float16)
    embeddings = np.stack(embeddings, axis=0)
    return embeddings


def get_clusters(documents, embeddings):
    print("Clustering...")
    # UMAP dimensionality reduction
    # reducer = umap.UMAP(
    #     n_components=10,
    #     n_neighbors=5,  # Adjust: small for local, larger for global
    #     min_dist=0.0,
    #     metric="cosine",
    #     random_state=42,
    # )
    # print("Reducing dimensions...")
    # embeddings = reducer.fit_transform(embeddings)
    # HDBSCAN clustering
    # clusterer = hdbscan.HDBSCAN(
    #     min_cluster_size=10,  # only clusters ≥10 points
    #     min_samples=10,  # require 10 neighbors for a core point
    #     metric="euclidean",
    #     cluster_selection_method="eom",
    #     cluster_selection_epsilon=0.01,  # split clusters if internal distances >0.1
    #     prediction_data=True,
    # )
    # labels = clusterer.fit_predict(embeddings)

    clusterer = DBSCAN(
        eps=0.02,  # cosine‐distance threshold: only ≥0.98 sim will link
        min_samples=5,  # allow even singleton clusters if no neighbor within eps
        metric="cosine",
    )
    labels = clusterer.fit_predict(embeddings)
    # Group texts by cluster
    text_cluster_pairs = list(zip(documents, labels))
    clusters = defaultdict(list)
    for text, cluster_id in text_cluster_pairs:
        if cluster_id != -1:  # Exclude noise points
            clusters[int(cluster_id)].append(text)

    # Sort clusters by size
    clusters = list(
        sorted(clusters.items(), key=lambda item: len(item[1]), reverse=True)
    )
    clusters_dicts = [
        {"id": cluster_id, "texts": texts} for cluster_id, texts in clusters
    ]

    return clusters_dicts


# Load OpenAI API key and paths
def create_label(api_key: str, prompt_path: str, clusters_dicts: list[dict]):
    api_key = os.getenv("OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)
    with open(prompt_path, encoding="utf-8") as f:
        base_prompt = f.read()

    random.seed(42)

    texts = []
    for cluster in clusters_dicts:
        cluster_texts = cluster["texts"]
        cluster_texts = random.sample(cluster_texts, 5)
        input_texts = [text[:512] for text in cluster_texts]
        prompt = base_prompt.format(Document="\n\n".join(input_texts))
        texts.append(prompt)
    topics = []
    for i in tqdm(texts):
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": i}],
            max_tokens=500,
            temperature=1.0,
            top_p=1.0,
        )
        topics.append(completion.choices[0].message.content)
    final_dict = []
    for topic, cluster in zip(topics, clusters_dicts):
        temp = {}
        temp["topic"] = topic
        temp["id"] = cluster["id"]
        temp["numTexts"] = len(cluster["texts"])
        temp["texts"] = cluster["texts"]
        final_dict.append(temp)
    return final_dict


if __name__ == "__main__":
    with open("data/input/news-10k-shuffled.json") as f:
        data = json.load(f)[:]
    documents = [i["text"] for i in data]
    embeddings = get_embeddings(
        data=data,
        openai_api_key="VnSocial",
        openai_api_base="http://10.159.19.101:30043/v1",
    )
    np.save("data/embeddings_news.npy", embeddings)
    cluster_dicts = get_clusters(documents=documents, embeddings=embeddings)
    cluster_dicts = cluster_dicts[:10]
    with open("data/cluster_dicts_news.json", "w", encoding="utf-8") as f:
        json.dump(cluster_dicts, f, ensure_ascii=False, indent=4)
    api_key = os.getenv("OPENAI_API_KEY")
    final_dict = create_label(
        api_key=api_key,
        prompt_path="prompt/merge_prompt.txt",
        clusters_dicts=cluster_dicts,
    )
    with open("data/final_output_news.json", "w", encoding="utf-8") as f:
        json.dump(final_dict, f, ensure_ascii=False, indent=4)
    total_final_texts = sum([len(i["texts"]) for i in final_dict])
    print(total_final_texts)
