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
import argparse
import os

random.seed(42)


# --------------------Create Embeddings--------------------
def get_embeddings(
    documents=list[str],
    api_key: str = "VnSocial",
    api_base: str = "http://10.159.19.101:30043/v1",
):
    client_embedding = OpenAI(
        api_key=api_key,
        base_url=api_base,
    )
    embeddings = []
    for doc in tqdm(documents, desc="Getting embeddings"):
        response = client_embedding.embeddings.create(
            input=doc,
            model="vnsocial_embedding",  # OpenAI recommended
        )
        embeddings_float16 = np.array(response.data[0].embedding, dtype=np.float32)
        embeddings.append(embeddings_float16)
    embeddings = np.stack(embeddings, axis=0)
    return embeddings


def get_clusters(documents, embeddings, data_type: str = "article"):
    print("Clustering...")
    if data_type == "article":
        clusterer = DBSCAN(
            eps=0.2,
            min_samples=5,
            metric="cosine",
        )
    elif data_type == "ttht":
        # UMAP dimensionality reduction
        reducer = umap.UMAP(
            n_components=50,
            n_neighbors=10,  # Adjust: small for local, larger for global
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        )
        embeddings = reducer.fit_transform(embeddings)
        # HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=50,
            min_samples=10,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )
    else:
        raise Exception("data_type must be in ['ttht', 'article']")
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
    client = OpenAI(api_key=api_key)
    with open(prompt_path, encoding="utf-8") as f:
        base_prompt = f.read()
    texts = []
    for cluster in clusters_dicts:
        originals = cluster["texts"]
        if not originals:
            continue  # no texts → nothing to label
        count = min(5, len(originals))
        sampled = random.sample(originals, count)
        input_texts = [t[:512] for t in sampled]
        prompt = base_prompt.format(Document="\n\n".join(input_texts))
        texts.append(prompt)

    topics = []
    for i in tqdm(texts, desc="Creating label"):
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
    VNSOCIAL_API_KEY = "VnSocial"  # Won't change
    VNSOCIAL_API_BASE = "http://10.159.19.101:30043/v1"  # Won't change

    parser = argparse.ArgumentParser(
        description="Script that cluster and label input data"
    )
    parser.add_argument(
        "-dt",
        "--data_type",
        help="'ttht' for TTHT and  or 'article' articles",
        default="article",
        type=str,
    )
    parser.add_argument(
        "-ji",
        "--json_input",
        help="Path to a JSON file that is a list of dicts, in each dict there is a 'text' field",
        default="data/input/news-10k-shuffled.json",
        type=str,
    )
    parser.add_argument(
        "-ow",
        "--overwrite",
        help="Overwrite existing embeddings database",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    JSON_INPUT = args.json_input

    if args.data_type == "ttht":
        EMBEDDINGS_PATH = "data/embeddings_TTHT.npy"
        CLUSTER_DICT_PATH = "data/cluster_dicts_TTHT.json"
        LABEL_PROMPT_PATH = "prompt/TTHT_label_prompt.txt"
        FINAL_OUTPUT_PATH = "data/final_output_TTHT.json"
    elif args.data_type == "article":
        EMBEDDINGS_PATH = "data/embeddings_news.npy"
        CLUSTER_DICT_PATH = "data/cluster_dicts_news.json"
        LABEL_PROMPT_PATH = "prompt/news_label_prompt.txt"
        FINAL_OUTPUT_PATH = "data/final_output_news.json"
    else:
        raise Exception("data_type must be in ['ttht', 'article']")

    with open(JSON_INPUT) as f:
        data = json.load(f)[:10_000]
    documents = [i["text"] for i in data]

    print("Getting embeddings...")
    if os.path.exists(EMBEDDINGS_PATH) and not args.overwrite:
        embeddings = np.load(EMBEDDINGS_PATH)
    else:
        embeddings = get_embeddings(
            documents=documents,
            api_key=VNSOCIAL_API_KEY,
            api_base=VNSOCIAL_API_BASE,
        )
        np.save(EMBEDDINGS_PATH, embeddings)

    cluster_dicts = get_clusters(
        documents=documents, embeddings=embeddings, data_type=args.data_type
    )
    cluster_dicts = cluster_dicts[
        :10
    ]  # Get TOP 10 clusters with the most number of members
    final_dict = create_label(
        api_key=os.getenv("OPENAI_API_KEY"),
        prompt_path=LABEL_PROMPT_PATH,
        clusters_dicts=cluster_dicts,
    )
    with open(FINAL_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_dict, f, ensure_ascii=False, indent=4)
    total_final_texts = sum([len(i["texts"]) for i in final_dict])
    print("Total number of texts (in top 10 clusters):", total_final_texts)
