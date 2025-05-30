import json
import os
import random
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

import numpy as np
from tqdm import tqdm
import hdbscan
import umap
from sklearn.cluster import DBSCAN
from openai import OpenAI


class HotTopicPipeline:
    def __init__(
        self,
        vnsocial_api_key: str,
        vnsocial_api_base: str,
        openai_api_key: str,
        input_path: Path,
        prompt_path: Path,
        output_path: Path,
        method: str = "article",
        top_k: int = 10,
        overwrite_embeddings: bool = False,
    ):
        random.seed(42)
        self.method = method
        self.top_k = top_k
        self.input_path = input_path
        self.prompt_path = prompt_path
        self.output_path = output_path
        self.embeddings_path = (
            Path("data/embeddings_news.npy")
            if method == "article"
            else Path("data/embeddings_TTHT.npy")
        )
        self.client_embedding = OpenAI(
            api_key=vnsocial_api_key, base_url=vnsocial_api_base
        )
        self.client_chat = OpenAI(api_key=openai_api_key)
        self.overwrite_embeddings = overwrite_embeddings

    def load_documents(self) -> List[str]:
        with open(self.input_path) as f:
            data = json.load(f)
        return [item["TITLE"] for item in data][:]

    def get_embeddings(self, docs: List[str]) -> np.ndarray:
        if self.embeddings_path.exists() and not self.overwrite_embeddings:
            return np.load(self.embeddings_path)

        embeddings = []
        for doc in tqdm(docs, desc="Getting embeddings"):
            resp = self.client_embedding.embeddings.create(
                input=doc,
                model="vnsocial_embedding",
            )
            vec = np.array(resp.data[0].embedding, dtype=np.float32)
            embeddings.append(vec)
        embeddings = np.stack(embeddings, axis=0)
        np.save(self.embeddings_path, embeddings)
        return embeddings

    def cluster(self, docs: List[str], embeddings: np.ndarray) -> List[Dict]:
        if self.method == "article":
            clusterer = DBSCAN(eps=0.2, min_samples=5, metric="cosine")
            labels = clusterer.fit_predict(embeddings)
        else:
            reducer = umap.UMAP(
                n_components=50,
                n_neighbors=10,
                min_dist=0.0,
                metric="cosine",
                random_state=42,
            )
            reduced = reducer.fit_transform(embeddings)
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=50,
                min_samples=10,
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=True,
            )
            labels = clusterer.fit_predict(reduced)

        clusters_map = defaultdict(list)
        for text, lbl in zip(docs, labels):
            if lbl != -1:
                clusters_map[int(lbl)].append(text)

        sorted_clusters = sorted(
            clusters_map.items(), key=lambda kv: len(kv[1]), reverse=True
        )
        return [
            {"id": cid, "texts": texts} for cid, texts in sorted_clusters[: self.top_k]
        ]

    def label_clusters(self, clusters: List[Dict]) -> List[Dict]:
        with open(self.prompt_path) as f:
            template = f.read()
        labeled = []
        for cluster in tqdm(clusters, desc="Labeling"):
            sample = random.sample(cluster["texts"], min(5, len(cluster["texts"])))
            truncated = [t[:512] for t in sample]
            prompt = template.format(Document="\n\n".join(truncated))
            resp = self.client_chat.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=1.0,
                top_p=1.0,
            )
            labeled.append(
                {
                    "id": cluster["id"],
                    "topic": resp.choices[0].message.content,
                    "numTexts": len(cluster["texts"]),
                    "texts": cluster["texts"],
                }
            )
        return labeled

    def run(self):
        docs = self.load_documents()
        embeddings = self.get_embeddings(docs)
        clusters = self.cluster(docs, embeddings)
        final = self.label_clusters(clusters)
        with open(self.output_path, "w") as f:
            json.dump(final, f, ensure_ascii=False, indent=4)
        total = sum(len(c["texts"]) for c in final)
        print(f"Total number of texts (in top {self.top_k} clusters): {total}")


if __name__ == "__main__":
    VNS_API_KEY = os.getenv("VNSOCIAL_API_KEY", "VnSocial")
    VNS_API_BASE = os.getenv("VNSOCIAL_API_BASE", "http://10.159.19.101:30043/v1")
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    input_path = "data/input/news_data_200k/jsons/2020-06-29.json"
    prompt_path = "prompt/news_label_prompt.txt"
    output_path = "data/lmao.json"
    method = "article"
    top_k = 10
    overwrite_embeddings = True
    pipeline = HotTopicPipeline(
        vnsocial_api_key=VNS_API_KEY,
        vnsocial_api_base=VNS_API_BASE,
        openai_api_key=OPENAI_KEY,
        input_path=input_path,
        prompt_path=prompt_path,
        output_path=output_path,
        method=method,
        top_k=top_k,
        overwrite_embeddings=overwrite_embeddings,
    )
    pipeline.run()
