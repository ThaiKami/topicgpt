import numpy as np
import json
from collections import defaultdict
import hdbscan
import umap

# Load data
data = np.load("data/texts_and_embeddings_news.npz", allow_pickle=True)
texts_loaded = data["texts"]  # 1-D array of strings
embeddings_loaded = data["embeddings"]  # 2-D float16 array

# UMAP dimensionality reduction
reducer = umap.UMAP(
    n_components=50,
    n_neighbors=10,  # Adjust: small for local, larger for global
    min_dist=0.0,
    metric="cosine",
    random_state=42,
)

embeddings = reducer.fit_transform(embeddings_loaded)

# HDBSCAN clustering
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=5,  # Smallest grouping you consider a cluster
    metric="euclidean",  # Distance metric
    cluster_selection_method="eom",
    prediction_data=True,
)
labels = clusterer.fit_predict(embeddings)

# Group texts by cluster
text_cluster_pairs = list(zip(texts_loaded, labels))
clusters = defaultdict(list)
for text, cluster_id in text_cluster_pairs:
    if cluster_id != -1:  # Exclude noise points
        clusters[int(cluster_id)].append(text)

# Sort clusters by size
clusters = list(sorted(clusters.items(), key=lambda item: len(item[1]), reverse=True))
clusters_dicts = [
    {"id": cluster_id, "texts": texts} for cluster_id, texts in clusters
]
# Save clusters to JSON
output_path = "data/output/hbdscan_news.json"
try:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clusters_dicts, f, ensure_ascii=False, indent=4)
    print(f"Clusters saved to {output_path}")
except Exception as e:
    print(f"Error saving clusters to {output_path}: {e}")
