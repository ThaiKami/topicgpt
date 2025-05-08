import numpy as np
import hdbscan
import umap
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import defaultdict
import json

# Load the saved embeddings
data = np.load("data/embeddings.npz", allow_pickle=True)
with open("data/input/news-100.json") as f:
    texts_data = json.load(f)
ids = data["ids"]  # list of document IDs
embeddings = data["embeddings"]  # shape (N, D)


# Reduce dimensions with UMAP for visualization
reducer_15d = umap.UMAP(
    n_components=30,
    n_neighbors=5,  # adjust: small for local, larger for global
    min_dist=0.0,
    metric="cosine",
    random_state=42,
)
embeddings_15d = reducer_15d.fit_transform(embeddings)

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=2,  # increase to merge small clusters
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,
)
cluster_labels = clusterer.fit_predict(embeddings_15d)

# 4. UMAP → 2 dimensions for plotting
reducer_2d = umap.UMAP(
    n_components=2, n_neighbors=5, min_dist=0.0, metric="cosine", random_state=42
)
embedding_2d = reducer_2d.fit_transform(embeddings)

# 5. Plot
plt.figure(figsize=(10, 8))
palette = cm.get_cmap("tab20", cluster_labels.max() + 2)
for label in np.unique(cluster_labels):
    mask = cluster_labels == label
    color = "lightgrey" if label == -1 else palette(label)
    plt.scatter(
        embedding_2d[mask, 0],
        embedding_2d[mask, 1],
        s=12,
        c=[color],
        label="Noise" if label == -1 else f"Cluster {label}",
        alpha=0.6,
    )
plt.title("HDBSCAN over UMAP-15D, visualized in UMAP-2D")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.legend(markerscale=2, fontsize="small", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(
    "hdbscan_umap_clusters.png",  # filename (can be .png, .pdf, .svg, .eps, etc.)
    dpi=300,  # resolution in dots per inch
    bbox_inches="tight",  # trim extra whitespace
    transparent=False,  # if True, background will be transparent
)
# 5. Print cluster counts
unique, counts = np.unique(cluster_labels, return_counts=True)
cluster_counts = dict(zip(unique, counts))
print("Cluster counts:", cluster_counts)

# 6. Build cluster → IDs dict
cluster_dict = defaultdict(list)
for doc_id, label in zip(ids, cluster_labels):
    cluster_dict[int(label)].append(doc_id)

# If you prefer a plain dict (rather than defaultdict):
cluster_dict = dict(cluster_dict)
cluster_sizes = {label: len(ids) for label, ids in cluster_dict.items()}
top10_labels = sorted(cluster_sizes, key=cluster_sizes.get, reverse=True)[:10]
top10_clusters = {label: cluster_dict[label] for label in top10_labels}
top10_ordered = [(label, cluster_dict[label]) for label in top10_labels]
cluster_ids_dict = {label: id_list for label, id_list in top10_ordered}
final_dict = {}
for cluster_id, ids in cluster_ids_dict.items():
    input_texts = [text["text"][:] for text in texts_data if text["docId"] in ids]
    final_dict[cluster_id] = {"texts": input_texts, "numTexts": len(input_texts)}
# 7. (Optional) Save to JSON file
with open("data/clusters_top10_sorted.json", "w", encoding="utf-8") as f:
    json.dump(final_dict, f, ensure_ascii=False, indent=2)
