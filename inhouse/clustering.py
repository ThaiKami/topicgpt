import json
from collections import defaultdict
from pathlib import Path

import hdbscan
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import umap


def load_data(embedding_file: Path, texts_file: Path):
    data = np.load(embedding_file, allow_pickle=True)
    ids = data["ids"]
    embeddings = data["embeddings"]
    with texts_file.open(encoding="utf-8") as f:
        texts_data = json.load(f)
    return ids, embeddings, texts_data


def reduce_dimensions(embeddings: np.ndarray):
    # Reduce to 15 dimensions for clustering
    reducer_15d = umap.UMAP(
        n_components=10,
        n_neighbors=10,  # adjust: small for local, larger for global
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    embeddings_15d = reducer_15d.fit_transform(embeddings)

    # Reduce to 2 dimensions for plotting
    reducer_2d = umap.UMAP(
        n_components=2,
        n_neighbors=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    embedding_2d = reducer_2d.fit_transform(embeddings)
    return embeddings_15d, embedding_2d


def cluster_embeddings(embeddings_15d: np.ndarray):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=50,  # increase to merge small clusters
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    return clusterer.fit_predict(embeddings_15d)


def plot_clusters(
    embedding_2d: np.ndarray, cluster_labels: np.ndarray, output_file: Path
):
    plt.figure(figsize=(10, 8))
    palette = cm.get_cmap("tab20", cluster_labels.max() + 2)
    for label in np.unique(cluster_labels):
        mask = cluster_labels == label
        color = "lightgrey" if label == -1 else palette(label)
        label_name = "Noise" if label == -1 else f"Cluster {label}"
        plt.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            s=12,
            c=[color],
            label=label_name,
            alpha=0.6,
        )
    plt.title("HDBSCAN over UMAP-15D, visualized in UMAP-2D")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(
        markerscale=2, fontsize="small", bbox_to_anchor=(1.05, 1), loc="upper left"
    )
    plt.tight_layout()
    plt.savefig(str(output_file), dpi=300, bbox_inches="tight", transparent=False)
    plt.close()


def build_cluster_dict(ids, cluster_labels):
    cluster_dict = defaultdict(list)
    for doc_id, label in zip(ids, cluster_labels):
        cluster_dict[int(label)].append(doc_id)
    cluster_dict.pop(-1, None)  # Remove noise label if present
    return cluster_dict


def get_top_clusters(cluster_dict: dict, top_n: int = 10):
    cluster_sizes = {label: len(doc_ids) for label, doc_ids in cluster_dict.items()}
    top_labels = sorted(cluster_sizes, key=cluster_sizes.get, reverse=True)[:top_n]
    return {label: cluster_dict[label] for label in top_labels}


def build_final_dict(top_clusters: dict, texts_data: list):
    # Build an index for faster lookup by docId
    texts_lookup = {text["docId"]: text.get("text", "") for text in texts_data}
    final_dict = {}
    for cluster_id, doc_ids in top_clusters.items():
        input_texts = [
            texts_lookup.get(doc_id, "") for doc_id in doc_ids if doc_id in texts_lookup
        ]
        final_dict[cluster_id] = {"texts": input_texts, "numTexts": len(input_texts)}
    return final_dict


def save_json(data: dict, output_file: Path):
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    base_dir = Path("data")
    embedding_file = base_dir / "embeddings.npz"
    texts_file = base_dir / "input" / "report-voice-20250101.json"
    output_plot = Path("hdbscan_umap_clusters.png")
    output_json = base_dir / "clusters_top10_sorted.json"

    ids, embeddings, texts_data = load_data(embedding_file, texts_file)
    embeddings_15d, embedding_2d = reduce_dimensions(embeddings)
    cluster_labels = cluster_embeddings(embeddings)

    unique, counts = np.unique(cluster_labels, return_counts=True)
    print("Cluster counts:", dict(zip(unique, counts)))

    plot_clusters(embedding_2d, cluster_labels, output_plot)

    cluster_dict = build_cluster_dict(ids, cluster_labels)
    top_clusters = get_top_clusters(cluster_dict, top_n=10)
    final_dict = build_final_dict(top_clusters, texts_data)
    save_json(final_dict, output_json)


if __name__ == "__main__":
    main()
