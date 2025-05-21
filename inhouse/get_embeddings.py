from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json
from pyvi.ViTokenizer import tokenize
from tqdm import tqdm
from collections import defaultdict
import hdbscan
import umap

# 1. Load model
model = SentenceTransformer("model/contrastiveloss_model", trust_remote_code=True)
data = json.load(open("data/input/report-voice-20250101.json"))
# 2. Encode texts
documents = [
    tokenize(i["text"][:512].replace("\n", " "))
    for i in tqdm(data[:], desc="Loading data")
]  # your list of strings
embeddings = model.encode(
    documents, batch_size=32, convert_to_numpy=True, show_progress_bar=True
)
reducer = umap.UMAP(
    n_components=50,
    n_neighbors=10,  # adjust: small for local, larger for global
    min_dist=0.0,
    metric="cosine",
    random_state=42,
)
embeddings = reducer.fit_transform(embeddings)

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=50,  # smallest grouping you consider a cluster
    metric="euclidean",  # distance metric
    cluster_selection_method="eom",
    prediction_data=True,
)
labels = clusterer.fit_predict(embeddings)
text_cluster_pairs = list(zip(documents, labels))

clusters = defaultdict(list)
for text, cluster_id in text_cluster_pairs:
    if cluster_id != -1:  # Exclude noise points
        clusters[int(cluster_id)].append(text)

clusters = dict(sorted(clusters.items(), key=lambda item: len(item[1]), reverse=True))

json.dump(
    clusters,
    open("data/output/hbdscan.json", "w", encoding="utf-8"),
    ensure_ascii=False,
    indent=4,
)
