from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
import random
import torch
import json
from tqdm import tqdm

# 0. Your corpus
with open("data/input/report-voice-20250101.json") as f:
    texts = [item["text"] for item in json.load(f)[:1000]]
# 1. Encode & pseudo-label
base_model = SentenceTransformer(
    "dangvantuan/vietnamese-document-embedding", trust_remote_code=True
)
embs = base_model.encode(
    texts, convert_to_numpy=True, show_progress_bar=True, batch_size=16
)

n_clusters = 10  # pick via elbow/silhouette or domain knowledge
km = KMeans(n_clusters=n_clusters, random_state=0).fit(embs)
labels = km.labels_  # array of length len(texts)


# 2. Sample triplets from pseudo-labels
def sample_triplets(texts, labels, num_triplets=2000):
    idx_by_label = {}
    for idx, lbl in enumerate(labels):
        idx_by_label.setdefault(lbl, []).append(idx)

    all_labels = list(idx_by_label.keys())
    triplets = []
    for _ in range(num_triplets):
        # choose anchor cluster
        anchor_lbl = random.choice(all_labels)
        neg_lbl = random.choice([l for l in all_labels if l != anchor_lbl])

        a, p = random.sample(idx_by_label[anchor_lbl], 2)
        n = random.choice(idx_by_label[neg_lbl])

        triplets.append(InputExample(texts=[texts[a], texts[p], texts[n]]))
    return triplets


train_examples = sample_triplets(texts, labels, num_triplets=3000)
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# 3. Build a fresh SBERT (transformer + pooling)
word_emb = models.Transformer("FacebookAI/xlm-roberta-base")
pool = models.Pooling(word_emb.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_emb, pool])

# 4. Triplet loss
train_loss = losses.TripletLoss(
    model=model, distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=0.5
)

# 5. Fine-tune
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=3, warmup_steps=100)

# 6. Save your tuned model
model.save("model/roberta-triplet")

# 7. (Optional) Re-embed & re-cluster for another round
new_embs = model.encode(texts, convert_to_numpy=True)
new_km = KMeans(n_clusters=n_clusters, random_state=0).fit(new_embs)
new_labels = new_km.labels_
# …you could sample new triplets from new_labels and repeat…
