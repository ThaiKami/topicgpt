import json
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os


def load_topics(json_path):
    """Load topics from the JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        topics_data = json.load(f)
    # return [item["topic"] for item in topics_data]
    return topics_data


def get_embeddings(
    topics,
    model_name="dangvantuan/vietnamese-embedding",
    max_seq_length=256,
    batch_size=16,
):
    """Compute embeddings for the list of topics."""
    model = SentenceTransformer(model_name)
    topics = [item["topic"] for item in topics]
    # Set the maximum sequence length for the first module (if applicable)
    if hasattr(model, "_first_module") and callable(model._first_module):
        model._first_module().max_seq_length = max_seq_length
    embeddings = model.encode(
        topics,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return embeddings


def cluster_embeddings(embeddings, eps=0.2, min_samples=2):
    """Cluster embeddings using DBSCAN with cosine metric."""
    # eps is a cosine distance threshold (1 - similarity)
    db = DBSCAN(metric="cosine", eps=eps, min_samples=min_samples)
    return db.fit_predict(embeddings)


def group_topics_by_label(labels, topics):
    """Group topics by their assigned DBSCAN labels."""
    groups = {}
    for label, topic in zip(labels, topics):
        groups.setdefault(label, []).append(topic)
    return groups


def print_clusters(groups):
    """Print clusters of topics."""
    for label, topics in groups.items():
        cluster_name = f"Cluster {label}" if label != -1 else "Noise"
        print(f"{cluster_name}:")
        for topic in topics:
            print("  ", topic)
        print()


def print_clusters(
    groups,
    LLMclient,
    prompt_path="prompt/merge_prompt.txt",
    json_path="data/final_output.json",
):
    """Print clusters of topics."""
    data = json.load(open(json_path, "r", encoding="utf-8"))
    with open(prompt_path, encoding="utf-8") as f:
        base_prompt = f.read()
    merged_topics = []
    for label, topics in groups.items():
        if label == -1:
            continue
        else:
            prompt = base_prompt.format(Document="\n\n".join(topics))
            completion = LLMclient.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.0,
                top_p=1.0,
            )
            merged_topics.append(completion.choices[0].message.content)

            for topic in topics:
                for topic_data in data:
                    if topic_data["topic"] == topic:
                        topic_data["topic"] = completion.choices[0].message.content
                        # break


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    json_path = "data/final_output.json"
    topics = load_topics(json_path)
    embeddings = get_embeddings(topics)
    labels = cluster_embeddings(embeddings)
    groups = group_topics_by_label(labels, topics)
    print_clusters(groups)


if __name__ == "__main__":
    main()
