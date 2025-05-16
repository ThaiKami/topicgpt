from sentence_transformers import SentenceTransformer
import json
import numpy as np
from tqdm import tqdm
import argparse
from pyvi.ViTokenizer import tokenize


def compute_embeddings_sbert(
    input_file,
    output_file,
    model_name="dangvantuan/vietnamese-embedding",
    batch_size=32,
):
    """
    Reads a JSONL corpus, computes embeddings for each text, and saves them.

    Arguments:
    - input_file: Path to the input .jsonl file. Each line should be a JSON object
                  with at least "id" and "text" fields.
    - output_file: Path to save the embeddings (.npz format).
    - model_name:  SentenceTransformer model to use.
    - batch_size:  Number of texts to process per batch.
    """
    # 1. Load IDs and texts
    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)
    ids, texts = [], []
    for obj in tqdm(data, desc="Loading data"):
        ids.append(obj["docId"])
        texts.append(tokenize(obj["text"]))

    # 2. Load the embedding model
    print(f"Loading model '{model_name}'...")
    model = SentenceTransformer(model_name, trust_remote_code=True)
    model._first_module().max_seq_length = 512
    # 3. Compute embeddings in batches
    print(f"Computing embeddings for {len(texts)} documents...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        max_length=256,
        truncation=True,
        padding="max_length",
    )

    # 4. Save to .npz
    print(f"Saving embeddings to '{output_file}'...")
    np.savez(output_file, ids=np.array(ids), embeddings=embeddings)
    print("Done.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Compute sentence embeddings for a JSONL corpus."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="data/input/news-100.json",
        help="Path to input JSONL file (one JSON object per line with 'id' and 'text').",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/embeddings.npz",
        help="Path to output .npz file (will contain 'ids' and 'embeddings').",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="dangvantuan/vietnamese-document-embedding",
        help="Name of the SentenceTransformer model to use.",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=32,
        help="Batch size for embedding computation.",
    )
    args = parser.parse_args()

    compute_embeddings_sbert(
        input_file=args.input,
        output_file=args.output,
        model_name=args.model,
        batch_size=args.batch_size,
    )
