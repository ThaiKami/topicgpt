from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
import json
import numpy as np
from tqdm import tqdm
import argparse
from pyvi.ViTokenizer import tokenize


def compute_embeddings_transformers(
    input_file, output_file, model_name, batch_size, device
):
    """
    Reads a JSONL corpus, computes mean-pooled embeddings for each text
    using plain transformers, and saves them to a .npz file.
    """
    # 1. Read in IDs & texts
    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)
    ids, texts = [], []
    for obj in data:
        ids.append(obj["docId"])
        texts.append(obj["text"])

    # 2. Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # 3. Batch & encode
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc, return_dict=True)
            last_hidden = out.last_hidden_state  # [B, T, D]

        # 4. Mean-pool, masking out pads
        attention_mask = enc["attention_mask"].unsqueeze(-1)  # [B, T, 1]
        sum_embeddings = torch.sum(last_hidden * attention_mask, dim=1)
        lengths = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        batch_embs = sum_embeddings / lengths  # [B, D]

        all_embs.append(batch_embs.cpu().numpy())

    # 5. Stack & save
    embeddings = np.vstack(all_embs)  # [N, D]
    np.savez(output_file, ids=ids, embeddings=embeddings)
    print(f"Saved {len(ids)} embeddings to {output_file}")


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
    for obj in data:
        ids.append(obj["docId"])
        texts.append(tokenize(obj["text"]))

    # 2. Load the embedding model
    print(f"Loading model '{model_name}'...")
    model = SentenceTransformer(model_name)
    model._first_module().max_seq_length = 256
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
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-i", "--input", default="data/input/news-100.json")
    # parser.add_argument("-o", "--output", default="data/embs_transformers.npz")
    # parser.add_argument("-m", "--model", default="FacebookAI/xlm-roberta-large")
    # parser.add_argument("-b", "--batch_size", type=int, default=32)
    # parser.add_argument("-d", "--device", default="cuda")
    # args = parser.parse_args()

    # compute_embeddings_transformers(
    #     args.input, args.output, args.model, args.batch_size, args.device
    # )

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
        default="dangvantuan/vietnamese-embedding",
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
