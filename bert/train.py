import json
from sentence_transformers import InputExample, SentenceTransformer, losses, evaluation
from sklearn.model_selection import train_test_split
from pyvi.ViTokenizer import tokenize
import os
from torch.utils.data import DataLoader

# 2. Paths & model
DATASET_PATH = "data/label/llm_pair_label.jsonl"
MODEL_PATH = "dangvantuan/vietnamese-embedding"

# 3. Load examples
examples = []

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        label = float(entry["label"])
        examples.append(
            InputExample(
                texts=[
                    tokenize(entry["text1"][:512]),
                    tokenize(entry["text2"][:512]),
                ],
                label=label,
            )
        )
# 4. Train/val split
zero = [i for i in examples if i.label == 0]
one = [i for i in examples if i.label == 1]
if len(zero) > len(one):
    zero = zero[: len(one)]
elif len(one) > len(zero):
    one = one[: len(zero)]
examples = zero + one
train_examples, val_examples = train_test_split(
    examples, test_size=0.1, random_state=42
)

# 5. Load SBERT & set max length
model = SentenceTransformer(MODEL_PATH, trust_remote_code=True)

# 6. DataLoader with smart batching
train_dataloader = DataLoader(
    train_examples, shuffle=True, batch_size=16, collate_fn=model.smart_batching_collate
)

# 7. Contrastive loss
train_loss = losses.ContrastiveLoss(model)

# 8. Evaluator
evaluator = evaluation.EmbeddingSimilarityEvaluator.from_input_examples(
    val_examples, name="val-eval"
)

# 9. Training
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=3,
    evaluation_steps=1000,
    warmup_steps=100,
    output_path="model/contrastiveloss_model",
)
