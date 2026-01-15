# backend/build_index_stream.py
"""
Builds a FAISS index incrementally from the train.jsonl file using SentenceTransformer.
Supports large datasets via chunked encoding + incremental index.add().
Stores meta.jsonl (same order as vectors added).
"""
from sentence_transformers import SentenceTransformer
import faiss
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
from utils import save_meta_append, read_meta

MODEL_NAME = "all-MiniLM-L6-v2"  # small & fast; can change to larger model if you have GPU
TRAIN_JSONL = Path("models/train.jsonl")
META_JSONL = Path("models/meta.jsonl")
INDEX_PATH = Path("models/faiss.index")
BATCH_SIZE = 256  # embedding batch size; reduce if OOM

def stream_docs(jsonl_path: Path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def build_index():
    if not TRAIN_JSONL.exists():
        raise FileNotFoundError("Run preprocess_stream.py to create models/train.jsonl")

    # init model
    embed_model = SentenceTransformer(MODEL_NAME)
    # load existing metas if resuming
    if META_JSONL.exists():
        existing_metas = read_meta(META_JSONL)
        print(f"Resuming: found {len(existing_metas)} existing metas.")
    else:
        existing_metas = []

    # If resuming, load index, otherwise create new
    if INDEX_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
        print("Loaded existing FAISS index.")
    else:
        # infer embedding dim
        dim = embed_model.get_sentence_embedding_dimension()
        index = faiss.IndexFlatIP(dim)  # use inner product after normalization
        print(f"Created new FAISS IndexFlatIP with dim={dim}")

    buffer_texts = []
    buffer_metas = []
    count = len(existing_metas)
    pbar_total = None

    # count total lines for progress (optional)
    with open(TRAIN_JSONL, "r", encoding="utf-8") as f:
        pbar_total = sum(1 for _ in f) - count

    print(f"Indexing {pbar_total} new documents (skipping {count} already in meta).")

    with open(TRAIN_JSONL, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, total=pbar_total)):
            if i < count:
                continue  # skip already-processed
            obj = json.loads(line)
            text = obj.get("response", "")
            buffer_texts.append(text)
            buffer_metas.append(obj)
            # when buffer full, encode & add
            if len(buffer_texts) >= BATCH_SIZE:
                emb = embed_model.encode(buffer_texts, convert_to_numpy=True, show_progress_bar=False)
                # normalize for IP similarity
                faiss.normalize_L2(emb)
                index.add(emb)
                save_meta_append(META_JSONL, buffer_metas)
                count += len(buffer_texts)
                buffer_texts, buffer_metas = [], []

        # final flush
        if buffer_texts:
            emb = embed_model.encode(buffer_texts, convert_to_numpy=True, show_progress_bar=False)
            faiss.normalize_L2(emb)
            index.add(emb)
            save_meta_append(META_JSONL, buffer_metas)
            count += len(buffer_texts)

    # write index out
    faiss.write_index(index, str(INDEX_PATH))
    print(f"Indexing complete. Total vectors: {count}. Index saved to {INDEX_PATH}")

if __name__ == "__main__":
    build_index()