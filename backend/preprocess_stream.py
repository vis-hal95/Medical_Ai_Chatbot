# backend/preprocess_stream.py
"""
Stream preprocess for large CSVs. Reads CSV in chunks, extracts user/doctor pairs, writes train/test jsonl
Adjust CSV_COLUMN heuristics if your CSV uses different headers.
"""
import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DATA_DIR = Path("data")
OUT_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True, parents=True)
CSV_FILENAME = "AI-MEDICAL-CHATBOT.csv"  # replace with your file name
CHUNKSIZE = 50_000  # number of rows per read chunk; tune for memory

def guess_columns(df: pd.DataFrame):
    # heuristic to find user and doctor columns
    cols = df.columns.tolist()
    user_cols = [c for c in cols if 'patient' in c.lower() or 'user' in c.lower() or 'question' in c.lower()]
    doc_cols = [c for c in cols if 'doctor' in c.lower() or 'response' in c.lower() or 'answer' in c.lower()]
    if user_cols and doc_cols:
        return user_cols[0], doc_cols[0]
    # fallback: use first two object/string columns
    obj_cols = [c for c in cols if pd.api.types.is_string_dtype(df[c])]
    if len(obj_cols) >= 2:
        return obj_cols[0], obj_cols[1]
    # last resort
    return cols[0], cols[1]

def stream_extract(csv_path: Path, out_train: Path, out_test: Path, sample_frac=0.05):
    # We'll produce a single combined jsonl then split at the end to avoid storing entire dataset in memory
    temp_jsonl = OUT_DIR / "all_pairs.jsonl"
    if temp_jsonl.exists():
        temp_jsonl.unlink()

    total_pairs = 0
    with pd.read_csv(csv_path, chunksize=CHUNKSIZE, iterator=True, encoding='utf-8', low_memory=True) as reader:
        for chunk in tqdm(reader, desc="Reading CSV chunks"):
            try:
                user_col, doc_col = guess_columns(chunk)
            except Exception:
                # fallback in case chunk bad
                user_col, doc_col = chunk.columns[0], chunk.columns[1]
            # drop rows with NaN in those columns
            chunk = chunk.dropna(subset=[user_col, doc_col])
            pairs = []
            for _, row in chunk.iterrows():
                u = str(row[user_col]).strip()
                d = str(row[doc_col]).strip()
                if len(u) > 2 and len(d) > 2:
                    pairs.append({"input": u, "response": d})
            # append to temp
            with open(temp_jsonl, "a", encoding="utf-8") as f:
                for p in pairs:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
            total_pairs += len(pairs)
    print(f"Extracted total {total_pairs} dialog pairs to {temp_jsonl}")

    # read temp and split into train/test streaming
    all_lines = []
    with open(temp_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            all_lines.append(json.loads(line))
    train, test = train_test_split(all_lines, test_size=sample_frac, random_state=42)
    def write_jsonl(lst, path):
        with open(path, "w", encoding="utf-8") as f:
            for x in lst:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

    write_jsonl(train, out_train)
    write_jsonl(test, out_test)
    print(f"Saved train {len(train)} and test {len(test)} to {out_train}, {out_test}")

if __name__ == "__main__":
    csv = DATA_DIR / CSV_FILENAME
    if not csv.exists():
        raise FileNotFoundError(f"Place your CSV at backend/data/{CSV_FILENAME}")
    stream_extract(csv, OUT_DIR / "train.jsonl", OUT_DIR / "test.jsonl", sample_frac=0.05)