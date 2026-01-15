# backend/utils.py
from typing import List, Dict
import json
from pathlib import Path

def save_meta_append(meta_path: Path, metas: List[Dict]):
    with open(meta_path, "a", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

def read_meta(meta_path: Path):
    metas = []
    if not meta_path.exists():
        return metas
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            metas.append(json.loads(line))
    return metas