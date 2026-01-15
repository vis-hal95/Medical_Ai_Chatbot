# # backend/app.py
# from fastapi import FastAPI
# from pydantic import BaseModel
# from sentence_transformers import SentenceTransformer
# import faiss, json, os
# from pathlib import Path
# from typing import List
# import openai

# # Config
# INDEX_PATH = Path("models/faiss.index")
# META_PATH = Path("models/meta.jsonl")
# EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
# K = 5  # top-k retrieval

# OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
# LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")  # change if you have different access

# if OPENAI_KEY:
#     openai.api_key = OPENAI_KEY

# app = FastAPI(title="Medical RAG Chatbot (demo)")

# # load embedding model & index lazily
# embed_model = SentenceTransformer(EMBED_MODEL_NAME)
# _index = None
# _metas = None

# def load_index_and_meta():
#     global _index, _metas
#     if _index is None:
#         if not INDEX_PATH.exists() or not META_PATH.exists():
#             raise FileNotFoundError("Index or meta not found. Run build_index_stream.py first.")
#         _index = faiss.read_index(str(INDEX_PATH))
#         # load metas into list
#         with open(META_PATH, "r", encoding="utf-8") as f:
#             _metas = [json.loads(line) for line in f]
#     return _index, _metas

# def retrieve(query: str, k=K):
#     index, metas = load_index_and_meta()
#     q_emb = embed_model.encode([query], convert_to_numpy=True)
#     faiss.normalize_L2(q_emb)
#     D, I = index.search(q_emb, k)
#     results = []
#     for idx in I[0]:
#         if idx < len(metas):
#             m = metas[idx].copy()
#             results.append(m)
#     return results

# def build_prompt(query: str, contexts: List[dict]):
#     ctx_texts = "\n\n".join([f"[{i+1}] {c.get('response','')}" for i,c in enumerate(contexts)])
#     prompt = (
#         "You are a helpful medical assistant for demonstration only. "
#         "Use only the retrieved snippets below to answer the user's question. "
#         "If the information is insufficient or ambiguous, explicitly say so and advise consulting a medical professional. "
#         f"\n\nRetrieved snippets:\n{ctx_texts}\n\nUser question: {query}\n\nAnswer concisely and cite snippet numbers when applicable."
#     )
#     return prompt

# class Query(BaseModel):
#     message: str
#     history: List[dict] = []

# @app.post("/api/chat")
# def chat(q: Query):
#     retrieved = retrieve(q.message, k=K)
#     prompt = build_prompt(q.message, retrieved)

#     # If OpenAI key is present, call API; otherwise, return retrieved snippets as answer (fallback)
#     if OPENAI_KEY:
#         try:
#             resp = openai.ChatCompletion.create(
#                 model=LLM_MODEL,
#                 messages=[
#                     {"role":"system","content":"You are a helpful medical assistant (demo)."},
#                     {"role":"user","content": prompt}
#                 ],
#                 temperature=0.0,
#                 max_tokens=512
#             )
#             text = resp["choices"][0]["message"]["content"]
#         except Exception as e:
#             text = f"[LLM call failed: {e}]\n\nRetrieved snippets:\n" + "\n\n".join([s.get("response","") for s in retrieved])
#     else:
#         # fallback to deterministic reply using retrieved snippets
#         text = "No LLM provider configured. Here are top retrieved snippets:\n\n" + "\n\n".join([f"[{i+1}] {s.get('response','')}" for i,s in enumerate(retrieved)])
#     return {"answer": text, "retrieved": retrieved}

# app.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import faiss
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import openai

# ----------------------------
# Initialize FastAPI
# ----------------------------
app = FastAPI(title="Medical AI Chatbot Backend")

# ----------------------------
# Add CORS middleware
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins (for development). Replace with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Define request schema
# ----------------------------
class ChatRequest(BaseModel):
    message: str
    top_k: int = 5  # number of retrieved snippets

# ----------------------------
# Load FAISS index and metadata
# ----------------------------
INDEX_PATH = "models/faiss.index"
META_PATH = "models/meta.jsonl"

if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
    raise FileNotFoundError("FAISS index or metadata not found. Run build_index_stream.py first.")

# Load FAISS index
faiss_index = faiss.read_index(INDEX_PATH)

# Load metadata
meta = []
with open(META_PATH, "r", encoding="utf-8") as f:
    for line in f:
        meta.append(json.loads(line))

# Load sentence-transformer model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# Optional: OpenAI API key
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# ----------------------------
# Utility function: retrieve top-k relevant snippets
# ----------------------------
def retrieve_snippets(query: str, top_k: int = 5):
    query_vec = embedding_model.encode([query], normalize_embeddings=True)
    distances, indices = faiss_index.search(query_vec, top_k)
    
    results = []
    for idx in indices[0]:
        if idx < len(meta):
            results.append(meta[idx]["response"])  # assuming your dataset has "response" field
    return results

# ----------------------------
# Chat endpoint
# ----------------------------
@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    query = request.message
    top_k = request.top_k
    
    # Step 1: retrieve relevant snippets
    snippets = retrieve_snippets(query, top_k)
    
    # Step 2: optionally, use OpenAI LLM to generate answer
    if OPENAI_API_KEY:
        prompt = f"Answer the question using the following snippets:\n\n{snippets}\n\nQuestion: {query}\nAnswer:"
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=200,
                temperature=0.7,
            )
            answer = response.choices[0].text.strip()
        except Exception as e:
            answer = "Error generating answer with OpenAI API."
    else:
        # fallback: just return retrieved snippets
        answer = "\n".join(snippets)
    
    return {
        "query": query,
        "answer": answer,
        "retrieved_snippets": snippets
    }

# ----------------------------
# Health check
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}