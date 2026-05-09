import os
import json
import pickle
import re
import numpy as np
import faiss
from google import genai

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
_client = genai.Client(api_key=GEMINI_API_KEY)

_catalog = None
_index = None


def _load():
    global _catalog, _index
    if _catalog is None:
        with open("catalog_meta.pkl", "rb") as f:
            _catalog = pickle.load(f)
        _index = faiss.read_index("catalog.faiss")


def retrieve(query: str, k: int = 15) -> list[dict]:
    _load()
    response = _client.models.embed_content(
        model="text-embedding-004",
        contents=[query],
    )
    vec = np.array(response.embeddings[0].values, dtype=np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-9)
    vec = vec.reshape(1, -1)
    scores, indices = _index.search(vec, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            item = dict(_catalog[idx])
            item["_score"] = float(score)
            results.append(item)
    return results
