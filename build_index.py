import os
import json
import pickle
import numpy as np
import faiss
from dotenv import load_dotenv
from google import genai

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=GEMINI_API_KEY)


def get_embeddings(texts):
    all_embeddings = []
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"Batch {i//batch_size + 1}...")
        response = client.models.embed_content(
            model="text-embedding-004",
            contents=batch,
        )
        for emb in response.embeddings:
            all_embeddings.append(emb.values)
    return np.array(all_embeddings, dtype=np.float32)


def build_index():
    with open("catalog.json") as f:
        catalog = json.load(f)

    texts = [
        f"{item['name']}. {item.get('description', '')} Type: {item.get('test_type', '')}."
        for item in catalog
    ]

    print(f"Building index for {len(texts)} items...")
    embeddings = get_embeddings(texts)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-9)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, "catalog.faiss")
    with open("catalog_meta.pkl", "wb") as f:
        pickle.dump(catalog, f)

    print(f"Done. {len(catalog)} items, dim={dim}")


if __name__ == "__main__":
    build_index()
