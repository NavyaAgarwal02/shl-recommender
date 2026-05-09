import json, pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def build_index():
    with open("catalog.json") as f:
        catalog = json.load(f)
    
    model = SentenceTransformer("all-MiniLM-L6-v2")  # free, fast, good enough
    
    # Build rich text representation for each assessment
    texts = []
    for item in catalog:
        text = f"{item['name']}. {item.get('description', '')} Type: {item.get('test_type','')}."
        texts.append(text)
    
    print("Encoding catalog...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product = cosine similarity (with normalized vecs)
    index.add(embeddings.astype(np.float32))
    
    faiss.write_index(index, "catalog.faiss")
    with open("catalog_meta.pkl", "wb") as f:
        pickle.dump(catalog, f)
    
    print(f"Index built: {len(catalog)} items, dim={dim}")


if __name__ == "__main__":
    build_index()