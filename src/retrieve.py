import os
from typing import List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from utils import load_pickle

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
INDEX_PATH = os.path.join(ARTIFACTS_DIR, "faiss.index")
META_PATH = os.path.join(ARTIFACTS_DIR, "metadata.pkl")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_resources():
    """Load FAISS index, metadata, and encoder model for retrieval.

    Returns:
        A tuple (index, meta, model) where:
        - index: FAISS index for similarity search
        - meta: dict with keys "chunks" (List[str]) and "metadatas" (List[dict])
        - model: SentenceTransformer for encoding queries
    """
    index = faiss.read_index(INDEX_PATH)
    meta = load_pickle(META_PATH)
    model = SentenceTransformer(MODEL_NAME)
    return index, meta, model


def search(query: str, k: int = 5) -> List[Tuple[float, str, dict]]:
    """Retrieve top-k similar chunks for a query.

    Encodes the query, performs inner-product search over the FAISS index,
    and assembles results with their scores and metadata.

    Args:
        query: User query to search for.
        k: Number of results to return.

    Returns:
        List of tuples (score, chunk_text, metadata) sorted by relevance.
    """
    index, meta, model = load_resources()

    q_emb = model.encode([query], normalize_embeddings=True)
    q_emb = np.array(q_emb, dtype="float32")
    scores, idxs = index.search(q_emb, k)

    chunks = meta["chunks"]
    metadatas = meta["metadatas"]

    results = []
    for score, i in zip(scores[0], idxs[0]):
        if i == -1:
            continue
        results.append((float(score), chunks[i], metadatas[i]))
    return results


if __name__ == "__main__":
    print(search("What is the leave policy?"))