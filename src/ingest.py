import os
import glob
import pickle
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from utils import ensure_dir, load_pdfs_as_markdown, chunk_text, save_pickle

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")
HR_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "HR_documents")
INDEX_PATH = os.path.join(ARTIFACTS_DIR, "faiss.index")
META_PATH = os.path.join(ARTIFACTS_DIR, "metadata.pkl")
EMB_PATH = os.path.join(ARTIFACTS_DIR, "embeddings.pkl")

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # fast and good trade-off


def collect_chunks() -> Dict:
    """Collect text chunks and metadata from PDFs in HR_documents.

    Loads PDFs as markdown text, splits them into overlapping chunks, and
    produces parallel lists of text chunks and their metadata.

    Returns:
        A dict with keys:
        - "chunks": List[str], the text chunks
        - "metadatas": List[dict], each with {"doc_id", "chunk_id"}
    """
    pdfs = sorted(glob.glob(os.path.join(HR_DIR, "*.pdf")))
    docs = load_pdfs_as_markdown(pdfs)

    chunks = []
    metadatas = []
    for doc_id, text in docs:
        parts = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for j, chunk in enumerate(parts):
            chunks.append(chunk)
            metadatas.append({"doc_id": doc_id, "chunk_id": j})
    return {"chunks": chunks, "metadatas": metadatas}


def build_index(chunks: List[str]):
    """Build a FAISS inner-product index and compute embeddings for chunks.

    Args:
        chunks: List of text chunks to embed and index.

    Returns:
        A tuple of (index, embeddings) where index is a FAISS IndexFlatIP and
        embeddings is a float32 numpy array of shape (N, D).
    """
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(chunks, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors = inner product
    index.add(embeddings)
    return index, embeddings


def main():
    """Run ingestion: collect chunks, build index, and persist artifacts.

    Saves FAISS index, metadata, and embeddings under the artifacts directory.
    Raises an error if no chunks are produced (e.g., missing PDFs).
    """
    ensure_dir(ARTIFACTS_DIR)

    data = collect_chunks()
    chunks = data["chunks"]
    metadatas = data["metadatas"]

    if not chunks:
        raise RuntimeError("No chunks produced. Ensure PDFs exist in HR_documents/")

    index, embeddings = build_index(chunks)

    faiss.write_index(index, INDEX_PATH)
    save_pickle({"chunks": chunks, "metadatas": metadatas}, META_PATH)
    save_pickle(embeddings, EMB_PATH)

    print(f"Ingestion complete. Saved index -> {INDEX_PATH}, metadata -> {META_PATH}, embeddings -> {EMB_PATH}")


if __name__ == "__main__":
    main()