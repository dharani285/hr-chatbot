import os
import pickle
from typing import List, Tuple

import numpy as np
from markdown_it import MarkdownIt


def ensure_dir(path: str) -> None:
    """Create a directory if it does not already exist.

    Args:
        path: Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


def load_pdfs_as_markdown(pdf_paths: List[str]) -> List[Tuple[str, str]]:
    """Load PDFs and convert each page to normalized markdown text.

    Each PDF is opened and processed page-by-page. The plain text extracted
    from each page is run through a Markdown parser to normalize structure.

    Args:
        pdf_paths: List of absolute or relative paths to PDF files.

    Returns:
        A list of (doc_id, text) tuples where doc_id encodes filename and page
        number as "<filename>::page_<N>" and text is markdown-formatted.
    """
    import fitz  # PyMuPDF

    md = MarkdownIt()
    docs = []
    for pdf_path in pdf_paths:
        doc = fitz.open(pdf_path)
        fname = os.path.basename(pdf_path)
        for page_idx in range(len(doc)):
            page = doc.load_page(page_idx)
            text = page.get_text("text")
            # Convert plain text through Markdown parser to normalize
            normalized = md.render(text)
            doc_id = f"{fname}::page_{page_idx+1}"
            docs.append((doc_id, normalized))
    return docs


def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks based on word count.

    This approximates token-based chunking by splitting on whitespace.

    Args:
        text: The input text to split.
        chunk_size: Target number of words per chunk.
        chunk_overlap: Number of words to overlap between consecutive chunks.

    Returns:
        A list of chunk strings.
    """
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i : i + chunk_size]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        i += max(1, chunk_size - chunk_overlap)
    return chunks


def save_pickle(obj, path: str) -> None:
    """Serialize a Python object to a file using pickle.

    Args:
        obj: The object to serialize.
        path: Destination file path.
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    """Load a Python object from a pickle file.

    Args:
        path: Pickle file path to read from.

    Returns:
        The deserialized Python object.
    """
    with open(path, "rb") as f:
        return pickle.load(f)