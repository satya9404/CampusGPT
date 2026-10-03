import os
import pickle

import faiss
import numpy as np

from rag.embeddings import create_embeddings
from rag.embeddings import create_embedding


# ============================================================
# PATHS
# ============================================================

RAG_FOLDER = "rag"

INDEX_PATH = os.path.join(
    RAG_FOLDER,
    "faiss.index"
)

CHUNKS_PATH = os.path.join(
    RAG_FOLDER,
    "chunks.pkl"
)


# ============================================================
# CREATE VECTOR STORE
# ============================================================

def create_vector_store(chunks):

    if not chunks:

        raise ValueError(
            "No chunks were supplied."
        )


    print()
    print("=" * 60)
    print(
        f"Creating embeddings for {len(chunks)} chunks..."
    )
    print("=" * 60)
    print()


    # --------------------------------------------------------
    # Extract text
    # --------------------------------------------------------

    texts = []

    valid_chunks = []


    for chunk in chunks:

        text = chunk.get(
            "text",
            ""
        )


        if not text:
            continue


        text = text.strip()


        if not text:
            continue


        texts.append(text)

        valid_chunks.append(chunk)


    if not texts:

        raise RuntimeError(
            "No valid text was found in the chunks."
        )


    # --------------------------------------------------------
    # Create LOCAL embeddings
    # --------------------------------------------------------

    vectors = create_embeddings(
        texts
    )


    if not vectors:

        raise RuntimeError(
            "No embeddings were created."
        )


    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    embeddings = np.asarray(
        vectors,
        dtype="float32"
    )


    # --------------------------------------------------------
    # Embedding dimension
    # --------------------------------------------------------

    dimension = embeddings.shape[1]


    print()
    print(
        f"Embedding dimension: {dimension}"
    )


    # --------------------------------------------------------
    # FAISS
    #
    # Because embeddings are normalized,
    # inner product works as cosine similarity.
    # --------------------------------------------------------

    index = faiss.IndexFlatIP(
        dimension
    )


    index.add(
        embeddings
    )


    # --------------------------------------------------------
    # Create RAG folder
    # --------------------------------------------------------

    os.makedirs(
        RAG_FOLDER,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Save FAISS index
    # --------------------------------------------------------

    faiss.write_index(
        index,
        INDEX_PATH
    )


    # --------------------------------------------------------
    # Save chunks
    # --------------------------------------------------------

    with open(
        CHUNKS_PATH,
        "wb"
    ) as file:

        pickle.dump(
            valid_chunks,
            file
        )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("FAISS VECTOR STORE CREATED")
    print("=" * 60)

    print(
        f"Chunks  : {len(valid_chunks)}"
    )

    print(
        f"Vectors : {len(vectors)}"
    )

    print(
        f"Dimension : {dimension}"
    )

    print(
        f"Index   : {INDEX_PATH}"
    )

    print(
        f"Chunks  : {CHUNKS_PATH}"
    )

    print("=" * 60)
    print()


# ============================================================
# LOAD VECTOR STORE
# ============================================================

def load_vector_store():

    if not os.path.exists(
        INDEX_PATH
    ):

        raise FileNotFoundError(
            f"FAISS index not found: {INDEX_PATH}"
        )


    if not os.path.exists(
        CHUNKS_PATH
    ):

        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_PATH}"
        )


    index = faiss.read_index(
        INDEX_PATH
    )


    with open(
        CHUNKS_PATH,
        "rb"
    ) as file:

        chunks = pickle.load(
            file
        )


    return index, chunks


# ============================================================
# SEARCH VECTOR STORE
# ============================================================

def search_vector_store(
    query,
    top_k=5
):

    # --------------------------------------------------------
    # Load FAISS
    # --------------------------------------------------------

    index, chunks = load_vector_store()


    # --------------------------------------------------------
    # Embed query locally
    # --------------------------------------------------------

    query_vector = create_embedding(
        query
    )


    if query_vector is None:

        return []


    query_vector = np.asarray(
        [query_vector],
        dtype="float32"
    )


    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    k = min(
        top_k,
        len(chunks)
    )


    scores, indices = index.search(
        query_vector,
        k
    )


    # --------------------------------------------------------
    # Prepare results
    # --------------------------------------------------------

    results = []


    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue


        if idx >= len(chunks):
            continue


        chunk = chunks[idx].copy()


        chunk["score"] = float(
            score
        )


        results.append(
            chunk
        )


    return results