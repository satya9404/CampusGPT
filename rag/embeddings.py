import os

from sentence_transformers import SentenceTransformer


# ============================================================
# CAMPUSGPT LOCAL EMBEDDINGS
# ============================================================

MODEL_NAME = os.getenv(
    "LOCAL_EMBEDDING_MODEL",
    "all-MiniLM-L6-v2"
)


print()
print("=" * 60)
print("Loading local embedding model...")
print("=" * 60)
print(f"Model: {MODEL_NAME}")
print()


# Load model once
model = SentenceTransformer(
    MODEL_NAME
)


print("Embedding model loaded successfully.")
print()


# ============================================================
# CREATE ONE EMBEDDING
# ============================================================

def create_embedding(text):

    if not text or not text.strip():
        return None

    vector = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return vector.tolist()


# ============================================================
# CREATE MULTIPLE EMBEDDINGS
# ============================================================

def create_embeddings(texts):

    valid_texts = []

    for text in texts:

        if text and text.strip():

            valid_texts.append(
                text.strip()
            )


    if not valid_texts:
        return []


    print(
        f"Creating local embeddings for "
        f"{len(valid_texts)} chunks..."
    )


    vectors = model.encode(
        valid_texts,

        batch_size=32,

        show_progress_bar=True,

        convert_to_numpy=True,

        normalize_embeddings=True
    )


    return vectors.tolist()