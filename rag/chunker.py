def chunk_text(text, chunk_size=150, overlap=30):
    """
    Split text into overlapping word-based chunks.
    """

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def chunk_pages(pages, chunk_size=150, overlap=30):
    """
    Convert page-level PDF data into chunks
    while preserving source and page information.
    """

    all_chunks = []

    for page in pages:

        chunks = chunk_text(
            page["text"],
            chunk_size,
            overlap
        )

        for chunk in chunks:

            all_chunks.append({
                "text": chunk,
                "source": page["source"],
                "page": page["page"]
            })

    return all_chunks