import os

from rag.pdf_loader import load_pdf
from rag.chunker import chunk_pages


DATA_FOLDER = "data"

all_pages = []

# Find every PDF in data folder
for filename in os.listdir(DATA_FOLDER):

    if filename.lower().endswith(".pdf"):

        file_path = os.path.join(DATA_FOLDER, filename)

        print(f"\nLoading: {filename}")

        pages = load_pdf(file_path)

        all_pages.extend(pages)


print("\n==============================")
print("TOTAL PAGES:", len(all_pages))
print("==============================")


# Chunk all PDFs
chunks = chunk_pages(
    all_pages,
    chunk_size=150,
    overlap=30
)


print("TOTAL CHUNKS:", len(chunks))


# Show first 5 chunks
for i, chunk in enumerate(chunks[:5], start=1):

    print("\n------------------------------")
    print("Chunk:", i)
    print("Source:", chunk["source"])
    print("Page:", chunk["page"])
    print("Text:")
    print(chunk["text"][:300])