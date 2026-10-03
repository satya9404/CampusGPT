import os
import sys

from rag.pdf_loader import load_all_pdfs
from rag.chunker import chunk_pages
from rag.vector_store import create_vector_store


# ============================================================
# CAMPUSGPT KNOWLEDGE BASE BUILDER
# ============================================================

DATA_FOLDER = "data"


def main():

    print()
    print("=" * 60)
    print("        NIT NAGALAND CAMPUSGPT")
    print("        KNOWLEDGE BASE BUILDER")
    print("=" * 60)
    print()


    # --------------------------------------------------------
    # Check data folder
    # --------------------------------------------------------

    if not os.path.exists(DATA_FOLDER):

        print("ERROR: data folder does not exist.")

        print()
        print("Create this folder:")
        print("    data/")

        sys.exit(1)


    # --------------------------------------------------------
    # Find PDF files
    # --------------------------------------------------------

    pdf_files = []

    for root, dirs, files in os.walk(DATA_FOLDER):

        for file in files:

            if file.lower().endswith(".pdf"):

                pdf_files.append(
                    os.path.join(root, file)
                )


    # --------------------------------------------------------
    # Check PDFs
    # --------------------------------------------------------

    if not pdf_files:

        print("No PDF files found inside data/")

        sys.exit(1)


    print(
        f"Found {len(pdf_files)} PDF files."
    )

    print()


    for i, pdf in enumerate(pdf_files, start=1):

        print(
            f"{i:03d}. {pdf}"
        )


    print()
    print("=" * 60)
    print("Loading PDFs...")
    print("=" * 60)
    print()


    # --------------------------------------------------------
    # LOAD ALL PDF DOCUMENTS
    # --------------------------------------------------------

    pages = load_all_pdfs(
        DATA_FOLDER
    )


    print()
    print(
        f"Loaded {len(pages)} pages."
    )


    if not pages:

        print()
        print(
            "ERROR: No PDF pages were loaded."
        )

        sys.exit(1)


    # --------------------------------------------------------
    # CHUNK DOCUMENTS
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Creating text chunks...")
    print("=" * 60)
    print()


    chunks = chunk_pages(
        pages,
        chunk_size=500,
        overlap=100
    )


    print(
        f"Created {len(chunks)} chunks."
    )


    if not chunks:

        print()
        print(
            "ERROR: No chunks were created."
        )

        sys.exit(1)


    # --------------------------------------------------------
    # CREATE VECTOR DATABASE
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Creating FAISS vector database...")
    print("=" * 60)
    print()


    create_vector_store(
        chunks
    )


    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("       INDEXING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print()
    print(
        f"PDF files : {len(pdf_files)}"
    )

    print(
        f"Pages     : {len(pages)}"
    )

    print(
        f"Chunks    : {len(chunks)}"
    )

    print()

    print(
        "CampusGPT knowledge base is ready."
    )

    print()


if __name__ == "__main__":

    main()