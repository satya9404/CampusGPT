import os

from pypdf import PdfReader


# ============================================================
# PDF LOADER
# NIT NAGALAND CAMPUSGPT
# ============================================================


def load_pdf(pdf_path):
    """
    Load one PDF and return its pages.
    """

    pages = []

    try:

        reader = PdfReader(pdf_path)

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            try:

                text = page.extract_text()

            except Exception:

                text = ""

            if text:

                text = text.strip()

            if not text:
                continue

            pages.append({

                "text": text,

                "source": os.path.basename(
                    pdf_path
                ),

                "page": page_number

            })

        print(
            f"Loaded: {os.path.basename(pdf_path)} "
            f"({len(pages)} pages)"
        )

    except Exception as e:

        print(
            f"ERROR loading {pdf_path}: {e}"
        )

    return pages


# ============================================================
# LOAD ALL PDF FILES
# ============================================================

def load_all_pdfs(data_folder="data"):
    """
    Automatically find and load every PDF
    inside the data folder and its subfolders.
    """

    all_pages = []

    if not os.path.exists(data_folder):

        print(
            f"ERROR: Folder '{data_folder}' does not exist."
        )

        return []


    # --------------------------------------------------------
    # Find every PDF recursively
    # --------------------------------------------------------

    pdf_files = []

    for root, directories, files in os.walk(
        data_folder
    ):

        for filename in files:

            if filename.lower().endswith(".pdf"):

                pdf_path = os.path.join(
                    root,
                    filename
                )

                pdf_files.append(
                    pdf_path
                )


    # --------------------------------------------------------
    # No PDFs
    # --------------------------------------------------------

    if not pdf_files:

        print(
            "No PDF files found."
        )

        return []


    print()
    print("=" * 60)
    print(
        f"Found {len(pdf_files)} PDF files"
    )
    print("=" * 60)
    print()


    # --------------------------------------------------------
    # Load every PDF
    # --------------------------------------------------------

    for pdf_path in pdf_files:

        pages = load_pdf(
            pdf_path
        )

        all_pages.extend(
            pages
        )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(
        f"Total pages loaded: {len(all_pages)}"
    )
    print("=" * 60)
    print()


    return all_pages


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

def load_pdfs(data_folder="data"):
    """
    Alias for older CampusGPT code.
    """

    return load_all_pdfs(
        data_folder
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    pages = load_all_pdfs("data")

    print()
    print(
        f"Total pages: {len(pages)}"
    )

    if pages:

        print()
        print("First document:")
        print("----------------")

        print(
            "Source:",
            pages[0]["source"]
        )

        print(
            "Page:",
            pages[0]["page"]
        )

        print(
            "Text:",
            pages[0]["text"][:500]
        )