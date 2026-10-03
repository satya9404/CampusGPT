from rag.pdf_loader import load_pdf


pdf_path = "data/aiml_notes.pdf"

pages = load_pdf(pdf_path)

print("\nNumber of pages:", len(pages))

for page in pages:
    print("\n-----------------------------")
    print("Source:", page["source"])
    print("Page:", page["page"])
    print("Text:")
    print(page["text"][:500])