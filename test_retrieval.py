from rag.retriever import search_documents


query = input("Ask CampusGPT: ")

results = search_documents(
    query,
    top_k=5
)

print("\n================================")
print("RETRIEVED DOCUMENTS")
print("================================")

for i, result in enumerate(results, start=1):

    print(f"\nResult {i}")
    print("----------------------------")

    print("Source:", result["source"])
    print("Page:", result["page"])
    print("Distance:", result["distance"])

    print("\nText:")
    print(result["text"][:500])