from rag.embeddings import create_embedding


text = "Machine learning is a branch of artificial intelligence."

vector = create_embedding(text)

print("Embedding created successfully!")
print("Vector type:", type(vector))
print("Vector length:", len(vector))
print("First 10 values:")
print(vector[:10])