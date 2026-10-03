from rag.generator import generate_answer


question = input(
    "Ask CampusGPT: "
)


result = generate_answer(
    question,
    top_k=5
)


print("\n")
print("=" * 60)
print("CAMPUSGPT ANSWER")
print("=" * 60)

print(result["answer"])


print("\n")
print("=" * 60)
print("SOURCES")
print("=" * 60)

for source in result["sources"]:

    print(
        f'{source["source"]} '
        f'(Page {source["page"]})'
    )