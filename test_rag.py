from rag.generator import generate_answer


question = input("Ask CampusGPT: ")


result = generate_answer(
    question
)


print("\n================================")
print("NIT NAGALAND CAMPUSGPT")
print("================================")

print("\nAnswer:")
print(result["answer"])


print("\nSources:")

for source in result["sources"]:

    print(
        f"- {source['source']} "
        f"(Page {source['page']})"
    )