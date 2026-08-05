from rag import get_retriever

retriever = get_retriever()

docs = retriever.invoke("conference reimbursement")

print("\nRetrieved Documents")
print("=" * 50)

for i, doc in enumerate(docs, 1):
    print(f"\nDocument {i}")
    print("-" * 40)
    print(doc.page_content)