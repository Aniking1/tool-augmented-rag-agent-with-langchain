from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

conversation_store = Chroma(
    collection_name="conversation_history",
    persist_directory="conversation_db",
    embedding_function=embeddings,
)


def save_conversation(user_message: str, ai_message: str):
    """
    Save one completed conversation turn.
    """

    conversation_store.add_documents(
        [
            Document(
                page_content=f"""
User:
{user_message}

Assistant:
{ai_message}
"""
            )
        ]
    )


def retrieve_conversations(query: str, k: int = 3) -> str:
    """
    Retrieve similar previous conversations.
    """

    docs = conversation_store.similarity_search(
        query,
        k=k,
    )

    if not docs:
        return "No previous conversations."

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


def print_conversation_history():
    """
    Print every stored conversation.
    """

    #docs = conversation_store.get()
    docs = conversation_store.get(include=["documents"])

    print("\nConversation History")
    print("=" * 60)

    if not docs["documents"]:
        print("No conversations stored yet.")
        return

    for i, text in enumerate(docs["documents"], start=1):
        print(f"\nConversation {i}")
        print("-" * 40)
        print(text)