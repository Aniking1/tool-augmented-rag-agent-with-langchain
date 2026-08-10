from langchain.tools import tool

from rag import build_vector_store, get_retriever
from conversation import retrieve_conversations


# -------------------------------------------------------
# Retriever
# -------------------------------------------------------

retriever = None


# -------------------------------------------------------
# Initialize RAG
# -------------------------------------------------------

def initialize_rag():
    """
    Ensure the knowledge base exists and initialize
    the retriever before the agent is invoked.
    """

    global retriever

    # Build only if the database does not already exist.
    build_vector_store()

    # Connect to the existing vector store.
    retriever = get_retriever()

    print("RAG knowledge base initialized.")


# -------------------------------------------------------
# Internal Knowledge Tool
# -------------------------------------------------------

@tool
def query_internal_knowledge(query: str) -> str:
    """
    Search the internal knowledge base and previous
    conversations using vector similarity retrieval.
    """

    if retriever is None:
        return (
            "The internal knowledge base has not been "
            "initialized."
        )

    # ---------------------------------------------------
    # Retrieve knowledge-base documents
    # ---------------------------------------------------

    docs = retriever.invoke(query)

    knowledge = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    if not knowledge:
        knowledge = "No relevant internal knowledge found."

    # ---------------------------------------------------
    # Retrieve previous conversations
    # ---------------------------------------------------

    previous = retrieve_conversations(query)

    # ---------------------------------------------------
    # Return hybrid retrieval result
    # ---------------------------------------------------

    return f"""
Knowledge Base
==============

{knowledge}

Previous Conversations
======================

{previous}
"""