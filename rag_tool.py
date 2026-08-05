from langchain.tools import tool

from rag import get_retriever
from conversation import retrieve_conversations

retriever = get_retriever()


@tool
def query_internal_knowledge(query: str) -> str:
    """
    Search both the knowledge base and previous conversations.
    """

    docs = retriever.invoke(query)

    knowledge = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    previous = retrieve_conversations(query)

    return f"""
Knowledge Base
==============

{knowledge}

Previous Conversations
======================

{previous}
"""