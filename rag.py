from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import EMBEDDING_MODEL, KNOWLEDGE_DB


# -------------------------------------------------------
# Paths
# -------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / KNOWLEDGE_DB


# -------------------------------------------------------
# Embeddings
# -------------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)


# -------------------------------------------------------
# Check whether the vector store already exists
# -------------------------------------------------------

def vector_store_exists() -> bool:
    """
    Check whether the Chroma knowledge base already exists.
    """

    return CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir())


# -------------------------------------------------------
# Build the vector store
# -------------------------------------------------------

def build_vector_store():
    """
    Build the Chroma knowledge base from files in data/.

    The database is created only if it does not already exist.
    """

    if vector_store_exists():
        print("Existing Chroma knowledge base found.")
        print("Skipping ingestion.")
        return

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Data directory not found: {DATA_DIR}"
        )

    loader = DirectoryLoader(
        str(DATA_DIR),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )

    documents = loader.load()

    if not documents:
        raise ValueError(
            f"No .txt documents found in {DATA_DIR}"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    chunks = splitter.split_documents(documents)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="knowledge_base",
    )

    print(
        f"Indexed {len(chunks)} document chunks "
        f"into the Chroma knowledge base."
    )


# -------------------------------------------------------
# Get retriever
# -------------------------------------------------------

def get_retriever():
    """
    Return a retriever connected to the existing Chroma
    knowledge base.
    """

    if not vector_store_exists():
        raise RuntimeError(
            "Knowledge base does not exist. "
            "Run build_vector_store() first."
        )

    vector_store = Chroma(
        collection_name="knowledge_base",
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    return vector_store.as_retriever(
        search_kwargs={"k": 4}
    )