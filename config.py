import os

from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "nvidia/nemotron-3-nano-30b-a3b:free"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

HF_API_KEY = os.getenv("HF_API_KEY")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

KNOWLEDGE_DB = "chroma_db"

CONVERSATION_DB = "conversation_db"

THREAD_ID = "travel-assistant"