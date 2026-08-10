# Tool-Augmented RAG Travel Assistant

## Overview

The **Tool-Augmented RAG Travel Assistant** is an intelligent AI application built with **LangChain 1.3** and **LangGraph**. It combines Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), tool calling, conversational memory, and persistent conversation storage to answer travel-related questions intelligently.

The assistant can:

* Retrieve information from an internal knowledge base.
* Calculate travel logistics costs.
* Book flights and hotels using custom tools.
* Convert currencies.
* Remember previous messages within a conversation.
* Store completed conversations for long-term retrieval.

---

# Features

### Tool Calling

The assistant automatically selects and uses the appropriate tools based on user requests.

Available tools include:

* Flight Booking Tool
* Hotel Booking Tool
* Currency Conversion Tool
* Internal Knowledge (RAG) Tool

---

### Retrieval-Augmented Generation (RAG)

The assistant searches an internal knowledge base built from local documents including:

* TXT
* PDF
* DOCX

Documents are embedded using HuggingFace embeddings and indexed in ChromaDB for semantic search.

Example queries:

* What is the company travel policy?
* What expenses qualify as logistics expenses?
* What is the approved hotel policy?

---

### Conversational Memory

The project uses **LangGraph MemorySaver** to maintain context throughout a chat session.

Example:

**User**

> I am travelling from Lagos to Nairobi.

Later...

**User**

> How many nights?

The assistant remembers the previous conversation without asking the user again.

---

### Persistent Conversation History

Every completed interaction is stored inside a dedicated ChromaDB collection.

Previous conversations can later be retrieved using semantic similarity search, enabling long-term conversational memory.

---

## Project Architecture

```text
                    User
                      │
                      ▼
                 LangChain Agent
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   Flight Tool   Hotel Tool   Currency Tool
        │
        ▼
   RAG Knowledge Tool
        │
        ▼
    Chroma Knowledge Base
        │
        ▼
 Local Documents (PDF/TXT/DOCX)

Memory Layer
────────────
LangGraph MemorySaver
        │
        ▼
Conversation Context

Persistent Storage
──────────────────
Conversation ChromaDB
```

---

# Technologies Used

* Python 3.13+
* LangChain 1.3
* LangGraph
* OpenRouter API
* ChromaDB
* HuggingFace Embeddings
* Sentence Transformers
* python-dotenv

---

# Project Structure

```text
tool-augmented-rag-agent/
│
├── agent.py
├── chat.py
├── main.py
├── tools.py
├── rag.py
├── rag_tool.py
├── build_rag.py
├── test_rag.py
├── memory.py
├── conversation.py
├── data/
├── chroma_db/
├── requirements.txt
├── .env
└── README.md
```

---

# Installation

Clone the repository

```bash
git clone <repository-url>
cd tool-augmented-rag-agent
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file.

```text
OPENROUTER_API_KEY=your_openrouter_api_key
HF_API_KEY=your_huggingface_token
```

---

# Build the Knowledge Base

Place your documents inside the `data/` folder.

Supported format:

* TXT

Build the vector database

```bash
python build_rag.py
```

Expected output

```text
Indexed 3 document chunks.
```

---

# Running the Project

## Interactive Chat

```bash
python chat.py
```

Example

```text
You:
According to company policy, what is the total logistics cost for travelling from Lagos to Nairobi for three days?
```

---

## Single Question Mode

```bash
python main.py "What is the logistics cost from Lagos to Nairobi for three days?"
```

---

# Example Capabilities

### Flight Cost

```text
How much is a round trip flight from Lagos to Nairobi?
```

---

### Hotel Cost

```text
I will stay for three nights.
```

---

### Currency Conversion

```text
Convert the total cost to Nigerian Naira.
```

---

### RAG

```text
What is the company's travel policy?
```

---

### Combined Reasoning

```text
According to company policy, what is the total logistics cost for travelling from Lagos to Nairobi for three days?
```

The assistant retrieves company policy from the knowledge base, calls the flight tool, calls the hotel tool, and combines the results into a single response.

---

# Memory

The assistant supports two levels of memory.

## Session Memory

Managed by LangGraph MemorySaver.

Allows multi-turn conversations within the same chat session.

---

## Persistent Memory

Completed conversations are stored in ChromaDB.

This enables semantic retrieval of previous interactions even after restarting the application.

---

# Testing

Test individual components

```bash
python test_tools.py
```

```bash
python test_rag.py
```

```bash
python build_rag.py
```

Run the complete assistant

```bash
python chat.py
```

---

# Future Improvements

* Real-time flight APIs
* Real-time hotel booking APIs
* Live currency exchange APIs
* Web search integration
* User authentication
* Multi-user conversation memory
* Streaming responses
* Agent observability with LangSmith
* Deployment using Docker
* REST API with FastAPI

---

# Learning Objectives

This project demonstrates practical implementation of:

* LangChain 1.3 Agents
* LangGraph
* Retrieval-Augmented Generation (RAG)
* Tool Calling
* Vector Databases
* ChromaDB
* HuggingFace Embeddings
* Conversation Memory
* Persistent Memory
* OpenRouter Integration
* Multi-tool AI Systems

---

# Author

**Aniebiet Kingsley Inyang**

Electrical Engineer | Data Scientist | AI Engineer

MSc Data Science (Distinction)

Passionate about building intelligent AI systems using Large Language Models, Retrieval-Augmented Generation, Agentic AI, and MLOps.
