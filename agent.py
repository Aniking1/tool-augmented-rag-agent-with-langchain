import os

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from memory import memory

from tools import (
    get_flight_booking,
    get_hotel_booking,
    convert_currency,
)

from rag_tool import (
    query_internal_knowledge,
    initialize_rag,
)


# -------------------------------------------------------
# Load environment variables
# -------------------------------------------------------

load_dotenv()


# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

MODEL_NAME = "nvidia/nemotron-3-nano-30b-a3b:free"


# -------------------------------------------------------
# OpenRouter LLM
# -------------------------------------------------------

llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


# -------------------------------------------------------
# Register tools
# -------------------------------------------------------

tools = [
    get_flight_booking,
    get_hotel_booking,
    convert_currency,
    query_internal_knowledge,
]


# -------------------------------------------------------
# System Prompt
# -------------------------------------------------------

SYSTEM_PROMPT = """
You are an intelligent AI Travel Assistant.

You have access to four tools:

1. Flight Booking
2. Hotel Booking
3. Currency Conversion
4. Internal Knowledge Search (RAG)

Guidelines:

• Use the Flight tool whenever flight information or
  airfare is required.

• Use the Hotel tool whenever accommodation pricing
  is required.

• If the user asks for the total logistics cost,
  calculate:

    Flight Cost + Hotel Cost

  using the available tools.

• Only use the Currency Conversion tool if the user
  explicitly requests another currency.

• Use the Internal Knowledge tool whenever company
  policies, travel policies, conference guidelines,
  reimbursements, or previous knowledge are requested.

• Answer clearly and concisely.

• If enough information is available, do not ask
  unnecessary follow-up questions.

• Prefer tool usage over guessing.

• When internal policy information is requested,
  retrieve it using the Internal Knowledge Search tool
  rather than relying on assumptions.
"""


# -------------------------------------------------------
# Build LangGraph Agent
# -------------------------------------------------------

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=memory,
    debug=False,
)


# -------------------------------------------------------
# Run Agent
# -------------------------------------------------------

def run_agent(
    prompt: str,
    thread_id: str = "travel-session",
) -> str:
    """
    Execute the travel agent.

    The RAG knowledge base is initialized before the
    agent is invoked.
    """

    # ---------------------------------------------------
    # IMPORTANT:
    # Ensure RAG exists BEFORE calling the agent.
    #
    # initialize_rag() does NOT rebuild an existing DB.
    # ---------------------------------------------------

    initialize_rag()

    # ---------------------------------------------------
    # Invoke agent
    # ---------------------------------------------------

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": thread_id,
            }
        },
    )

    return result["messages"][-1].content


# -------------------------------------------------------
# Optional streaming
# -------------------------------------------------------

def stream_agent(
    prompt: str,
    thread_id: str = "travel-session",
):
    """
    Stream intermediate LangGraph events.
    """

    initialize_rag()

    for event in agent.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        },
        config={
            "configurable": {
                "thread_id": thread_id,
            }
        },
        stream_mode="updates",
    ):
        print(event)