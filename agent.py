import os

from dotenv import load_dotenv
from memory import memory
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from tools import (
    get_flight_booking,
    get_hotel_booking,
    convert_currency,
)

from rag_tool import query_internal_knowledge

load_dotenv()

MODEL_NAME = "nvidia/nemotron-3-nano-30b-a3b:free"

llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

tools = [
    get_flight_booking,
    get_hotel_booking,
    convert_currency,
    query_internal_knowledge,
]

system_prompt = """
You are an AI travel assistant.

You MUST use the available tools whenever the user asks about:

- flights
- hotels
- travel cost
- logistics cost

Never answer from your own knowledge if a tool exists.

Always call the appropriate tool first before responding.
"""

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt,
    checkpointer=memory,
)

def run_agent(prompt: str, thread_id: str = "travel-assistant") -> str:
    """
    Run the travel agent with persistent conversation memory.
    """

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
                "thread_id": thread_id
            }
        }
    )

    return result["messages"][-1].content