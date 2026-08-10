import sys
#import uuid

from agent import run_agent
from conversation import (
    save_conversation,
    print_conversation_history,
)

#THREAD_ID = f"travel-{uuid.uuid4()}"
THREAD_ID = "travel-assistant"


def main():

    if len(sys.argv) < 2:
        print("Usage:")
        print('python main.py "Your question"')
        return

    user_prompt = sys.argv[1]

    final_response = run_agent(
        user_prompt,
        thread_id=THREAD_ID,
    )

    save_conversation(
        user_prompt,
        final_response,
    )

    print_conversation_history()

    print("\nFinal Response")
    print("=" * 60)
    print(final_response)


if __name__ == "__main__":
    main()