from agent import run_agent

THREAD_ID = "travel-assistant"

print("=" * 60)
print("Travel Assistant Chat")
print("Type 'exit' to quit.")
print("=" * 60)

while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        print("\nGoodbye!")
        break

    response = run_agent(
        user_input,
        thread_id=THREAD_ID,
    )

    print(f"\nAssistant: {response}")