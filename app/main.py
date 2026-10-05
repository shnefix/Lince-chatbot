from app.chatbot import Chatbot


def main():
    chatbot = Chatbot()

    print("\nChatbot ready!")
    print("Type 'exit' to quit.\n")

    while True:
        user_message = input("You: ")

        if user_message.lower() == "exit":
            break

        response = chatbot.chat(user_message)

        print(f"Bot: {response}\n")


if __name__ == "__main__":
    main()
