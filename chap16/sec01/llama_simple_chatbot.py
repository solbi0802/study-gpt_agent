from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# 모델 초기화
# llm = ChatOpenAI(model="gpt-4o-mini")
llm = ChatOllama(model="deepseek-r1:8b") 

messages = [
    SystemMessage("You are a helpful assistant."),
]

while True:
    user_input = input("You\t: ").strip()

    if user_input in ["exit", "quit", "q"]:
        print("Goodbye!")
        break

    messages.append(HumanMessage(user_input))

    response = llm.invoke(messages)
    print("Bot\t: ", response.content)

    messages.append(AIMessage(response.content))
