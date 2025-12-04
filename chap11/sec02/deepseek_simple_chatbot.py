from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

llm = ChatOllama(model="deepseek-r1:8b") 

messages = [
    SystemMessage("너는 사용자를 도와주는 상담사야."),
]

while True:
    user_input = input("사용자: ")

    if user_input == "exit":
        break
    
    messages.append( 
        HumanMessage(user_input)
    )  
    
    response = llm.stream(messages) # 스트림 출력
    # response로 스트림 출력되는 부분을 받아 터미널 창에 차례로 출력
    ai_message = None
    for chunk in response:
        print(chunk.content, end="")
        if ai_message is None:
            ai_message = chunk
        else:
            ai_message += chunk
    print('')
    # 책에서는 </think> 이후만 출력되게 split 처리했는데, 현재는 불필요함
    message_only = ai_message.content
    messages.append(AIMessage(message_only))

    # print("AI: " + response.content)
