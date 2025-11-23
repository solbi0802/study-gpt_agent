from openai import OpenAI  # 주석처리
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")  # 환경 변수에서 API 키 가져오기
client = OpenAI(api_key=api_key)  # 오픈AI 클라이언트의 인스턴스 생성

llm = ChatOpenAI(model="gpt-4o")  # ChatOpenAI 클래스의 인스턴스 생성


# def get_ai_response(messages):
#     response = client.chat.completions.create(
#         model="gpt-4o",  # 응답 생성에 사용할 모델 지정
#         temperature=0.9,  # 응답 생성에 사용할 temperature 설정
#         messages=messages,  # 대화 기록을 입력으로 전달
#     )
#     return response.choices[0].message.content  # 생성된 응답의 내용 반환

messages = [
    SystemMessage("너는 사용자를 도와주는 상담사야."),  # 초기 시스템 메시지
]

while True:
    user_input = input("사용자: ")  # 사용자 입력 받기

    if user_input == "exit":  # 사용자가 대화를 종료하려는지 확인인
        break
    
    messages.append(
        HumanMessage(user_input)
    )  # 사용자 메시지를 대화 기록에 추가 
    
    ai_response = llm.invoke(messages)  # 대화 기록을 기반으로 AI 응답 가져오기
    messages.append(
        ai_response
    )  # AI 응답 대화 기록에 추가하기

    print("AI: " + ai_response.content)  # AI 응답 출력
