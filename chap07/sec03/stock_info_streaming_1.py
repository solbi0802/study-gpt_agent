from gpt_functions import get_current_time, tools, get_yf_stock_info, get_yf_stock_history, get_yf_stock_recommendations
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import streamlit as st

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")  # 환경 변수에서 API 키 가져오기

client = OpenAI(api_key=api_key)  # 오픈AI 클라이언트의 인스턴스 생성

def get_ai_response(messages, tools=None, stream=True):
    response = client.chat.completions.create(
        model="gpt-4o",  # 응답 생성에 사용할 모델을 지정합니다.
        stream=stream, # (1) 스트리밍 출력을 위해 설정
        messages=messages,  # 대화 기록을 입력으로 전달합니다.
        tools=tools,  # 사용 가능한 도구 목록을 전달합니다.
    )

    if stream: 
        for chunk in response:
            yield chunk  # 생성된 응답의 내용을 yield로 순차적으로 반환합니다.
    else:
        return response  # 생성된 응답의 내용을 반환합니다.


st.title("💬 Chatbot")   

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "너는 사용자를 도와주는 상담사야."},  # 초기 시스템 메시지
    ] 

for msg in st.session_state.messages:
    if msg["role"] == "assistant" or msg["role"] == "user": # assistant 혹은 user 메시지인 경우만
        st.chat_message(msg["role"]).write(msg["content"])


if user_input := st.chat_input():    # 사용자 입력 받기
    st.session_state.messages.append({"role": "user", "content": user_input})  # 사용자 메시지를 대화 기록에 추가
    st.chat_message("user").write(user_input)  # 사용자 메시지를 브라우저에서도 출력
    
    ai_response = get_ai_response(st.session_state.messages, tools=tools)
    # print(ai_message) 

    content = ''
    tool_calls = None # # tool_calls 초기화
    
    with st.chat_message("assistant").empty(): # 스트림릿 챗 메시지 초기화
        for chunk in ai_response:
            content_chunk = chunk.choices[0].delta.content # 청크 속 content 추출
            if content_chunk: # 만약 content_chunk가 있다면, 
                print(content_chunk, end="")	 # 터미널에 줄바꿈 없이 이어서 출력
                content += content_chunk # content에 덧붙이기
                st.markdown(content) # 스트림릿 챗 메시지에 마크다운으로 출력
        
    print('\n===========')
    print(content)

    # ai_message = ai_response.choices[0].message
    # tool_calls = ai_message.tool_calls  # AI 응답에 포함된 tool_calls를 가져옵니다.
    if tool_calls:  # tool_calls가 있는 경우
        for tool_call in tool_calls:
            tool_name = tool_call.function.name # 실행해야한다고 판단한 함수명 받기
            tool_call_id = tool_call.id         # tool_call 아이디 받기    
            arguments = json.loads(tool_call.function.arguments) # 문자열을 딕셔너리로 변환    
            
            if tool_name == "get_current_time":  
                func_result = get_current_time(timezone=arguments['timezone'])
            elif tool_name == "get_yf_stock_info":
                func_result = get_yf_stock_info(ticker=arguments['ticker'])
            elif tool_name == "get_yf_stock_history":  # get_yf_stock_history 함수 호출
                func_result = get_yf_stock_history(
                    ticker=arguments['ticker'], 
                    period=arguments['period']
                )
            elif tool_name == "get_yf_stock_recommendations":  # get_yf_stock_recommendations 함수 호출
                func_result = get_yf_stock_recommendations(
                    ticker=arguments['ticker']
                )

            st.session_state.messages.append({
                "role": "function",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": func_result,
            })


        st.session_state.messages.append({"role": "system", "content": "이제 주어진 결과를 바탕으로 답변할 차례다."}) 
        ai_response = get_ai_response(st.session_state.messages, tools=tools) # 다시 GPT 응답 받기
        ai_message = ai_response.choices[0].message

    st.session_state.messages.append({
        "role": "assistant",
        "content": content # 원래는 ai_message.content 였음
    })  # ③ AI 응답을 대화 기록에 추가합니다.

    print("AI\t: " + content)  # AI 응답 출력
    # st.chat_message("assistant").write(content)  # 위에서 스트림 방식 출력하므로 불필요