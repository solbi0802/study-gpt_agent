from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from typing_extensions import TypedDict
from typing import List

from utils import save_state 

from datetime import datetime
import os 

# 현재 폴더 경로 찾기
# 랭그래프 이미지로 저장 및 추후 작업 결과 파일 저장 경로로 활용
filename = os.path.basename(__file__) # 현재 파일명 반환
absolute_path = os.path.abspath(__file__) # 현재 파일의 절대 경로 반환
current_path = os.path.dirname(absolute_path) # 현재 .py 파일이 있는 폴더 경로 

# 모델 초기화
llm = ChatOpenAI(model="gpt-4o") 

# 상태 정의
class State(TypedDict):
    messages: List[AnyMessage | str] # 여러 종류의 메시지 타입(AIMessage, HumanMessage..etc) 을 하나로 통합하여 표현하기 위해 사용되는 타입 별칭

# 사용자와 대화할 노드(agent): communicator
def communicator(state: State):
    print("\n\n============ COMMUNICATOR ============")

    # 시스템 프롬프트 정의
    communicator_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI팀의 커뮤니케이터로서, 
        AI팀의 진행상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위한 대화를 나눈다. 

        messages: {messages}
        """
    )

    #② 시스템 프롬프트와 모델을 연결
    system_chain = communicator_system_prompt | llm

    # 상태에서 메시지를 가져옴
    messages = state["messages"]

    # 입력값 정의
    inputs = {"messages": messages}

    # 스트림되는 메시지를 출력하면서, gathered에 모으기
    gathered = None

    print('\nAI\t: ', end='')
    for chunk in system_chain.stream(inputs):
        print(chunk.content, end='')

        if gathered is None:
            gathered = chunk
        else:
            gathered += chunk

    messages.append(gathered)

    return {"messages": messages}

# 상태 그래프 정의
graph_builder = StateGraph(State)

# Nodes
graph_builder.add_node("communicator", communicator)

# Edges
graph_builder.add_edge(START, "communicator")
graph_builder.add_edge("communicator", END)

graph = graph_builder.compile()

graph.get_graph().draw_mermaid_png(output_file_path=absolute_path.replace('.py', '.png'))

# 상태 초기화
state = State(
    messages = [
        SystemMessage(
                f"""
            너희 AI들은 사용자의 요구에 맞는 책을 쓰는 작가팀이다.
            사용자가 사용하는 언어로 대화하라.

            현재시각은 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}이다.

            """
        )
    ],
)

while True:
    user_input = input("\nUser\t: ").strip()

    if user_input.lower() in ['exit', 'quit', 'q']:
        print("Goodbye!")
        break
    
    state["messages"].append(HumanMessage(user_input))
    state = graph.invoke(state)

    print('\n------------------------------------ MESSAGE COUNT\t', len(state["messages"]))

    save_state(current_path, state) # 현재 state 내용 저장
