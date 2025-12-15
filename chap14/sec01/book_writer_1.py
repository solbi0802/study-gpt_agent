from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from typing_extensions import TypedDict
from typing import List

from utils import save_state, get_outline, save_outline 

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
    messages: List[AnyMessage | str]

# 목차를 작성하는 노드(agent)
def content_strategist(state: State):
    print("\n\n============ CONTENT STRATEGIST ============")

    # 시스템 프롬프트 정의
    content_strategist_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI팀의 콘텐츠 전략가(Content Strategist)로서,
        이전 대화 내용을 바탕으로 사용자의 요구사항을 분석하고, AI팀이 쓸 책의 세부 목차를 결정한다.

        지난 목차가 있다면 그 버전을 사용자의 요구에 맞게 수정하고, 없다면 새로운 목차를 제안한다.

        --------------------------------
        - 지난 목차: {outline}
        --------------------------------
        - 이전 대화 내용: {messages}
        """
    )

    # 시스템 프롬프트와 모델을 연결
    content_strategist_chain = content_strategist_system_prompt | llm | StrOutputParser()

    messages = state["messages"]        # 상태에서 메시지를 가져옴
    outline = get_outline(current_path) # 저장된 목차를 가져옴

    # 입력값 정의
    inputs = {
        "messages": messages,
        "outline": outline
    }

    # 목차 작성
    gathered = ''
    for chunk in content_strategist_chain.stream(inputs):
        gathered += chunk
        print(chunk, end='')

    print()

    save_outline(current_path, gathered) # 목차 저장

    # 메시지 추가    
    content_strategist_message = f"[Content Strategist] 목차 작성 완료: ${gathered}"
    print(content_strategist_message)
    messages.append(AIMessage(content_strategist_message))

    return {"messages": messages} # 메시지 업데이트


# 사용자와 대화할 노드(agent): communicator
def communicator(state: State):
    print("\n\n============ COMMUNICATOR ============")

    # 시스템 프롬프트 정의
    communicator_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI팀의 커뮤니케이터로서, 
        AI팀의 진행상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위한 대화를 나눈다. 

        사용자도 outline(목차)을 이미 보고 있으므로, 다시 출력할 필요는 없다.

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
graph_builder.add_node("content_strategist", content_strategist)

# Edges
graph_builder.add_edge(START, "content_strategist") # edge 변경 및 추가
graph_builder.add_edge("content_strategist", "communicator")  # edge변경 및 추가
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
