from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")

from typing import Annotated # annotated는 타입 힌트를 사용할 때 사용하는 함수
from typing_extensions import TypedDict # TypedDict는 딕셔너리 타입을 정의할 때 사용하는 함수

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    """
    State 클래스는 TypedDict를 상속받습니다.

    속성:
        messages (Annotated[list[str], add_messages]): 메시지들은 "list" 타입을 가집니다.
       'add_messages' 함수는 이 상태 키가 어떻게 업데이트되어야 하는지를 정의합니다.
        (이 경우, 메시지를 덮어쓰는 대신 리스트에 추가합니다)
    """
    messages: Annotated[list[str], add_messages]

# StateGraph 클래스를 사용하여 State 타입의 그래프를 생성합니다.
graph_builder = StateGraph(State)


def generate(state: State):
    """
    주어진 상태를 기반으로 챗봇의 응답 메시지를 생성합니다.

    매개변수:
    state (State): 현재 대화 상태를 나타내는 객체로, 이전 메시지들이 포함되어 있습니다.
		
    반환값:
    dict: 모델이 생성한 응답 메시지를 포함하는 딕셔너리. 
          형식은 {"messages": [응답 메시지]}입니다.
    """ 
    return {"messages": [model.invoke(state["messages"])]}

graph_builder.add_node("generate", generate)

graph_builder.add_edge(START, "generate")
graph_builder.add_edge("generate", END)    

from langgraph.checkpoint.memory import MemorySaver
memory = MemorySaver()

config = {"configurable": {"thread_id": "abcd"}}

graph = graph_builder.compile(checkpointer=memory)

#------------ 여기서부터 달라진 코드가 있습니다.   
from langchain.schema import HumanMessage

while True:
    user_input = input("You\t:")
    
    if user_input in ["exit", "quit", "q"]:
        break
	#②
    for event in graph.stream({
        "messages": [HumanMessage(user_input)]}, 
        config,
        stream_mode="values"
    ):
        event["messages"][-1].pretty_print()

    print(f'\n현재 메시지 갯수: {len(event["messages"])}\n-------------------\n')
