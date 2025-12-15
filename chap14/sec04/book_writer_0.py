from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from typing_extensions import TypedDict
from typing import List

from utils import save_state, get_outline, save_outline 
from models import Task
from tools import retrieve, web_search, add_web_pages_json_to_chroma 
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
    task_history: List[Task]    
    references: dict

def supervisor(state: State): # supervisor 에이전트 추가
    print("\n\n============ SUPERVISOR ============")

    # 시스템 프롬프트 정의
    supervisor_system_prompt = PromptTemplate.from_template(
        """
        너는 AI 팀의 supervisor로서 AI 팀의 작업을 관리하고 지도한다.
        사용자가 원하는 책을 써야 한다는 최종 목표를 염두에 두고, 
        사용자의 요구를 달성하기 위해 현재 해야할 일이 무엇인지 결정한다.

        supervisor가 활용할 수 있는 agent는 다음과 같다.     
        - content_strategist: 사용자의 요구사항이 명확해졌을 때 사용한다. AI 팀의 콘텐츠 전략을 결정하고, 전체 책의 목차(outline)를 작성한다. 
        - communicator: AI 팀에서 해야 할 일을 스스로 판단할 수 없을 때 사용한다. 사용자에게 진행상황을 사용자에게 보고하고, 다음 지시를 물어본다. 
        - web_search_agent: 웹 검색을 통해 목차(outline) 작성에 필요한 정보를 확보한다.
        - vector_search_agent: 벡터 DB 검색을 통해 목차(outline) 작성에 필요한 정보를 확보한다.

        아래 내용을 고려하여, 현재 해야할 일이 무엇인지, 사용할 수 있는 agent를 단답으로 말하라.

        ------------------------------------------
        previous_outline: {outline}
        ------------------------------------------
        messages:
        {messages}
        """
    )

    # 체인 연결
    supervisor_chain = supervisor_system_prompt | llm. with_structured_output(Task)    

    # 메시지 가져오기
    messages = state.get("messages", [])		#⑤

    # inputs 설정
    inputs = {
        "messages": messages,
        "outline": get_outline(current_path)
    }

    # task 문자열로 생성
    task = supervisor_chain.invoke(inputs) 	#⑦
    task_history = state.get("task_history", [])    # 작업 이력 가져오기
    task_history.append(task)                    	# 작업 이력에 추가

   
    # 메시지 추가
    supervisor_message = AIMessage(f"[Supervisor] {task}")
    messages.append(supervisor_message)
    print(supervisor_message.content)

    # state 업데이트
    return {
        "messages": messages, 
        "task_history": task_history
    }

# supervisor's route
def supervisor_router(state: State):
    task = state['task_history'][-1]
    return task.agent			

def vector_search_agent(state: State):
    print("\n\n============ VECTOR SEARCH AGENT ============")
    
    tasks = state.get("task_history", [])
    task = tasks[-1]
    if task.agent != "vector_search_agent":
        raise ValueError(f"Vector Search Agent가 아닌 agent가 Vector Search Agent를 시도하고 있습니다.\n {task}")

    vector_search_system_prompt = PromptTemplate.from_template(
        """
        너는 다른 AI Agent 들이 수행한 작업을 바탕으로, 
        목차(outline) 작성에 필요한 정보를 벡터 검색을 통해 찾아내는 Agent이다.

        현재 목차(outline)을 작성하는데 필요한 정보를 확보하기 위해, 
        다음 내용을 활용해 적절한 벡터 검색을 수행하라. 

        - 검색 목적: {mission}
        --------------------------------
        - 과거 검색 내용: {references}
        --------------------------------
        - 이전 대화 내용: {messages}
        --------------------------------
        - 목차(outline): {outline}
        """
    )

    # inputs 설정
    mission = task.description
    references = state.get("references", {"queries": [], "docs": []})
    messages = state["messages"]
    outline = get_outline(current_path)

    inputs = {
        "mission": mission,
        "references": references,
        "messages": messages,
        "outline": outline
    }

    # LLM과 벡터 검색 모델 연결
    llm_with_retriever = llm.bind_tools([retrieve]) 
    vector_search_chain = vector_search_system_prompt | llm_with_retriever

    # LLM과 벡터 검색 모델 연결
    search_plans = vector_search_chain.invoke(inputs)
    # 검색할 내용 출력
    for tool_call in search_plans.tool_calls:
        print('-----------------------------------', tool_call)
        args = tool_call["args"]
       
        query = args["query"] 
        retrieved_docs = retrieve(args)
		#① (1) 결과 담아 두기
        references["queries"].append(query) 
        references["docs"] += retrieved_docs
    
    unique_docs = []
    unique_page_contents = set()

    for doc in references["docs"]:
        if doc.page_content not in unique_page_contents:
            unique_docs.append(doc)
            unique_page_contents.add(doc.page_content)
    references["docs"] = unique_docs

    # 검색 결과 출력 – 쿼리 출력
    print('Queries:--------------------------')
    queries = references["queries"]
    for query in queries:
        print(query)
    
    # 검색 결과 출력 – 문서 청크 출력
    print('References:--------------------------')
    for doc in references["docs"]:
        print(doc.page_content[:100])
        print('--------------------------')

    # task 완료
    tasks[-1].done = True
    tasks[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 새로운 task 추가
    new_task = Task(
        agent="communicator",
        done=False,
        description="AI팀의 진행상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위한 대화를 나눈다",
        done_at=""
    )
    tasks.append(new_task)

    # vector search agent의 작업후기를 메시지로 생성
    msg_str = f"[VECTOR SEARCH AGENT] 다음 질문에 대한 검색 완료: {queries}"
    message = AIMessage(msg_str)
    print(msg_str)

    messages.append(message)
    # state 업데이트
    return {
        "messages": messages,
        "task_history": tasks,
        "references": references
    }


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
    content_strategist_message = f"[Content Strategist] 목차 작성 완료"
    print(content_strategist_message)
    messages.append(AIMessage(content_strategist_message))

    task_history = state.get("task_history", []) # task_history 가져오기
    # 최근 task 작업완료(done) 처리하기
    if task_history[-1].agent != "content_strategist": 
        raise ValueError(f"Content Strategist가 아닌 agent가 목차 작성을 시도하고 있습니다.\n {task_history[-1]}")
    
    task_history[-1].done = True
    task_history[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 다음 작업이 communicator로 사용자와 대화하는 것이므로 새 작업 추가 
    new_task = Task(
        agent="communicator",
        done=False,
        description="AI팀의 진행상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위한 대화를 나눈다",
        done_at=""
    )
    task_history.append(new_task)

    print(new_task)

    # 현재 state를 업데이트한다. 
    return {
        "messages": messages,
        "task_history": task_history
    }


def web_search_agent(state: State): #① (0)
    print("\n\n============ WEB SEARCH AGENT ============")

    # 작업 리스트 가져와서 web search agent 가 할 일인지 확인하기
    tasks = state.get("task_history", [])
    task = tasks[-1]

    if task.agent != "web_search_agent":
        raise ValueError(f"Web Search Agent가 아닌 agent가 Web Search Agent를 시도하고 있습니다.\n {task}")
    
    #③ 시스템 프롬프트 정의
    web_search_system_prompt = PromptTemplate.from_template(
        """
        너는 다른 AI Agent 들이 수행한 작업을 바탕으로, 
        목차(outline) 작성에 필요한 정보를 웹 검색을 통해 찾아내는 Web Search Agent이다.

        현재 부족한 정보를 검색하고, 복합적인 질문은 나눠서 검색하라.

        - 검색 목적: {mission}
        --------------------------------
        - 과거 검색 내용: {references}
        --------------------------------
        - 이전 대화 내용: {messages}
        --------------------------------
        - 목차(outline): {outline}
        --------------------------------
        - 현재 시각 : {current_time}
        """
    )
    
    #④ 기존 대화 내용 가져오기
    messages = state.get("messages", [])

    #⑤ 인풋 자료 준비하기
    inputs = {
        "mission": task.description,
        "references": state.get("references", {"queries": [], "docs": []}),
        "messages": messages,
        "outline": get_outline(current_path),
        "current_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    #⑥ LLM과 웹 검색 모델 연결
    llm_with_web_search = llm.bind_tools([web_search])

    #⑦ 시스템 프롬프트와 모델을 연결
    web_search_chain = web_search_system_prompt | llm_with_web_search

    #⑧ 웹 검색 tool_calls 가져오기
    search_plans = web_search_chain.invoke(inputs)

    #⑨ 어떤 내용을 검색했는지 담아두기
    queries = []

    #⑩ 검색 계획(tool_calls)에 따라 검색하기
    for tool_call in search_plans.tool_calls:
        print('-------- web search --------', tool_call)
        args = tool_call["args"]
        
        queries.append(args["query"])

        # (10)  검색 결과를 chroma에 추가
        _, json_path = web_search.invoke(args)
        print('json_path:', json_path)

        # (10)  JSON 파일을 chroma에 추가
        add_web_pages_json_to_chroma(json_path)

    #⑪ (11) task 완료
    tasks[-1].done = True
    tasks[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    #⑪ (11) 새로운 task 추가
    task_desc = "AI팀이 쓸 책의 세부 목차를 결정하기 위한 정보를 벡터 검색을 통해 찾아낸다."
    task_desc += f" 다음 항목이 새로 추가되었다\n: {queries}"
    
    new_task = Task(
        agent="vector_search_agent",
        done=False,
        description=task_desc,
        done_at=""
    )

    tasks.append(new_task)

    #⑫ (12) 작업 후기 메시지
    msg_str = f"[WEB SEARCH AGENT] 다음 질문에 대한 검색 완료: {queries}"
    messages.append(AIMessage(msg_str))

    #⑬ (13) state 업데이트
    return {
        "messages": messages,
        "task_history": tasks
    }


# 사용자와 대화할 노드(agent): communicator
def communicator(state: State):
    print("\n\n============ COMMUNICATOR ============")

    # 시스템 프롬프트 정의
    communicator_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI팀의 커뮤니케이터로서, 
        AI팀의 진행상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위한 대화를 나눈다. 

        사용자도 outline(목차)을 이미 보고 있으므로, 다시 출력할 필요는 없다.
        outline: {outline} 
        --------------------------------
        messages: {messages}
        """
    )

    #② 시스템 프롬프트와 모델을 연결
    system_chain = communicator_system_prompt | llm

    # 상태에서 메시지를 가져옴
    messages = state["messages"]

    # 입력값 정의
    inputs = {
        "messages": messages,
        "outline": get_outline(current_path)
    }

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

    task_history = state.get("task_history", []) 
    if task_history[-1].agent != "communicator":
        raise ValueError(f"Communicator가 아닌 agent가 대화를 시도하고 있습니다.\n {task_history[-1]}")
    
    task_history[-1].done = True
    task_history[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    return {
        "messages": messages,
        "task_history": task_history
    }


# 상태 그래프 정의
graph_builder = StateGraph(State)

# Nodes
graph_builder.add_node("supervisor", supervisor)     
graph_builder.add_node("communicator", communicator)
graph_builder.add_node("content_strategist", content_strategist)
graph_builder.add_node("vector_search_agent", vector_search_agent)
graph_builder.add_node("web_search_agent", web_search_agent)

# Edges
graph_builder.add_edge(START, "supervisor")
graph_builder.add_conditional_edges(
    "supervisor", 
    supervisor_router,
    {
        "content_strategist": "content_strategist",
        "communicator": "communicator",
        "vector_search_agent": "vector_search_agent", 
        "web_search_agent": "web_search_agent"
    }
)
graph_builder.add_edge("content_strategist", "communicator")
graph_builder.add_edge("web_search_agent", "vector_search_agent") #③
graph_builder.add_edge("vector_search_agent", "communicator")
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
    task_history=[]
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
