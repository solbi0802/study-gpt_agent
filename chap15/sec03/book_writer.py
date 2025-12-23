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
    user_request: str 
    ai_recommendation: str # AI의 추천을 저장하는 변수
    supervisor_call_count: int # supervisor 호출 횟수를 저장하는 변수


def business_analyst(state: State): #
    print("\n\n============ BUSINESS ANALYST ============")

    #② (1) 시스템 프롬프트 정의
    business_analyst_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI팀의 비즈니스 애널리스트로서, 
        AI팀의 진행상황과 "사용자 요구사항"을 토대로,
        현 시점에서 'ai_recommendation'과 최근 사용자의 발언을 바탕으로 요구사항이 무엇인지 판단한다.
        지난 요청사항이 달성되었는지 판단하고, 현 시점에서 어떤 작업을 해야 하는지 결정한다.

        다음과 같은 템플릿 형태로 반환한다. 
        ```
        - 목표: OOOO \n 방법: OOOO
        ```

        ------------------------------------
        *AI 추천(ai_recommendation)* : {ai_recommendation}
        ------------------------------------
        사용자 최근 발언: {user_last_comment}
        ------------------------------------
        참고자료: {references}
        ------------------------------------
        목차 (outline): {outline}
        ------------------------------------
        "messages": {messages}
        """
    )
    
    #③ (2) 시스템 프롬프트와 모델을 연결
    ba_chain = business_analyst_system_prompt | llm | StrOutputParser()

    #(3) 상태에서 메시지를 가져옴
    messages = state["messages"]

    # (3) 사용자의 마지막 발언을 가져옴
    user_last_comment = None
    for m in messages[::-1]:
        if isinstance(m, HumanMessage):
            user_last_comment = m.content
            break

    # (3) 입력 자료 준비
    inputs = {
        "ai_recommendation": state.get("ai_recommendation", None),
        "previous_user_request": state.get("user_request", None),
        "references": state.get("references", {"queries": [], "docs": []}),
        "outline": get_outline(current_path),
        "messages": messages,
        "user_last_comment": user_last_comment
    }

    #⑤ (4) 시스템 프롬프트를 통해 사용자 요구사항을 분석
    user_request = ba_chain.invoke(inputs)

    #⑥ (5) businessage analyst의 결과를 메시지에 추가
    business_analyst_message = f"[Business Analyst] {user_request}"
    print(business_analyst_message)
    messages.append(AIMessage(business_analyst_message))

    save_state(current_path, state) #⑦ (6) 현재 state 내용 저장

    return {
        "messages": messages,
        "user_request": user_request,
        "ai_recommendation": ""
    }


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
        - web_search_agent: vector_search_agent를 시도하고, 검색 결과(references)에 필요한 정보가 부족한 경우 사용한다. 웹 검색을 통해 해당 정보를 Vector DB에 보강한다. 
        - vector_search_agent: 목차 작성을 위해 필요한 자료를 확보하기 위해 벡터 DB 검색을 한다. 

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

    supervisor_call_count = state.get("supervisor_call_count", 0)

    if supervisor_call_count > 2:
        print("Supervisor 호출 횟수 초과: Communicator 호출")
        task = Task(
            agent="communicator", 
            done=False, 
            description="supervisor 호출 횟수 초과했으므로, 현재까지의 진행상황을 사용자에게 보고한다. ",
            done_at="", 
        )
    else:
        task = supervisor_chain.invoke(inputs)

    task_history = state.get("task_history", [])    # 작업 이력 가져오기
    task_history.append(task)                    	# 작업 이력에 추가
   
    # 메시지 추가
    supervisor_message = AIMessage(f"[Supervisor] {task}")
    messages.append(supervisor_message)
    print(supervisor_message.content)

    # state 업데이트
    return {
        "messages": messages, 
        "task_history": task_history, 
        "supervisor_call_count": supervisor_call_count + 1
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

    # # communicator로 보내는 Task 생성 코드 삭제
    # new_task = Task(
    #     agent="communicator",
    #     done=False,
    #     description="AI팀의 진행상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위한 대화를 나눈다",
    #     done_at=""
    # )
    # tasks.append(new_task)

    # vector search agent의 작업후기를 메시지로 생성
    msg_str = f"[VECTOR SEARCH AGENT] 다음 질문에 대한 검색 완료: {queries}"
    message = AIMessage(msg_str)
    print(msg_str)

    messages.append(message)
    ai_recommendation = "현재 참고자료(references)가 목차(outline)를 개선하는데 충분한지 확인하라. 충분하다면 content_strategist로 목차 작성을 하라. "
    
    # state 업데이트
    return {
        "messages": messages,
        "task_history": tasks,
        "references": references,
        "ai_recommendation": ai_recommendation
    }


# 목차를 작성하는 노드(agent)
def content_strategist(state: State):
    print("\n\n============ CONTENT STRATEGIST ============")
    
    task_history = state.get("task_history", []) # 작업 이력 가져오기 (아래에 있던 코드 위로 이동)
    task = task_history[-1]
    if task.agent != "content_strategist":
        raise ValueError(f"Content Strategist가 아닌 agent가 목차 작성을 시도하고 있습니다.\n {task}")
    
    # 시스템 프롬프트 정의
    content_strategist_system_prompt = PromptTemplate.from_template(
        """
        너는 책을 쓰는 AI팀의 콘텐츠 전략가(Content Strategist)로서,
        이전 대화 내용을 바탕으로 사용자의 요구사항을 분석하고, AI팀이 쓸 책의 세부 목차를 결정한다.

        지난 목차가 있다면 그 버전을 사용자의 요구에 맞게 수정하고, 없다면 새로운 목차를 제안한다.
        목차를 작성하는데 필요한 정보는 "참고 자료"에 있으므로 활용한다. 
        
        다음 정보를 활용하여 목차를 작성하라. 
        - 사용자 요구사항(user_request)
        - 작업(task)
        - 검색 자료 (references)
        - 기존 목차 (previous_outline)
        - 이전 대화 내용(messages)

        너의 작업 목표는 다음과 같다:
        1. 만약 "기존 목차 구조 (previous_outline)"이 존재한다면, 사용자의 요구사항을 토대로 "기존 목차 구조"에서 어떤 부분을 수정하거나 추가할지 결정한다.
        - "이번 목차 작성의 주안점"에 사용자 요구사항(user_request)을 충족시키는 것을 명시해야 한다.
        2. 책의 전반적인 구조(chapter, section)를 설계하고, 각 chpater와 section의 제목을 정한다.
        3. 책의 전반적인 세부구조(chapter, section, sub-section)를 설계하고, sub-section 하부의 주요 내용을 리스트 형태로 정리한다.
        4. 목차의 논리적인 흐름이 사용자 요구를 충족시키는지 확인한다.
        5. 참고자료 (references)를 적극 활용하여 근거에 기반한 목차를 작성한다.
        6. 참고문헌은 반드시 참고자료(references) 자료를 근거로 작성해야 하며, 최대한 풍부하게 준비한다. URL은 전체 주소를 적어야 한다.
        7. 추가 자료나 리서치가 필요한 부분을 파악하여 supervisor에게 요청한다.

        사용자 요구사항(user_request)을 최우선으로 반영하는 목차로 만들어야 한다. 

        --------------------------------
        - 사용자 요구사항(user_request): 
        {user_request}
        --------------------------------
        - 작업(task): 
        {task}
        --------------------------------
        - 참고 자료 (references)
        {references}
        --------------------------------
        - 기존 목차 (previous_outline)
        {outline}
        --------------------------------
        - 이전 대화 내용(messages)
        {messages}
        --------------------------------

        작성 형식 아래 양식을 지키되 하부 항목으로 더 세분화해도 좋다. 목차(outline) 양식의 챕터, 섹션 등 항목의 갯수는 필요한만큼 추가하라. 
        섹션 갯수는 최소 2개 이상이어야 하며, 더 많으면 좋다. 

        outline_template은 예시로 앞부분만 제시한 것이다. 각 장은 ':---CHAPTER DIVIDER---:'로 구분한다.
        outline_template:
        {outline_template}

        사용자가 추가 피드백을 제공할 수 있도록 논리적인 흐름과 주요 목차 아이디어를 제안하라.    
        """
    )

    # 시스템 프롬프트와 모델을 연결
    content_strategist_chain = content_strategist_system_prompt | llm | StrOutputParser()

    user_request = state.get("user_request", "") # 사용자 요구사항 가져오기

    messages = state["messages"]        # 상태에서 메시지를 가져옴
    outline = get_outline(current_path) # 저장된 목차를 가져옴

    with open(f"{current_path}/templates/outline_template.md", "r", encoding='utf-8') as f:
        outline_template = f.read()  

    # 입력값 정의
    inputs = {
        "user_request": user_request,  #사용자 요구사항 user_request
        "task": task,
        "messages": messages,
        "outline": outline, 
        "references": state.get("references", {"queries": [], "docs": []}),
        "outline_template": outline_template # 템플릿 이용하기
    }

    # 목차 작성
    gathered = ''
    for chunk in content_strategist_chain.stream(inputs):
        gathered += chunk
        print(chunk, end='')

    print()

    save_outline(current_path, gathered) # 목차 저장

    # 템플릿을 활용한 작업 후기 메시지 찾기
    if '-----: DONE :-----' in gathered:
        review = gathered.split('-----: DONE :-----')[1]
    else: 
        review = gathered[-200:]


    # 메시지 추가    
    content_strategist_message = f"[Content Strategist] 목차 작성 완료: outline 작성 완료\n {review}"
    print(content_strategist_message)
    messages.append(AIMessage(content_strategist_message))

    # task_history = state.get("task_history", []) # task_history 가져오기
    # # 최근 task 작업완료(done) 처리하기
    # if task_history[-1].agent != "content_strategist": 
    #     raise ValueError(f"Content Strategist가 아닌 agent가 목차 작성을 시도하고 있습니다.\n {task_history[-1]}")
    
    task_history[-1].done = True
    task_history[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # # 다음 작업이 communicator로 사용자와 대화하는 것이므로 새 작업 추가 
    # new_task = Task(
    #     agent="communicator",
    #     done=False,
    #     description="AI팀의 진행상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위한 대화를 나눈다",
    #     done_at=""
    # )
    # task_history.append(new_task)
    # print(new_task)

    # 현재 state를 업데이트한다. 
    return {
        "messages": messages,
        "task_history": task_history
    }


def outline_reviewer(state: State): # (0)
    print("\n\n============ OUTLINE REVIEWER ============")

    # (1) 시스템 프롬프트 정의
    outline_reviewer_system_prompt = PromptTemplate.from_template(
        """
        너는 AI팀의 목차 리뷰어로서, AI팀이 작성한 목차(outline)를 검토하고 문제점을 지적한다. 

        - outline이 사용자의 요구사항을 충족시키는지 여부
        - outline의 논리적인 흐름이 적절한지 여부
        - 근거에 기반하지 않은 내용이 있는지 여부
        - 주어진 참고자료(references)를 충분히 활용했는지 여부
        - 참고자료가 충분한지, 혹은 잘못된 참고자료가 있는지 여부
        - example.com 같은 더미 URL이 있는지 여부: 
        - 실제 페이지 URL이 아닌 대표 URL로 되어 있는 경우 삭제 해야함: 어떤 URL이 삭제되어야 하는지 명시하라.
        - 기타 리뷰 사항

        그 분석 결과를 설명하고, 다음 어떤 작업을 하면 좋을지 제안하라.
        
        - 분석결과: outline이 사용자의 요구사항을 충족시키는지 여부
        - 제안사항: (vector_search_agent, communicator 중 어떤 agent를 호출할지)

        ------------------------------------------
        user_request: {user_request}
        ------------------------------------------
        references: {references}
        ------------------------------------------
        outline: {outline}
        ------------------------------------------
        messages: {messages}
        """
    )
    # (2) inputs에 들어갈 내용 정리    
    user_request = state.get("user_request", None)
    outline = get_outline(current_path)
    references = state.get("references", {"queries": [], "docs": []})
    messages = state.get("messages", [])

    inputs = {
        "user_request": user_request,
        "outline": outline,
        "references": references,
        "messages": messages
    }

    # 시스템 프롬프트와 모델을 연결
    outline_reviewer_chain = outline_reviewer_system_prompt | llm

    # (3) 목차 리뷰
    review = outline_reviewer_chain.stream(inputs)

    gathered = None

    for chunk in review:
        print(chunk.content, end='')

        if gathered is None:
            gathered = chunk
        else:
            gathered += chunk
    # (4) outline_review 에이전트의 작업 후기를 메시지에 추가
    if '[OUTLINE REVIEW AGENT]' not in gathered.content:
        gathered.content = f"[OUTLINE REVIEW AGENT] {gathered.content}"

    print(gathered.content)
    messages.append(gathered)

    # (5) ai_recommendation은 목차 리뷰 결과를 사용
    ai_recommendation = gathered.content

    return {"messages": messages, "ai_recommendation": ai_recommendation}

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
        "task_history": task_history,
        "supervisor_call_count": 0
    }


# 상태 그래프 정의
graph_builder = StateGraph(State)

# Nodes
graph_builder.add_node("business_analyst", business_analyst)
graph_builder.add_node("supervisor", supervisor)     
graph_builder.add_node("communicator", communicator)
graph_builder.add_node("content_strategist", content_strategist)
graph_builder.add_node("outline_reviewer", outline_reviewer)
graph_builder.add_node("vector_search_agent", vector_search_agent)
graph_builder.add_node("web_search_agent", web_search_agent)

# Edges
graph_builder.add_edge(START, "business_analyst")
graph_builder.add_edge("business_analyst", "supervisor")
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

graph_builder.add_edge("content_strategist", "outline_reviewer")
graph_builder.add_edge("outline_reviewer", "business_analyst")
graph_builder.add_edge("web_search_agent", "vector_search_agent") #③
graph_builder.add_edge("vector_search_agent", "business_analyst")
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
    task_history=[], 
    references={"queries": [], "docs": []}, 
    user_request=""
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
