from pydantic import BaseModel, Field
from typing import Literal

class Task(BaseModel):
    agent: Literal[
        "content_strategist",
        "communicator",
        "web_search_agent",
        "vector_search_agent",
    ] = Field(
        ...,
        description="""
        작업을 수행하는 agent의 종류.
        - content_strategist: 콘텐츠 전략을 수립하는 작업을 수행한다. 사용자의 요구사항이 명확해졌을 때 사용한다. AI 팀의 콘텐츠 전략을 결정하고, 전체 책의 목차(outline)를 작성한다. 
        - communicator: AI 팀에서 해야 할 일을 스스로 판단할 수 없을 때 사용한다. 사용자에게 진행상황을 사용자에게 보고하고, 다음 지시를 물어본다.
        - web_search_agent: 웹 검색을 통해 목차(outline) 작성에 필요한 정보를 확보한다.
        - vector_search_agent: 벡터 DB 검색을 통해 목차(outline) 작성에 필요한 정보를 확보한다.
        """
    )
	
    done: bool = Field(..., description="종료 여부")
    description: str = Field(..., description="어떤 작업을 해야 하는지에 대한 설명")
	
    done_at: str = Field(..., description="할 일이 완료된 날짜와 시간")
	
    def to_dict(self):
        return {
            "agent": self.agent,
            "done": self.done,
            "description": self.description,
            "done_at": self.done_at
        }  
