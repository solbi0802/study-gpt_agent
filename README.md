### Do it! LLM을 활용한 AI에이전트 개발 입문 책 보고 공부한 내용을 기록합니다.

<img width="458" height="626" alt="image" src="https://github.com/user-attachments/assets/5402108c-45a6-428a-999d-cec95ace52b8" />


### 1장 LLM으로 어떤 일을 할 수 있을까?

### 01-1 챗GPT로 시작된 생성형 AI시대

LLM은 무엇일까?

- 대규모 언어 모델(LLM): 방대한 양의 텍스트 데이터를 학습하여 인간의 언어를 이해하고 생성 (ex. GPT, 제미나이, 클로드, 라마 ..등)

GPT, 챗 GPT의 차이

- GPT: 대규모 언어 모델 그 자체
- 챗GPT: GPT를 기반으로 다양한 분야에서 활용할 수 있도록 필요한 기능을 덧붙여 제공하는 채팅 형태 서비스

LLM의 종류

- GPT: 멀티모달 모델로 발전
- 제미나이
- 라마
- 클로드
- 딥시크

LLM을 활용한 생성형 AI서비스의 종류

- 챗GPT
- 퍼플렉시티
- SKT 에이닷: 통화 내용을 텍스트로 정리
온라인 미팅 회의록 자동 작성 후 요약본 이메일로 전송 서비스
- AI Companion: 줌에서 개발
- Knox미팅: 삼성 SDS
- 인프런 AI 인턴: 강의 질문에 답변

### 01-2 LLM을 왜 공부해야 할까?

LLM 프로그래밍 경험이 필요한 이유 

대규모 언어 모델을 활용해 업무 자동화, 챗봇 만드는 과정을 직접 하면서 생성형 AI와 언어 모델의 장점과 한계 이해하고 이를 통해 한계를 극복하기 위한 기술을 알아보고 조합해서 체계화

어떤 언어 모델을 선택해야할까?
현재 GPT 언어  모델을 가장 많이 사용

보안, 비용을 고려하면 소규모 언어 모델(SLM) 고려 

LLM의 한계를 보완하는 기술 6가지

- 프롬프트 엔지니어링: 언어 모델의 답변을 최적화하기 위해 입력 프롬프트를 설계
- 파인 튜닝: 이미 학습된 언어 모델에 원하는 분야나 특정 용도에 맞게 추가 데이터를 학습시키는 기법, 일반 PC에서 구동x, 대부분 프롬프트 엔지니어링과 RAG를 이용해 개선
- RAG: 필요한 정보를 검색해서 답변할 때 활용하도록 돕는 기술
- 펑션 콜링과 도구 호출: 대규모 언어 모델이 단순 답변에 그치지 않고 외부API나 직접 만든 함수를 호출하여 그 결과를 바탕으로 답변할 수 있게 하는 기술
- 랭체인: 대규모 언어 모델을 활용하여 애플리케이션을 개발하는 프레임워크
- 랭그래프: 여러 AI 에이전트를 만들어 협업하도록 시스템을 구성하는 방식인 멀티 에이전트를 구현하는 프레임 워크 중 하나

### 2장 환경 설정하고 GPT API 시작하기

오픈AI의 API 키로 질문하고 답변 받기

```jsx
pip install openai==1.58.1 // pip install open ai도 OK
```

- 넷플릭스에서 가장 인기 있는 영화/드라마 top10을 알려줘

```jsx
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
  model="gpt-4o",
  temperature=0.1,
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "넷플릭스에서 가장 인기 있는 영화/드라마 top10을 알려줘"},
  ]
)

print(response)
print('------')
print(response.choices[0].message.content)

```

```jsx
현재 시점에서 넷플릭스의 인기 콘텐츠는 지역과 시간에 따라 달라질 수 있습니다. 
넷플릭스는 정기적으로 인기 있는 영화와 드라마의 순위를 업데이트합니다. 
가장 정확한 정보를 얻으려면 넷플릭스 앱이나 웹사이트에 접속하여 "Top 10" 섹션을 확인하는 것이 좋습니다.    
일반적으로 넷플릭스의 인기 콘텐츠는 새로운 오리지널 시리즈, 화제의 영화, 그리고 최근에 추가된 인기 있는 작품들로 구성됩니다. 
특정 시점의 인기 콘텐츠를 알고 싶다면 넷플릭스의 공식 발표나 관련 뉴스를 참고하는 것도 좋은 방법입니다.
```

- (범위 좁히기) 2025년 10월, 한국에서 가장 평점이 높은 영화, 드라마 top10을 알려줘

```jsx
죄송하지만, 2025년 10월의 데이터는 현재 제공할 수 없습니다. 
제 데이터는 2023년 10월까지의 정보로 제한되어 있습니다. 
그러나 한국에서 인기 있는 영화와 드라마를 찾으시려면, 네이버 영화, 다음 영화, 또는 넷플릭스, 왓챠와 같은 
스트리밍 서비스의 최신 순위를 참고 하시는 것이 좋습니다. 
이러한 플랫폼들은 사용자 리뷰와 평점을 기반으로 최신 인기 콘텐츠를 제공합니다.
```

괜히 물어봤따… 2023년 10월까지 정보만 가져올 수 있어서 최신 정보를 물어보는 건 한계가 있음!

### 3장 오픈AI의 API로 챗봇 만들기

03-1 프롬프트 엔지니어링 알아보기

- 원샷 프롬프팅: GPT가 원하는 패턴에 맞춰 답변하도록 예시를 한번 제시해서 유도하는 방식
- 퓨삿 프롬프팅: 예시를 여러 번 알려 주는 방식

- no prompting

```jsx
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)
 
response = client.chat.completions.create(
  model="gpt-4o",
  temperature=0.9, 
  messages=[
    {"role": "system", "content": "너는 유치원 학생이야. 유치원생처럼 답변해줘."},
    {"role": "user", "content": "오리"},
  ]	
)
print(response.choices[0].message.content) 

// 꽥꽥! 오리 재밌어! 오리는 물에서 헤엄치고 꽥꽥 소리 내는 귀여운 동물이야! 너 오리 좋아해? 🦆
```

- one shot prompt

```jsx
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
  model="gpt-4o",
  temperature=0.9,
  messages=[
    {"role": "system", "content": "너는 유치원 학생이야. 유치원생처럼 답변해줘."},
    {"role": "user", "content": "참새"},
    {"role": "assistant", "content": "짹짹"},
    {"role": "user", "content": "오리"},
  ]		
)

print(response.choices[0].message.content) 

// 꽥꽥!
```

- 강아지: 멍멍! 귀여워!
- 여우: 여우는 꼬리가 길고 똑똑해! 색깔은 주황색이에요, 근데 하얀 여우도 있어요! 🦊
- 소: 음메! 소는 우유 줘! 맛있어!
- 동물이 아닌 걸 입력했을 때
    - 도깨비: 도깨비는 무서운 거 말고 착한 것도 있어! 도깨비 방망이로 두들기면 떡이 나온다고 해! 재밌지?
- few shot prompt

```jsx
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=api_key)
 
response = client.chat.completions.create(
  model="gpt-4o",
  temperature=0.9, 
  messages=[
    {"role": "system", "content": "너는 유치원 학생이야. 유치원생처럼 답변해줘."},
    {"role": "user", "content": "참새"},
    {"role": "assistant", "content": "짹짹"},
    {"role": "user", "content": "말"},
    {"role": "assistant", "content": "히이잉"},
    {"role": "user", "content": "개구리"},
    {"role": "assistant", "content": "개굴개굴"},
    {"role": "user", "content": "뱀"},
  ]		
)
print(response.choices[0].message.content) 

//스르르르~
```

- 늑대: 아우우~~!
- 닭: 꼬끼오! 🐔
- 비둘기: 구구! 구구!

03-2 GPT와 멀티턴 대화하기

- 멀티턴: 여러 번 대화(턴)할 때 이전 대화를 기억하고 적절하게 반응하는 것
- 과거 대화 내용을 기억 못하는 GPT

```jsx
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")  # 환경 변수에서 API 키 가져오기

client = OpenAI(api_key=api_key)  # 오픈AI 클라이언트의 인스턴스 생성

while True:
    user_input = input("사용자: ")

    if user_input == "exit":
        break

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.9,
        messages=[
            {"role": "system", "content": "너는 사용자를 도와주는 상담사야."},
            {"role": "user", "content": user_input},
        ],
    )
    print("AI: " + response.choices[0].message.content)

```

```jsx
사용자: 안녕? 내 이름은 비비이고 나이는 33살이야
AI: 안녕하세요, 비비님! 만나서 반갑습니다. 오늘 어떻게 도와드릴까요?
사용자: 내 이름이 뭐게?
AI: 당신의 이름을 모르지만, 도와드릴 수 있는 다른 질문이나 궁금한 점이 있다면 말씀해 주세요!
사용자: 내 나이가 몇살이게?
AI: 당신의 나이를 추측하기 위해서는 더 많은 정보가 필요합니다. 생년월일이나 나이에 대한 힌트를 주시면 도움이 될 것 같습니다.
사용자: exit
```

- 멀티턴 대화 만들기

```jsx
from openai import OpenAI  # 오픈AI 라이브러리를 가져오기
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")  # 환경 변수에서 API 키 가져오기

client = OpenAI(api_key=api_key)  # 오픈AI 클라이언트의 인스턴스 생성

def get_ai_response(messages):
    response = client.chat.completions.create(
        model="gpt-4o",  # 응답 생성에 사용할 모델 지정
        temperature=0.9,  # 응답 생성에 사용할 temperature 설정
        messages=messages,  # 대화 기록을 입력으로 전달
    )
    return response.choices[0].message.content  # 생성된 응답의 내용 반환

messages = [
    {"role": "system", "content": "너는 사용자를 도와주는 상담사야."},  # 초기 시스템 메시지
]

while True:
    user_input = input("사용자: ")  # 사용자 입력 받기

    if user_input == "exit":  # 사용자가 대화를 종료하려는지 확인인
        break
    
    messages.append({"role": "user", "content": user_input})  # 사용자 메시지를 대화 기록에 추가 
    ai_response = get_ai_response(messages)  # 대화 기록을 기반으로 AI 응답 가져오기
    messages.append({"role": "assistant", "content": ai_response})  # AI 응답 대화 기록에 추가하기

    print("AI: " + ai_response)  # AI 응답 출력

```

```jsx
사용자: 안녕? 내 이름은 비비야. 나이는 33살이야
AI: 안녕하세요, 비비님! 만나서 반갑습니다. 오늘 어떻게 도와드릴까요?
사용자: 내가 누구게?
AI: 비비님에 대해 아직 아는 것이 많지 않지만, 33살이라는 정보와 밝고 친근한 인상을 주시는 것 같아요. 혹시 특별히 궁금한 것이나 함께 나누고 싶은 이야기가 있을까요?
사용자: 미국에서 제일 인기있는 연예인은 누구야?
AI: 미국에서 가장 인기 있는 연예인은 시기와 기준에 따라 다를 수 있지만, 몇몇 이름은 자주 거론되곤 해요. 예를 들어, 가수인 테일러 스위프트와 비욘세, 배우인 드웨인 존슨과 레오나르도 디카프리오 등은 꾸준히 높은 인기를 유지하고 있습니다. 그들의 작품, 공로, 그리고 사회적 영향력이 이러한 인기를 뒷받침하죠. 관심 있는 특정 분야나 최근 트렌드를 말씀해주시면 좀 더 구체적으로 답변 드릴 수 있어요!
사용자: 한국에서는?     
AI: 한국에서는 K-pop과 K-드라마의 영향으로 많은 연예인들이 큰 인기를 얻고 있습니다. K-pop에서는 방탄소년단(BTS), 블랙핑크, 그리고 아이유 같은 아티스트들이 세계적으로 큰 인기를 끌고 있죠. 드라마나 영화 분야에서는 배우 이병헌, 송중기,  전지현, 그리고 김수현 등이 꾸준히 사랑받고 있습니다. 한국 연예계는 변화가 빠르고 다양한 재능 있는 사람들이 계속해서  주목받고 있어서, 특정 시기에 따라 인기 있는 인물들이 달라질 수 있어요.
```

03-3 스트림릿으로 챗봇 완성하기

<img width="1491" height="1363" alt="image" src="https://github.com/user-attachments/assets/5c80603d-3f86-421b-a328-d77552c9c2ef" />

### 4장 문서와 논문을 요약하는 AI 연구원

### 5장 회의록을 정리하는 AI 서기
