from glob import glob
from openai import OpenAI
from dotenv import load_dotenv
import os
import base64

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") 
client = OpenAI(api_key=api_key)

# 이미지 파일을 base64로 인코딩하는 함수
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    

def image_quiz(image_path):
    base64_image = encode_image(image_path) # 이미지를 base64로 인코딩

    quiz_prompt = """
    제공된 이미지를 바탕으로, 다음과 같은 양식으로 퀴즈를 만들어주세요. 
    정답은 1~4 중 하나만 해당하도록 출제하세요.
    아래는 예시입니다. 
    ----- 예시 -----

    Q: 다음 이미지에 대한 설명 중 옳지 않은 것은 무엇인가요?
    - (1) 베이커리에서 사람들이 빵을 사고 있는 모습이 담겨 있습니다.
    - (2) 맨 앞에 서 있는 사람은 빨간색 셔츠를 입고 있습니다.
    - (3) 기차를 타기 위해 줄을 서 있는 사람들이 있습니다.
    - (4) 점원은 노란색 티셔츠를 입고 있습니다.
        
    정답: (4) 점원은 노란색 티셔츠가 아닌 파란색 티셔츠를 입고 있습니다.
    (주의: 정답은 1~4 중 하나만 선택되도록 출제하세요.)
    ======
    """

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": quiz_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",  # 응답 생성에 사용할 모델을 지정
        messages=messages # 대화 기록을 입력으로 전달
    )

    return response.choices[0].message.content

# 이미지 경로를 받아 퀴즈를 만드는 함수
q = image_quiz("./chap06/data/images/busan_dive.jpg")
print(q)



txt = '' # 문제들을 계속 붙여 나가기 위해 빈 문자열 선언
no = 1 # 문제 번호를 위해 선언
for g in glob('./chap06/data/images/*.jpg'): 
    try:
        q = image_quiz(g) # 문제 출제 (가끔 GPT에서 에러 발생할 수 있으므로 try문 사용)
    except Exception as e:
        print(e)
        continue

    divider = f'## 문제 {no}\n\n'
    print(divider)
    
    txt += divider  
    # 파일명 추출해 이미지 링크 만들기
    filename = os.path.basename(g) # ③ 마크다운에 표시할 이미지 파일 경로 설정   
    txt += f'![image]({filename})\n\n' 

    # 문제 추가
    print(q)
    txt += q + '\n\n---------------------\n\n'
    # ④ 마크다운 파일로 저장
    with open('./chap06/data/images/image_quiz.md', 'w', encoding='utf-8') as f:
        f.write(txt)
    
    no += 1 # 문제 번호 증가

