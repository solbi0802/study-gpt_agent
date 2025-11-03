from openai import OpenAI
from dotenv import load_dotenv
import os
import pymupdf

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

def pdf_to_text(pdf_file_path: str):
    pdf_file_path = "chap04/data/과정기반 작물모형을 이용한 웹 기반 밀 재배관리 의사결정 지원시스템 설계 및 구축.pdf"
    doc = pymupdf.open(pdf_file_path)

    header_height = 80
    footer_height = 80

    full_text = ''

    for page in doc:
        rect = page.rect # 페이지 크기 가져오기
        
        header = page.get_text(clip=(0, 0, rect.width , header_height))
        footer = page.get_text(clip=(0, rect.height - footer_height, rect.width , rect.height))
        text = page.get_text(clip=(0, header_height, rect.width , rect.height - footer_height))

        full_text += text + '\n------------------------------------\n'

    # 파일명만 추출
    pdf_file_name = os.path.basename(pdf_file_path)
    pdf_file_name = os.path.splitext(pdf_file_name)[0] # 확장자 제거

    txt_file_path = f'chap04/data/{pdf_file_name}_with_preprocessing.txt'

    with open(txt_file_path, 'w', encoding='utf-8') as f:
        f.write(full_text)

    return txt_file_path


def summarize_txt(file_path: str): # ①
    client = OpenAI(api_key=api_key)

    # ② 주어진 텍스트 파일을 읽어들인다.
    with open(file_path, 'r', encoding='utf-8') as f:
        txt = f.read()

    # ③ 요약을 위한 시스템 프롬프트를 생성한다.
    system_prompt = f'''
    너는 다음 글을 요약하는 봇이다. 아래 글을 읽고, 저자의 문제 인식과 주장을 파악하고, 주요 내용을 요약하라. 

    작성해야 하는 포맷은 다음과 같다. 
    
    # 제목

    ## 저자의 문제 인식 및 주장 (15문장 이내)
    
    ## 저자 소개

    
    =============== 이하 텍스트 ===============

    { txt }
    '''

    print(system_prompt)
    print('=========================================')

    # ④ OpenAI API를 사용하여 요약을 생성한다.
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
        ]
    )

    return response.choices[0].message.content

def summarize_pdf(pdf_file_path: str, output_file_path: str):
    txt_file_path = pdf_to_text(pdf_file_path)
    summary = summarize_txt(txt_file_path)

    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(summary)


if __name__ == '__main__':
    pdf_file_path = "chap04/data/llm논문.pdf"
    summarize_pdf(pdf_file_path, 'chap04/output/llm_summary2.txt')