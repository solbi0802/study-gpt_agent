import pymupdf
import os

pdf_file_path = "chap04/data/인격에 대한 칸트의 관점과 인공지능.pdf"
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

txt_file_path = f'chap04/output/{pdf_file_name}_with_preprocessing.txt'

with open(txt_file_path, 'w', encoding='utf-8') as f:
    f.write(full_text)
