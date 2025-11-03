import pymupdf
import os

# ① PDF 문서를 PyMuPDF로 열기
pdf_file_path = "chap04/data/인격에 대한 칸트의 관점과 인공지능.pdf"
doc = pymupdf.open(pdf_file_path)

full_text = ''

# ② 문서 페이지 반복하면서 페이지 내용 추출하기
for page in doc: # 문서 페이지 반복
    text = page.get_text() # 페이지 텍스트 추출
    full_text += text

# ③ PDF 파일명 추출
pdf_file_name = os.path.basename(pdf_file_path)
pdf_file_name = os.path.splitext(pdf_file_name)[0] # 확장자 제거

# ④ 텍스트 파일로 저장
txt_file_path = f"chap04/output/{pdf_file_name}.txt"
with open(txt_file_path, 'w', encoding='utf-8') as f:
    f.write(full_text)
