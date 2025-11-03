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
    {"role": "user", "content": "비둘기"},
  ]		
)
print(response.choices[0].message.content) 
