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
    {"role": "user", "content": "2025년 10월, 한국에서 가장 평점이 높은 영화, 드라마 top10을 알려줘"},
  ]
)

print(response)
print('------')
print(response.choices[0].message.content)
