import os
from openai import OpenAI
from dotenv import load_dotenv
import time

load_dotenv()

class AgentClient:
    def __init__(self):
        self.api_key = os.getenv("API_KEY", "EMPTY")
        self.base_url = os.getenv("BASE_URL", "http://localhost:8000/v1")
        self.model_name = os.getenv("MODEL_NAME", "gpt-4") 
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def request_llm(self, messages, max_retries=3):
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0,  
                    max_tokens=1024,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"请求失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                time.sleep(2)  
        return None
