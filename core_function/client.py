import os
from openai import OpenAI
from dotenv import load_dotenv
import time

# 加载环境变量（存储你的 API_KEY 和 BASE_URL）
load_dotenv()

class AgentClient:
    def __init__(self):
        """
        初始化客户端。
        支持本地 vLLM 部署的模型或在线 API。
        """
        self.api_key = os.getenv("API_KEY", "EMPTY")
        self.base_url = os.getenv("BASE_URL", "http://localhost:8000/v1")
        self.model_name = os.getenv("MODEL_NAME", "gpt-4")  # 可根据需要调整模型名称
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def request_llm(self, messages, max_retries=3):
        """
        向大模型发送请求，包含基础的重试机制。
        """
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0,  # 重点：科研评测通常设为0以保证结果可复现
                    max_tokens=1024,
                    # stop=["</call>"] # 可选：在解析到调用结束符时停止，节省 token
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"请求失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                time.sleep(2)  # 等待后重试
        return None

# 测试代码
if __name__ == "__main__":
    client = AgentClient()
    test_messages = [
        {"role": "system", "content": "你是一个严谨的科研助手。"},
        {"role": "user", "content": "请确认你的通信状态，并简述你对 HotpotQA 多跳推理任务的理解。"}
    ]
    result = client.request_llm(test_messages)
    print(f"模型回复：\n{result}")